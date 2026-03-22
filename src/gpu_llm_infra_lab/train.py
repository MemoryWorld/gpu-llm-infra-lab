from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.checkpoint import checkpoint
from torch.utils.data import DataLoader, DistributedSampler

from .data_loader import CharLMBlockDataset, load_corpus
from .tiny_gpt import Block, TinyGPT


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_lr(it: int, warmup_iters: int, max_iters: int, base_lr: float) -> float:
    if it < warmup_iters:
        return base_lr * (it + 1) / max(1, warmup_iters)
    decay_ratio = (it - warmup_iters) / max(1, max_iters - warmup_iters)
    return 0.5 * base_lr * (1.0 + math.cos(math.pi * decay_ratio))


def ddp_setup() -> tuple[int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return 0, 1, 0
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    # NCCL is not available on Windows; gloo works for CPU/some GPU multi-proc setups.
    backend = "nccl" if torch.cuda.is_available() and sys.platform != "win32" else "gloo"
    dist.init_process_group(backend=backend)
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def ddp_cleanup() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def build_model_with_checkpointing(
    cfg: dict, vocab_size: int, use_checkpointing: bool, device: torch.device
) -> TinyGPT:
    mcfg = cfg["model"]
    model = TinyGPT(
        vocab_size=vocab_size,
        block_size=mcfg["block_size"],
        n_layer=mcfg["n_layer"],
        n_head=mcfg["n_head"],
        n_embd=mcfg["n_embd"],
        dropout=mcfg["dropout"],
    ).to(device)

    if not use_checkpointing:
        return model

    class CheckpointBlock(torch.nn.Module):
        def __init__(self, block: Block) -> None:
            super().__init__()
            self.block = block

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return checkpoint(self.block, x, use_reentrant=False)

    new_blocks = torch.nn.ModuleList(
        CheckpointBlock(b) for b in model.blocks  # type: ignore[arg-type]
    )
    model.blocks = new_blocks  # type: ignore[assignment]
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train TinyGPT with AMP / grad accum / optional DDP")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--resume", type=str, default="", help="Path to checkpoint .pt")
    parser.add_argument("--out_dir", type=str, default="runs/demo")
    parser.add_argument("--max-iters", type=int, default=0, help="Override config train.max_iters if > 0")
    args = parser.parse_args()

    rank, world_size, local_rank = ddp_setup()
    is_main = rank == 0

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if args.max_iters > 0:
        cfg["train"]["max_iters"] = args.max_iters

    set_seed(cfg["train"]["seed"] + rank)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available() and is_main:
        print("Warning: CUDA not available; training on CPU will be very slow.")

    train_path = cfg["data"]["train_path"]
    tok, token_ids = load_corpus(train_path)
    vocab_size = cfg["model"].get("vocab_size") or tok.vocab_size
    if cfg["model"].get("vocab_size") and cfg["model"]["vocab_size"] != tok.vocab_size:
        raise ValueError("config model.vocab_size must match tokenizer or be 0 for auto")

    ds = CharLMBlockDataset(token_ids, cfg["model"]["block_size"])
    sampler = DistributedSampler(ds, shuffle=True) if world_size > 1 else None
    dl = DataLoader(
        ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    model = build_model_with_checkpointing(
        cfg, vocab_size, cfg["train"]["use_checkpointing"], device
    )
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    raw_model = model.module if isinstance(model, DDP) else model
    use_amp = cfg["train"]["use_amp"] and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    opt = torch.optim.AdamW(
        raw_model.parameters(),
        lr=cfg["train"]["learning_rate"],
        weight_decay=cfg["train"]["weight_decay"],
        betas=(0.9, 0.95),
    )

    start_iter = 0
    out_dir = Path(args.out_dir)
    if is_main:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "meta.json").write_text(
            json.dumps({"tokenizer_chars": list(tok.stoi.keys()), "config": cfg}, indent=2),
            encoding="utf-8",
        )

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        raw_model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"])
        scaler.load_state_dict(ckpt["scaler"])
        start_iter = ckpt.get("iter", 0) + 1
        if is_main:
            print(f"Resumed from iter {start_iter}")

    tcfg = cfg["train"]
    log_iv = cfg["logging"]["log_interval"]
    eval_iv = cfg["logging"]["eval_interval"]
    max_iters = tcfg["max_iters"]

    model.train()
    it = start_iter
    dl_iter = iter(dl)
    t0 = time.perf_counter()
    tokens_seen = 0

    while it < max_iters:
        if sampler is not None:
            sampler.set_epoch(it)
        accum = tcfg["grad_accum_steps"]
        opt.zero_grad(set_to_none=True)
        last_loss = 0.0

        for micro_step in range(accum):
            try:
                x, y = next(dl_iter)
            except StopIteration:
                dl_iter = iter(dl)
                x, y = next(dl_iter)
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            tokens_seen += x.numel() * world_size

            lr = get_lr(it, tcfg["warmup_iters"], max_iters, tcfg["learning_rate"])
            for pg in opt.param_groups:
                pg["lr"] = lr

            with torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.float16):
                _, loss = model(x, y)
                loss = loss / accum

            scaler.scale(loss).backward()
            last_loss = loss.item() * accum

        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(raw_model.parameters(), tcfg["grad_clip"])
        scaler.step(opt)
        scaler.update()

        if is_main and it % log_iv == 0:
            dt = time.perf_counter() - t0
            tok_per_s = tokens_seen / max(dt, 1e-6)
            print(
                f"iter {it:6d} | loss {last_loss:.4f} | lr {lr:.2e} | "
                f"tokens/s (approx) {tok_per_s:,.0f} | wall {dt:.1f}s"
            )

        if is_main and it > 0 and it % eval_iv == 0:
            raw_model.eval()
            with torch.no_grad():
                ctx = torch.zeros((1, 1), dtype=torch.long, device=device)
                sample = raw_model.generate(ctx, max_new_tokens=120, temperature=0.9)
            out_ids = sample[0].tolist()
            gen = tok.decode(out_ids)
            print("---- sample ----")
            print(gen)
            print("----------------")
            raw_model.train()

        if is_main and it > 0 and it % eval_iv == 0:
            ckpt_path = out_dir / f"ckpt_iter_{it}.pt"
            torch.save(
                {
                    "iter": it,
                    "model": raw_model.state_dict(),
                    "optimizer": opt.state_dict(),
                    "scaler": scaler.state_dict(),
                    "config": cfg,
                },
                ckpt_path,
            )

        it += 1

    if is_main:
        torch.save(
            {
                "iter": it - 1,
                "model": raw_model.state_dict(),
                "optimizer": opt.state_dict(),
                "scaler": scaler.state_dict(),
                "config": cfg,
            },
            out_dir / "ckpt_final.pt",
        )
        print(f"Done. Checkpoints in {out_dir.resolve()}")

    ddp_cleanup()


if __name__ == "__main__":
    main()
