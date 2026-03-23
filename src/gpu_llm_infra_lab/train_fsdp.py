"""Minimal FSDP wrapper training entrypoint.

This is intentionally lightweight: it reuses TinyGPT and dataset code paths
but swaps DDP for FSDP when distributed world size > 1.
"""

from __future__ import annotations

import argparse
import os
import time
from functools import partial

import torch
import torch.distributed as dist
import yaml
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from torch.utils.data import DataLoader, DistributedSampler

from .data_loader import CharLMBlockDataset, load_corpus
from .tiny_gpt import TinyGPT
from .train import get_lr, set_seed


def setup_dist() -> tuple[int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def ensure_single_process_group() -> None:
    if dist.is_initialized():
        return
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29511"
    os.environ.setdefault("USE_LIBUV", "0")
    dist.init_process_group(
        backend="gloo",
        rank=0,
        world_size=1,
        init_method=f"tcp://{os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}",
    )


def cleanup() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser(description="TinyGPT training with FSDP wrapper")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--out_dir", type=str, default="runs/fsdp_demo")
    parser.add_argument("--max-iters", type=int, default=0)
    parser.add_argument(
        "--force-fsdp",
        action="store_true",
        help="Wrap with FSDP even for WORLD_SIZE=1 (single-node wrapper sanity check)",
    )
    args = parser.parse_args()

    rank, world_size, local_rank = setup_dist()
    is_main = rank == 0

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if args.max_iters > 0:
        cfg["train"]["max_iters"] = args.max_iters

    set_seed(cfg["train"]["seed"] + rank)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    tok, token_ids = load_corpus(cfg["data"]["train_path"])
    ds = CharLMBlockDataset(token_ids, cfg["model"]["block_size"])
    sampler = DistributedSampler(ds, shuffle=True) if world_size > 1 else None
    dl = DataLoader(
        ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=sampler is None,
        sampler=sampler,
        pin_memory=torch.cuda.is_available(),
    )

    mcfg = cfg["model"]
    model = TinyGPT(
        vocab_size=mcfg.get("vocab_size") or tok.vocab_size,
        block_size=mcfg["block_size"],
        n_layer=mcfg["n_layer"],
        n_head=mcfg["n_head"],
        n_embd=mcfg["n_embd"],
        dropout=mcfg["dropout"],
    ).to(device)

    if args.force_fsdp:
        ensure_single_process_group()
    if world_size > 1 or args.force_fsdp:
        wrap_policy = partial(size_based_auto_wrap_policy, min_num_params=20000)
        model = FSDP(model, auto_wrap_policy=wrap_policy)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["train"]["learning_rate"],
        weight_decay=cfg["train"]["weight_decay"],
        betas=(0.9, 0.95),
    )
    use_amp = cfg["train"]["use_amp"] and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    model.train()
    t0 = time.perf_counter()
    tokens_seen = 0
    dl_iter = iter(dl)
    max_iters = cfg["train"]["max_iters"]
    accum = cfg["train"]["grad_accum_steps"]
    for it in range(max_iters):
        if sampler is not None:
            sampler.set_epoch(it)
        opt.zero_grad(set_to_none=True)
        last_loss = 0.0
        for _ in range(accum):
            try:
                x, y = next(dl_iter)
            except StopIteration:
                dl_iter = iter(dl)
                x, y = next(dl_iter)
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            tokens_seen += x.numel() * world_size

            lr = get_lr(it, cfg["train"]["warmup_iters"], max_iters, cfg["train"]["learning_rate"])
            for pg in opt.param_groups:
                pg["lr"] = lr
            with torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.float16):
                _, loss = model(x, y)
                loss = loss / accum
            scaler.scale(loss).backward()
            last_loss = loss.item() * accum
        scaler.step(opt)
        scaler.update()

        if is_main and it % cfg["logging"]["log_interval"] == 0:
            dt = time.perf_counter() - t0
            print(
                f"[fsdp] iter {it:6d} | loss {last_loss:.4f} | "
                f"tokens/s (approx) {tokens_seen / max(dt, 1e-6):,.0f}"
            )

    if is_main:
        print("FSDP run finished.")
    cleanup()


if __name__ == "__main__":
    main()
