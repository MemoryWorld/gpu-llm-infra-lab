"""Compare FP32 vs dynamic-quantized CPU inference latency on TinyGPT (portfolio demo)."""

from __future__ import annotations

import argparse
import time

import torch
import torch.nn as nn

from .ckpt_utils import state_dict_for_plain_tinygpt
from .data_loader import load_corpus
from .tiny_gpt import TinyGPT


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--corpus", type=str, default="data/sample_corpus.txt")
    parser.add_argument("--steps", type=int, default=200)
    args = parser.parse_args()

    tok, _ = load_corpus(args.corpus)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    cfg = ckpt["config"]
    mcfg = cfg["model"]
    vocab_size = mcfg.get("vocab_size") or tok.vocab_size

    model = TinyGPT(
        vocab_size=vocab_size,
        block_size=mcfg["block_size"],
        n_layer=mcfg["n_layer"],
        n_head=mcfg["n_head"],
        n_embd=mcfg["n_embd"],
        dropout=0.0,
    )
    model.load_state_dict(state_dict_for_plain_tinygpt(ckpt["model"]))
    model.eval()

    block = mcfg["block_size"]
    x = torch.randint(0, vocab_size, (1, block))

    def measure(m: nn.Module, label: str) -> None:
        with torch.no_grad():
            for _ in range(20):
                m(x, None)
            t0 = time.perf_counter()
            for _ in range(args.steps):
                m(x, None)
            dt = (time.perf_counter() - t0) / args.steps
        print(f"{label}: {dt*1000:.3f} ms/forward (CPU)")

    measure(model, "fp32 eager")

    q_model = torch.ao.quantization.quantize_dynamic(
        model, {nn.Linear}, dtype=torch.qint8, inplace=False
    )
    measure(q_model, "dynamic quant Linear (qint8)")


if __name__ == "__main__":
    main()
