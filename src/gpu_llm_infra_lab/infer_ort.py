"""Benchmark ONNX Runtime inference latency (CPU or CUDA EP if available)."""

from __future__ import annotations

import argparse
import time

import numpy as np
import onnxruntime as ort
import torch

from .ckpt_utils import vocab_size_from_checkpoint


def pick_providers(prefer_cuda: bool) -> list[str]:
    available = ort.get_available_providers()
    out: list[str] = []
    if prefer_cuda and "CUDAExecutionProvider" in available:
        out.append("CUDAExecutionProvider")
    out.append("CPUExecutionProvider")
    return out


def resolve_seq_len(sess: ort.InferenceSession, override: int) -> int:
    if override > 0:
        return override
    shape = sess.get_inputs()[0].shape
    if len(shape) >= 2 and isinstance(shape[1], int) and shape[1] > 0:
        return int(shape[1])
    return 128


def main() -> None:
    parser = argparse.ArgumentParser(description="ONNX Runtime latency benchmark")
    parser.add_argument("--onnx", type=str, required=True)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=15)
    parser.add_argument("--seq", type=int, default=0, help="Sequence length; 0 = infer from model or 128")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument(
        "--vocab-max",
        type=int,
        default=256,
        help="Upper bound for random token ids if --ckpt is not set",
    )
    parser.add_argument("--ckpt", type=str, default="", help="Use checkpoint to set vocab size for valid token ids")
    parser.add_argument("--cuda", action="store_true", help="Prefer CUDAExecutionProvider if installed")
    args = parser.parse_args()

    if args.ckpt:
        ckpt = torch.load(args.ckpt, map_location="cpu")
        vocab_hi = vocab_size_from_checkpoint(ckpt["model"])
    else:
        vocab_hi = args.vocab_max

    providers = pick_providers(args.cuda)
    sess = ort.InferenceSession(args.onnx, providers=providers)
    inp = sess.get_inputs()[0]
    seq = resolve_seq_len(sess, args.seq)
    x = np.random.randint(0, max(1, vocab_hi), size=(args.batch, seq), dtype=np.int64)
    feeds = {inp.name: x}

    for _ in range(args.warmup):
        sess.run(None, feeds)

    t0 = time.perf_counter()
    for _ in range(args.steps):
        sess.run(None, feeds)
    dt = (time.perf_counter() - t0) / args.steps

    active = sess.get_providers()
    print(f"ORT providers (session): {active}")
    print(f"input {inp.name} shape={x.shape} dtype={x.dtype}")
    print(f"latency: {dt * 1000:.3f} ms/run (mean of {args.steps} runs after {args.warmup} warmup)")


if __name__ == "__main__":
    main()
