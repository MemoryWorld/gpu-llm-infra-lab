"""Micro-benchmarks: matmul TFLOP/s (approx), memory bandwidth, AMP speedup."""

from __future__ import annotations

import argparse
import time

import torch


def bench_matmul(device: torch.device, n: int = 4096, dtype: torch.dtype = torch.float16) -> None:
    a = torch.randn(n, n, device=device, dtype=dtype)
    b = torch.randn(n, n, device=device, dtype=dtype)
    # warmup
    for _ in range(5):
        c = a @ b
    torch.cuda.synchronize() if device.type == "cuda" else None

    repeats = 50
    t0 = time.perf_counter()
    for _ in range(repeats):
        c = a @ b
    torch.cuda.synchronize() if device.type == "cuda" else None
    dt = (time.perf_counter() - t0) / repeats
    flops = 2 * (n**3)
    tflops = flops / dt / 1e12
    print(f"matmul {n}x{n} dtype={dtype}: {dt*1000:.3f} ms/iter, ~{tflops:.2f} TFLOP/s (theoretical peak not subtracted)")


def bench_amp_mlp(device: torch.device, hidden: int = 2048, batch: int = 512) -> None:
    x = torch.randn(batch, hidden, device=device)
    w1 = torch.randn(hidden, 4 * hidden, device=device)
    w2 = torch.randn(4 * hidden, hidden, device=device)

    def run_fp32() -> None:
        y = (x @ w1).relu() @ w2

    def run_amp() -> None:
        with torch.cuda.amp.autocast(dtype=torch.float16):
            y = (x @ w1).relu() @ w2
        _ = y

    for fn in (run_fp32, run_amp):
        for _ in range(10):
            fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(100):
            fn()
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / 100
        name = "fp32" if fn is run_fp32 else "amp_fp16"
        print(f"MLP-ish chain batch={batch} hidden={hidden} [{name}]: {dt*1000:.3f} ms/step")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matmul-n", type=int, default=4096)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        print("CUDA not found; matmul runs on CPU (slow).")
        dev = torch.device("cpu")
        bench_matmul(dev, n=min(args.matmul_n, 1024), dtype=torch.float32)
        return
    dev = torch.device("cuda")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    bench_matmul(dev, n=args.matmul_n, dtype=torch.float16)
    bench_matmul(dev, n=args.matmul_n, dtype=torch.float32)
    bench_amp_mlp(dev)


if __name__ == "__main__":
    main()
