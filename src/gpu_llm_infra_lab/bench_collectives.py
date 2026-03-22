"""
Benchmark all-reduce style synchronization (DDP / NCCL mental model).

Single process: measures tensor copy + reduction on GPU as a stand-in for local work.
Multi-GPU: run with torchrun --nproc_per_node=N and compare all_reduce latency.
"""

from __future__ import annotations

import os
import sys
import time

import torch
import torch.distributed as dist


def setup() -> tuple[int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return 0, 1, 0
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    backend = "nccl" if torch.cuda.is_available() and sys.platform != "win32" else "gloo"
    dist.init_process_group(backend=backend)
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def cleanup() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def main() -> None:
    rank, world_size, local_rank = setup()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    n = 50_000_000
    tensor = torch.ones(n // world_size, device=device, dtype=torch.float32)

    if world_size > 1:
        for _ in range(10):
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize() if device.type == "cuda" else None
        t0 = time.perf_counter()
        for _ in range(20):
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize() if device.type == "cuda" else None
        dt = (time.perf_counter() - t0) / 20
        if rank == 0:
            elems = tensor.numel() * world_size
            print(f"all_reduce SUM float32 ~{elems/1e6:.1f}M elems per rank, world={world_size}: {dt*1000:.3f} ms/iter")
    else:
        # local reduction + sync as a teaching baseline
        acc = torch.zeros_like(tensor)
        for _ in range(10):
            acc.copy_(tensor * 2)
            acc += tensor
        torch.cuda.synchronize() if device.type == "cuda" else None
        t0 = time.perf_counter()
        for _ in range(100):
            acc.copy_(tensor * 2)
            acc += tensor
        torch.cuda.synchronize() if device.type == "cuda" else None
        dt = (time.perf_counter() - t0) / 100
        print(f"single-GPU fake reduce {tensor.numel()/1e6:.1f}M elems: {dt*1000:.3f} ms/iter")
        print("Tip: on Linux with 2+ GPUs run: torchrun --standalone --nproc_per_node=2 -m gpu_llm_infra_lab.bench_collectives")

    cleanup()


if __name__ == "__main__":
    main()
