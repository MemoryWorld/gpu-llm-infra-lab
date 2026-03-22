"""
Toy GPU job scheduler simulation (FIFO vs Smallest-GPU-Memory-First).

Not a production scheduler — demonstrates reasoning about utilization and queueing.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field


@dataclass(order=True)
class Job:
    mem_gb: float
    duration_s: float
    name: str = field(compare=False)


def fifo_finish_time(jobs: list[Job], gpus: int) -> float:
    """One job per GPU at a time, round-robin assignment by arrival order."""
    loads = [0.0] * gpus
    for i, j in enumerate(jobs):
        g = i % gpus
        loads[g] += j.duration_s
    return max(loads)


def greedy_by_memory(jobs: list[Job], gpus: int) -> float:
    """Assign next job to the GPU with smallest current load (list scheduling)."""
    loads = [0.0] * gpus
    sorted_jobs = sorted(jobs, key=lambda x: x.mem_gb, reverse=True)
    for j in sorted_jobs:
        g = min(range(gpus), key=lambda i: loads[i])
        loads[g] += j.duration_s
    return max(loads)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", type=int, default=4)
    args = parser.parse_args()

    # Synthetic training jobs: (mem_gb, duration_s, name)
    jobs = [
        Job(20, 3600, "llm-7b"),
        Job(40, 7200, "llm-13b"),
        Job(10, 900, "cv-exp"),
        Job(8, 1200, "cv-exp2"),
        Job(22, 4000, "llm-7b-tune"),
    ]

    f = fifo_finish_time(jobs, args.gpus)
    g = greedy_by_memory(jobs, args.gpus)
    print(f"GPUs: {args.gpus}")
    print(f"FIFO makespan (sum per-GPU queue, parallel): {f/3600:.2f} h")
    print(f"Greedy-by-memory list scheduling makespan:     {g/3600:.2f} h")
    print("Interpretation: packing and policy affect utilization and tail latency.")


if __name__ == "__main__":
    main()
