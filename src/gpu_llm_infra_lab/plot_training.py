"""Plot loss and throughput curves from train.py logs."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt


LINE_RE = re.compile(
    r"iter\s+(?P<iter>\d+)\s+\|\s+loss\s+(?P<loss>[0-9.]+)\s+\|.*tokens/s \(approx\)\s+(?P<tps>[0-9,]+)"
)


def _read_text_auto(path: Path) -> str:
    raw = path.read_bytes()
    for enc in ("utf-8", "utf-16", "utf-16-le"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="ignore")


def parse_log(log_path: Path) -> tuple[list[int], list[float], list[float]]:
    iters: list[int] = []
    losses: list[float] = []
    tps: list[float] = []

    text = _read_text_auto(log_path).replace("\x00", "")
    for line in text.splitlines():
        m = LINE_RE.search(line)
        if not m:
            continue
        iters.append(int(m.group("iter")))
        losses.append(float(m.group("loss")))
        tps.append(float(m.group("tps").replace(",", "")))
    return iters, losses, tps


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot training metrics from text log")
    parser.add_argument("--log", type=str, required=True)
    parser.add_argument("--out", type=str, default="artifacts/train_curve.png")
    args = parser.parse_args()

    log_path = Path(args.log)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    iters, losses, tps = parse_log(log_path)
    if not iters:
        raise ValueError(f"No metrics found in log: {log_path}")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    ax1.plot(iters, losses, marker="o", linewidth=2)
    ax1.set_ylabel("Loss")
    ax1.set_title("Training Loss")
    ax1.grid(alpha=0.3)

    ax2.plot(iters, tps, marker="o", linewidth=2)
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Tokens / s")
    ax2.set_title("Approx Throughput")
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    print(f"Saved plot -> {out_path.resolve()}")


if __name__ == "__main__":
    main()
