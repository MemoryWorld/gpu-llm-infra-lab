"""Render a simple Triton model-repository layout image for README."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt


LAYOUT = """model_repository/
  tiny_gpt_onnx/
    config.pbtxt
    1/
      model.onnx

  tiny_gpt_trt/
    config.pbtxt
    1/
      model.plan
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Render Triton repo layout screenshot-like PNG")
    parser.add_argument("--out", type=str, default="artifacts/triton_model_repo_layout.png")
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(7.5, 4.2), facecolor="#111827")
    ax = fig.add_subplot(111)
    ax.set_facecolor("#111827")
    ax.axis("off")
    ax.text(
        0.03,
        0.95,
        "Triton Model Repository Layout",
        fontsize=14,
        fontweight="bold",
        color="#e5e7eb",
        va="top",
        ha="left",
    )
    ax.text(
        0.04,
        0.84,
        LAYOUT,
        family="monospace",
        fontsize=11.5,
        color="#93c5fd",
        va="top",
        ha="left",
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    print(f"Saved image -> {out_path.resolve()}")


if __name__ == "__main__":
    main()
