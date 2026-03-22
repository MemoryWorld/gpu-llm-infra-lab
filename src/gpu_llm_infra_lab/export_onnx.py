"""Export TinyGPT to ONNX for TensorRT / ONNX Runtime (install CUDA PyTorch + `pip install onnx` locally)."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn

from .ckpt_utils import state_dict_for_plain_tinygpt
from .data_loader import load_corpus
from .tiny_gpt import TinyGPT


class TinyGPTLogitsOnly(nn.Module):
    """ONNX-friendly wrapper: token ids -> logits (no loss)."""

    def __init__(self, core: TinyGPT) -> None:
        super().__init__()
        self.core = core

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        logits, _ = self.core(idx, None)
        return logits


def main() -> None:
    parser = argparse.ArgumentParser(description="Export TinyGPT checkpoint to ONNX")
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--corpus", type=str, default="data/sample_corpus.txt")
    parser.add_argument("--out", type=str, default="artifacts/tiny_gpt.onnx")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument(
        "--static",
        action="store_true",
        help="Do not mark batch/seq as dynamic (easier first build with trtexec)",
    )
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
    wrapped = TinyGPTLogitsOnly(model)

    block = mcfg["block_size"]
    dummy = torch.zeros(1, block, dtype=torch.long)
    dynamic_axes = None if args.static else {
        "input_ids": {0: "batch", 1: "seq"},
        "logits": {0: "batch", 1: "seq"},
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # dynamo=False: legacy TorchScript trace exporter (no onnxscript); better for TensorRT + small models.
    torch.onnx.export(
        wrapped,
        dummy,
        str(out_path),
        input_names=["input_ids"],
        output_names=["logits"],
        dynamic_axes=dynamic_axes,
        opset_version=args.opset,
        do_constant_folding=True,
        dynamo=False,
    )

    try:
        import onnx

        onnx.checker.check_model(str(out_path))
        print(f"ONNX check OK -> {out_path.resolve()}")
    except ImportError:
        print(f"Exported (install `onnx` to run checker) -> {out_path.resolve()}")


if __name__ == "__main__":
    main()
