from __future__ import annotations

import torch
from torch.utils.data import Dataset

from .char_tokenizer import CharTokenizer


class CharLMBlockDataset(Dataset):
    """Overlapping blocks of length block_size+1 (input / target shifted by 1)."""

    def __init__(self, token_ids: list[int], block_size: int) -> None:
        self.block_size = block_size
        self.data = token_ids
        if len(self.data) < block_size + 1:
            raise ValueError("Corpus too short for block_size; add more text to data/.")

    def __len__(self) -> int:
        return len(self.data) - self.block_size

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        chunk = self.data[idx : idx + self.block_size + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y


def load_corpus(path: str) -> tuple[CharTokenizer, list[int]]:
    with open(path, encoding="utf-8") as f:
        text = f.read()
    if not text.strip():
        raise ValueError(f"Empty corpus: {path}")
    tok = CharTokenizer.from_text(text)
    return tok, tok.encode(text)
