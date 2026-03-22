from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CharTokenizer:
    """Fixed vocabulary char-level tokenizer (no external deps)."""

    stoi: dict[str, int]
    itos: dict[int, str]

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

    @classmethod
    def from_text(cls, text: str) -> "CharTokenizer":
        chars = sorted(set(text))
        stoi = {ch: i for i, ch in enumerate(chars)}
        itos = {i: ch for ch, i in stoi.items()}
        return cls(stoi=stoi, itos=itos)

    def encode(self, s: str) -> list[int]:
        return [self.stoi[c] for c in s]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos[i] for i in ids)
