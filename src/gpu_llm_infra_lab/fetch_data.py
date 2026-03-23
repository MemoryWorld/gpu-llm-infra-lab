"""Download small public corpora for language-model experiments."""

from __future__ import annotations

import argparse
from pathlib import Path
from urllib.request import urlopen


TINY_SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download public training corpus")
    parser.add_argument("--dataset", choices=["tinyshakespeare"], default="tinyshakespeare")
    parser.add_argument("--out", type=str, default="data/tinyshakespeare.txt")
    args = parser.parse_args()

    url = TINY_SHAKESPEARE_URL
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with urlopen(url, timeout=30) as r:
        text = r.read().decode("utf-8")
    out_path.write_text(text, encoding="utf-8")

    print(f"Downloaded {args.dataset} -> {out_path.resolve()} ({len(text):,} chars)")


if __name__ == "__main__":
    main()
