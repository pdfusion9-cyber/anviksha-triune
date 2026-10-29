"""Character-level dataset over a Hindi text corpus."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


class CharDataset:
    def __init__(self, text_path: Path, seq_len: int = 128, max_chars: int | None = None):
        text = Path(text_path).read_text(encoding="utf-8")
        if max_chars:
            text = text[:max_chars]
        self.vocab = sorted(set(text))
        self.stoi = {c: i for i, c in enumerate(self.vocab)}
        self.itos = {i: c for c, i in self.stoi.items()}
        self.data = np.array([self.stoi[c] for c in text], dtype=np.int64)
        self.seq_len = seq_len
        split = int(0.9 * len(self.data))
        self.train = self.data[:split]
        self.valid = self.data[split:]

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def batch(self, source: np.ndarray, batch: int, seq_len: int,
              rng: np.random.Generator) -> tuple[torch.Tensor, torch.Tensor]:
        starts = rng.integers(0, len(source) - seq_len - 1, size=batch)
        inputs = np.stack([source[s : s + seq_len] for s in starts])
        targets = np.stack([source[s + 1 : s + seq_len + 1] for s in starts])
        return torch.from_numpy(inputs), torch.from_numpy(targets)
