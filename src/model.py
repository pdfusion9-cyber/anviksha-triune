"""Character-level language models: a baseline transformer and the triune variant."""
from __future__ import annotations

import torch
import torch.nn as nn

from blocks import BaselineBlock, TriuneBlock
from memory import TriuneMemory


class CharLM(nn.Module):
    """Shared scaffolding: embedding, positional encoding, blocks, head."""

    def __init__(self, vocab: int, dim: int, layers: int, heads: int,
                 seq_len: int, dropout: float = 0.1):
        super().__init__()
        self.embed = nn.Embedding(vocab, dim)
        self.position = nn.Embedding(seq_len, dim)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab, bias=False)
        self.seq_len = seq_len
        self.dim = dim
        self.layers = nn.ModuleList()
        self.memory = None

    def embed_input(self, tokens: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(tokens.shape[1], device=tokens.device)
        return self.embed(tokens) + self.position(positions)[None]

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.embed_input(tokens)
        for layer in self.layers:
            x = layer(x)
        return self.head(self.norm(x))


class BaselineLM(CharLM):
    def __init__(self, vocab: int, dim: int = 96, layers: int = 3, heads: int = 4,
                 seq_len: int = 128, dropout: float = 0.1):
        super().__init__(vocab, dim, layers, heads, seq_len, dropout)
        self.layers = nn.ModuleList(
            [BaselineBlock(dim, heads, dropout) for _ in range(layers)]
        )


class TriuneLM(CharLM):
    """Baseline stack with the memory tiers attached to every block."""

    def __init__(self, vocab: int, dim: int = 96, layers: int = 3, heads: int = 4,
                 seq_len: int = 128, dropout: float = 0.1,
                 conscious_slots: int = 8, subconscious_slots: int = 128,
                 unconscious_slots: int = 512, active_nodes: int = 2):
        super().__init__(vocab, dim, layers, heads, seq_len, dropout)
        self.memory = TriuneMemory(
            dim,
            conscious_slots=conscious_slots,
            subconscious_slots=subconscious_slots,
            unconscious_slots=unconscious_slots,
        )
        self.layers = nn.ModuleList(
            [
                TriuneBlock(dim, heads, memory=self.memory, active_nodes=active_nodes,
                            dropout=dropout)
                for _ in range(layers)
            ]
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        self.memory.reset_working_memory(tokens.shape[0])
        return super().forward(tokens)
