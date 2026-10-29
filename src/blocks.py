"""Transformer blocks: a baseline and a triune variant.

The triune block adds two things to a standard pre-norm block:

  1. a 2x2x2 node bank — eight small expert nodes, of which only the top two
     run for each token. Sparse by construction, so the bank costs roughly two
     nodes' worth of compute while holding eight nodes' worth of parameters.
  2. a memory read/write against `TriuneMemory`, gated by a learned scalar so
     the model can ignore memory when it is not useful.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from memory import TriuneMemory


class BaselineBlock(nn.Module):
    """Plain pre-norm transformer block, used as the control."""

    def __init__(self, dim: int, heads: int, dropout: float = 0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim), nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor, mask=None) -> torch.Tensor:
        attended, _ = self.attention(x, x, x, need_weights=False)
        x = x + attended
        return x + self.mlp(self.norm2(x))


class Node2x2x2(nn.Module):
    """Eight expert nodes addressed by a (level, row, column) index."""

    def __init__(self, dim: int, hidden: int | None = None):
        super().__init__()
        hidden = hidden or dim * 2
        self.nodes = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim)
                )
                for _ in range(8)
            ]
        )

    def forward(self, x: torch.Tensor, index: int) -> torch.Tensor:
        return self.nodes[index](x)


class TriuneBlock(nn.Module):
    """Attention, then a sparse node bank, then a memory read and write."""

    def __init__(
        self,
        dim: int,
        heads: int,
        memory: TriuneMemory | None = None,
        active_nodes: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.attention = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)

        self.bank = Node2x2x2(dim)
        self.router = nn.Linear(dim, 8)
        self.active_nodes = active_nodes

        self.memory = memory
        self.memory_norm = nn.LayerNorm(dim)
        self.memory_scale = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attended, _ = self.attention(self.norm1(x), self.norm1(x), self.norm1(x),
                                     need_weights=False)
        x = x + attended

        # --- sparse node bank: route each token to its top-n nodes
        flat = self.norm2(x).reshape(-1, x.shape[-1])
        logits = self.router(flat)
        top = logits.topk(self.active_nodes, dim=-1)
        weights = torch.softmax(top.values, dim=-1)

        node_output = torch.zeros_like(flat)
        for position in range(self.active_nodes):
            index = top.indices[:, position]
            weight = weights[:, position].unsqueeze(-1)
            for node_id in index.unique().tolist():
                selected = index == node_id
                node_output[selected] += weight[selected] * self.bank(flat[selected], node_id)
        x = x + node_output.view_as(x)

        # --- memory read, then write
        if self.memory is not None:
            batch, length, dim = x.shape
            hidden = self.memory_norm(x)
            memory_context, confidence = self.memory.read(hidden)
            x = x + torch.tanh(self.memory_scale) * memory_context

            # only the last position writes, and only with the weight the model
            # itself assigns to not knowing
            importance = (1.0 - confidence).detach()[:, -1]
            self.memory.write(hidden[:, -1, :], importance)
        return x
