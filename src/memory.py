"""A three-tier memory with promotion, decay and eviction.

The mapping from the design brief to mechanisms that can actually be measured:

    conscious      working memory. Small, volatile, rewritten every step.
                   Nothing here is permanent. This is where a representation
                   exists while it is being used.
    subconscious   consolidated memory. Larger, persistent, addressed by
                   similarity. Written only by promotion from conscious, read
                   only by top-k retrieval.
    unconscious    cold store. Largest, rarely read, holds what has aged out of
                   subconscious but is still worth keeping.

The important property is that only `conscious` is on the hot path. Retrieval
from the other two tiers costs O(k * D) for k items, not O(M * D) for the whole
store, so a bigger store does not slow the model down.

This is a memory hierarchy, not a brain. The names come from the brief; the
mechanisms are bounded caches with decay, a write-back policy and similarity
retrieval. What makes it brain-*like* is that nothing in the fast tier is
permanent and that what survives is decided by use, not by explicit design.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TierStats:
    conscious_writes: int = 0
    promotions: int = 0
    demotions: int = 0
    evictions: int = 0
    retrievals: int = 0
    conscious_occupancy: float = 0.0
    subconscious_occupancy: float = 0.0
    unconscious_occupancy: float = 0.0

    def as_dict(self) -> dict:
        return {
            "conscious_writes": self.conscious_writes,
            "promotions": self.promotions,
            "demotions": self.demotions,
            "evictions": self.evictions,
            "retrievals": self.retrievals,
            "conscious_occupancy": round(self.conscious_occupancy, 4),
            "subconscious_occupancy": round(self.subconscious_occupancy, 4),
            "unconscious_occupancy": round(self.unconscious_occupancy, 4),
        }


class TriuneMemory(nn.Module):
    """Working memory that consolidates into a persistent store.

    All three tiers live in registered buffers so the module survives
    `.to(device)` and can be checkpointed with the rest of the model.
    """

    def __init__(
        self,
        dim: int,
        conscious_slots: int = 8,
        subconscious_slots: int = 128,
        unconscious_slots: int = 512,
        decay: float = 0.9,
        promote_threshold: float = 0.55,
        retrieve_k: int = 4,
    ):
        super().__init__()
        self.dim = dim
        self.conscious_slots = conscious_slots
        self.subconscious_slots = subconscious_slots
        self.unconscious_slots = unconscious_slots
        self.decay = decay
        self.promote_threshold = promote_threshold
        self.retrieve_k = retrieve_k

        # conscious: per-sequence working slots plus a scalar vitality per slot
        self.register_buffer("conscious", torch.zeros(1, conscious_slots, dim))
        self.register_buffer("vitality", torch.zeros(1, conscious_slots))
        # subconscious: persistent rows, a usage counter and a "occupied" mask
        self.register_buffer("subconscious", torch.zeros(subconscious_slots, dim))
        self.register_buffer("sub_usage", torch.zeros(subconscious_slots))
        self.register_buffer("sub_occupied", torch.zeros(subconscious_slots, dtype=torch.bool))
        # unconscious: cold rows and their last-touch counter
        self.register_buffer("unconscious", torch.zeros(unconscious_slots, dim))
        self.register_buffer("un_occupied", torch.zeros(unconscious_slots, dtype=torch.bool))

        # projections that turn a hidden state into a memory key/value
        self.to_key = nn.Linear(dim, dim)
        self.to_value = nn.Linear(dim, dim)
        self.gate = nn.Linear(dim * 2, 1)
        self.stats = TierStats()

    def reset_working_memory(self, batch: int) -> None:
        """Called at the start of every sequence: conscious state is not kept."""
        self.conscious = torch.zeros(
            batch, self.conscious_slots, self.dim, device=self.conscious.device,
            dtype=self.conscious.dtype,
        )
        self.vitality = torch.zeros(
            batch, self.conscious_slots, device=self.vitality.device,
            dtype=self.vitality.dtype,
        )

    def read(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (memory_context, confidence) for a hidden-state sequence.

        `hidden` is [B, L, D]. Conscious slots are attended per position; the
        subconscious contributes only its top-k rows, so its size never enters
        the per-token cost.
        """
        batch, length, dim = hidden.shape
        device = hidden.device
        # buffers are not learnable, so they are detached: gradients flow
        # through the projections and the hidden state, not the store
        slots = self.conscious[:batch].clone().detach()

        query = self.to_key(hidden)                       # [B,L,D]
        scores = torch.einsum("bld,bsd->bls", query, slots) / dim**0.5
        weights = torch.softmax(scores, dim=-1)
        conscious_context = torch.einsum("bls,bsd->bld", weights, slots)

        occupied = self.sub_occupied.nonzero(as_tuple=True)[0]
        if len(occupied) > 0:
            rows = self.subconscious[occupied].clone().detach()    # [M,D]
            flat = query.reshape(batch * length, dim)
            similarity = torch.einsum("nd,md->nm", flat, rows) / dim**0.5
            k = min(self.retrieve_k, len(occupied))
            top_scores, top_index = similarity.topk(k, dim=-1)
            retrieved = rows[top_index]                   # [N,k,D]
            retrieval_weights = torch.softmax(top_scores, dim=-1).unsqueeze(-1)
            subconscious_context = (retrieved * retrieval_weights).sum(1)
            subconscious_context = subconscious_context.view(batch, length, dim)
            confidence = torch.sigmoid(top_scores.max(dim=-1).values).view(batch, length)
            self.stats.retrievals += 1
        else:
            subconscious_context = torch.zeros_like(conscious_context)
            confidence = torch.zeros(batch, length, device=device)

        gate = torch.sigmoid(
            self.gate(torch.cat([conscious_context, hidden], dim=-1))
        )
        context = gate * conscious_context + (1 - gate) * subconscious_context
        return context, confidence

    @torch.no_grad()
    def write(self, hidden: torch.Tensor, importance: torch.Tensor) -> None:
        """Update conscious memory, then promote, decay and evict.

        `hidden` is [B, D] and `importance` is [B] — how much this step's
        representation is worth remembering.
        """
        batch = hidden.shape[0]
        device = hidden.device
        slots = self.conscious[:batch]
        vitality = self.vitality[:batch]

        key = self.to_value(hidden)  # [B,D]
        similarity = torch.einsum("bd,bsd->bs", key, slots) / self.dim**0.5
        target = similarity.argmax(dim=-1)  # [B]

        # write into the most similar slot, blending with what is there
        for index in range(batch):
            slot = int(target[index])
            blend = 0.5
            slots[index, slot] = blend * key[index] + (1 - blend) * slots[index, slot]
            vitality[index, slot] = torch.clamp(
                vitality[index, slot] + importance[index], max=2.0
            )
        self.conscious[:batch] = slots
        self.vitality[:batch] = vitality * self.decay
        self.stats.conscious_writes += batch

        # promotion: strong, stable slots move to the persistent tier
        strong = self.vitality[:batch] > self.promote_threshold
        if strong.any():
            candidates = slots[strong]
            self._promote(candidates)

        # eviction: slots that faded below the floor are dropped
        faded = self.vitality[:batch] < 0.05
        if faded.any():
            slots[faded] = 0.0
            vitality[faded] = 0.0
            self.conscious[:batch] = slots
            self.vitality[:batch] = vitality
            self.stats.evictions += int(faded.sum().item())

        self._update_occupancy()

    @torch.no_grad()
    def _promote(self, candidates: torch.Tensor) -> None:
        """Write candidate rows into subconscious, deduplicating by similarity."""
        for row in candidates:
            free = (~self.sub_occupied).nonzero(as_tuple=True)[0]
            occupied = self.sub_occupied.nonzero(as_tuple=True)[0]

            duplicate = False
            if len(occupied) > 0:
                similarity = F.cosine_similarity(
                    self.subconscious[occupied], row.unsqueeze(0), dim=-1
                )
                if float(similarity.max()) > 0.95:
                    best = occupied[int(similarity.argmax())]
                    self.sub_usage[best] += 1.0
                    duplicate = True

            if duplicate:
                continue

            if len(free) > 0:
                target = int(free[0])
                self.subconscious[target] = row
                self.sub_usage[target] = 1.0
                self.sub_occupied[target] = True
                self.stats.promotions += 1
            else:
                # store full: the least used row is demoted to the cold tier
                weakest = int(self.sub_usage.argmin())
                self._demote(self.subconscious[weakest])
                self.subconscious[weakest] = row
                self.sub_usage[weakest] = 1.0
                self.stats.promotions += 1

    @torch.no_grad()
    def _demote(self, row: torch.Tensor) -> None:
        free = (~self.un_occupied).nonzero(as_tuple=True)[0]
        if len(free) > 0:
            target = int(free[0])
            self.unconscious[target] = row
            self.un_occupied[target] = True
            self.stats.demotions += 1
        else:
            # cold store full: overwrite the oldest slot (ring behaviour)
            self.unconscious[torch.randint(0, self.unconscious_slots, (1,)).item()] = row

    def _update_occupancy(self) -> None:
        self.stats.conscious_occupancy = float(
            (self.vitality[: self.conscious.shape[0]] > 0.05).float().mean()
        )
        self.stats.subconscious_occupancy = float(self.sub_occupied.float().mean())
        self.stats.unconscious_occupancy = float(self.un_occupied.float().mean())

    @torch.no_grad()
    def recall_from_unconscious(self, hidden: torch.Tensor, top_k: int = 2) -> torch.Tensor:
        """Explicit cold-tier access. Never called on the hot path.

        This exists so the design can be exercised and measured; the block only
        uses it when the subconscious returns low confidence.
        """
        occupied = self.un_occupied.nonzero(as_tuple=True)[0]
        if len(occupied) == 0:
            return torch.zeros_like(hidden)
        rows = self.unconscious[occupied]
        similarity = torch.einsum("bd,md->bm", hidden, rows) / self.dim**0.5
        k = min(top_k, len(occupied))
        scores, index = similarity.topk(k, dim=-1)
        weights = torch.softmax(scores, dim=-1).unsqueeze(-1)
        return (rows[index] * weights).sum(1)
