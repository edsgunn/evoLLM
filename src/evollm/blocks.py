"""The block economy (§2.2).

One pool per room. Everything that consumes device memory — KV cache and
adapter weights — draws blocks from the same pool. The controller is the
authority on this accounting: per-agent KV consumption is exactly
ceil(tokens / block_size) because the controller drives the rollout and knows
every agent's token count (§4.2).

Nothing here is a score. Blocks are either available or they are not.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Holding:
    adapter_blocks: int = 0
    kv_blocks: int = 0

    @property
    def total(self) -> int:
        return self.adapter_blocks + self.kv_blocks


@dataclass
class BlockPool:
    """Authoritative free-block accounting for one room."""

    capacity: int
    block_size: int
    holdings: dict[str, Holding] = field(default_factory=dict)

    @property
    def used(self) -> int:
        return sum(h.total for h in self.holdings.values())

    @property
    def free(self) -> int:
        return self.capacity - self.used

    def kv_blocks_for(self, tokens: int) -> int:
        return math.ceil(tokens / self.block_size)

    # ── adapter blocks ────────────────────────────────────────────────────
    def try_reserve_adapter(self, agent_id: str, blocks: int) -> bool:
        """Reserve adapter blocks for a (new or arriving) agent.

        Adapter blocks are reserved before the agent has any KV (§4.2), so a
        birth or arrival fails on adapter-block availability.
        """
        if self.free < blocks:
            return False
        self.holdings.setdefault(agent_id, Holding()).adapter_blocks += blocks
        return True

    # ── KV growth ─────────────────────────────────────────────────────────
    def kv_needs_block(self, agent_id: str, new_token_count: int) -> bool:
        """True iff growing this agent's context to new_token_count requires
        allocating a KV block it does not yet hold."""
        held = self.holdings.setdefault(agent_id, Holding()).kv_blocks
        return self.kv_blocks_for(new_token_count) > held

    def try_grow_kv(self, agent_id: str, new_token_count: int) -> bool:
        """Grow the agent's KV holding to cover new_token_count tokens.

        Returns False if a new block is needed and the pool is empty — the
        scarcity event that constitutes death (§2.5). No state changes on
        failure; the eviction policy decides who dies, after which the caller
        retries.
        """
        holding = self.holdings.setdefault(agent_id, Holding())
        needed = self.kv_blocks_for(new_token_count)
        extra = needed - holding.kv_blocks
        if extra <= 0:
            return True
        if self.free < extra:
            return False
        holding.kv_blocks = needed
        return True

    # ── release ───────────────────────────────────────────────────────────
    def release_all(self, agent_id: str) -> None:
        self.holdings.pop(agent_id, None)

    def release_adapter(self, agent_id: str) -> None:
        h = self.holdings.get(agent_id)
        if h is not None:
            h.adapter_blocks = 0
            if h.total == 0:
                self.holdings.pop(agent_id, None)

    # ── eviction (§2.5) ───────────────────────────────────────────────────
    def random_holder(self, rng: np.random.Generator) -> str | None:
        """Pick a victim with probability proportional to blocks held —
        i.e. choose a random held block and return its owner. Content-blind."""
        ids = [a for a, h in self.holdings.items() if h.total > 0]
        if not ids:
            return None
        weights = np.array([self.holdings[a].total for a in ids], dtype=np.float64)
        return ids[int(rng.choice(len(ids), p=weights / weights.sum()))]


def adapter_blocks_needed(adapter_bytes: int, block_bytes: int) -> int:
    return math.ceil(adapter_bytes / block_bytes)
