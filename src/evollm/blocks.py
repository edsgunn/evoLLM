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
    # Blocks this agent owns on another agent's behalf: subject id -> blocks.
    # A parent carries its children's adapters (§3.2), so reproduction has a
    # cost denominated in memory that is genuinely in use rather than in a tax
    # on nothing. Nothing extra is allocated — the child's adapter is the same
    # 22 blocks the engine really registered; only the owner differs.
    dependents: dict[str, int] = field(default_factory=dict)

    @property
    def dependent_blocks(self) -> int:
        return sum(self.dependents.values())

    @property
    def total(self) -> int:
        return self.adapter_blocks + self.kv_blocks + self.dependent_blocks


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

    def reserve_dependent(self, owner_id: str, subject_id: str,
                          blocks: int) -> bool:
        """Charge `blocks` of `subject_id`'s footprint to `owner_id`.

        Used at birth: each parent takes on a share of the child's adapter, so
        a prolific agent carries the memory its offspring occupy and dies
        sooner for it. The blocks are not new — the caller must NOT also
        reserve them against the subject — so total room usage is unchanged
        and no device memory is wasted to create the incentive.
        """
        if self.free < blocks:
            return False
        h = self.holdings.setdefault(owner_id, Holding())
        h.dependents[subject_id] = h.dependents.get(subject_id, 0) + blocks
        return True

    def release_dependent(self, subject_id: str) -> int:
        """Free every dependent charge for `subject_id`, wherever it is held.

        Called when the subject dies: the child's adapter is gone, so whoever
        was carrying it stops paying.
        """
        freed = 0
        for owner, h in list(self.holdings.items()):
            n = h.dependents.pop(subject_id, 0)
            freed += n
            if h.total == 0:
                self.holdings.pop(owner, None)
        return freed

    def revert_dependents(self, owner_id: str) -> dict[str, int]:
        """Hand an owner's dependent charges back to the subjects themselves.

        Called when the OWNER dies. The child is still alive and its adapter
        is still registered, so the blocks must keep being accounted for —
        they revert to the child, which is who they were always describing.
        """
        h = self.holdings.get(owner_id)
        if h is None or not h.dependents:
            return {}
        moved = dict(h.dependents)
        h.dependents.clear()
        for subject, n in moved.items():
            self.holdings.setdefault(subject, Holding()).adapter_blocks += n
        return moved

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
    def random_holder(self, rng: np.random.Generator,
                      eligible: set[str] | None = None) -> str | None:
        """Pick a victim with probability proportional to blocks held —
        i.e. choose a random held block and return its owner. Content-blind.

        `eligible` restricts the draw to agents the caller can actually kill.
        Not every holder is one: a newborn holds its adapter while still in
        `_pending_arrivals`, and a migrant holds its full footprint at the
        destination before the source has released it (§4.5). Both are in the
        ledger and neither is in `self.agents`, so drawing them raised a
        KeyError that killed two runs the first time this policy was ever
        used. Their blocks still count toward what fills the room — they are
        genuinely allocated — they just cannot be the ones to die.
        """
        ids = [a for a, h in self.holdings.items()
               if h.total > 0 and (eligible is None or a in eligible)]
        if not ids:
            return None
        weights = np.array([self.holdings[a].total for a in ids], dtype=np.float64)
        return ids[int(rng.choice(len(ids), p=weights / weights.sum()))]


def adapter_blocks_needed(adapter_bytes: int, block_bytes: int) -> int:
    return math.ceil(adapter_bytes / block_bytes)
