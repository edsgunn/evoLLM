# Crashed: first use of `eviction: random_holder` (2026-08-25)

Jobs 6118397 and 6118523 both died with `KeyError` in `_append_token`, after
21 and 55 minutes. Archived because they are truncated, not because they are
wrong — the data up to the crash is valid and is quoted below.

## The bug

`BlockPool.random_holder` drew a victim from the whole holdings ledger, and
the controller then did `self.agents[victim_id]`. Not every holder is in
`self.agents`:

- a newborn holds its adapter while still in `_pending_arrivals`
- a migrant holds its full footprint at the destination before the source
  releases it (§4.5)

Both are genuinely allocated and correctly count toward filling the room, but
neither can be killed. Drawing one raised `KeyError`.

This was latent from the start. `random_holder` had been in the config since
the beginning and **no run had ever used it**, so nothing exercised the path.
The mock smoke test passed because the collision is a race — it needs the pool
to be exhausted at the moment a pending arrival holds blocks.

Fixed by giving `random_holder` an `eligible` set and passing the killable
agents; when nobody else can die, scarcity falls on the requester as under the
`requester` policy. Three regression tests, verified to fail without the fix.

## What the partial data already shows

The headline prediction was that the dying agent should stop being smaller
than average. It inverted immediately:

| | context at death / room mean |
|---|---|
| requester (chr001, 26,951 deaths) | **0.63** |
| random_holder (6118397, 82 deaths) | **1.41** |
| random_holder + investment (6118523, 792 deaths) | **1.43** |

Deaths are overwhelmingly `pool_exhausted_evicted` (780 of 792), so the policy
is doing the work rather than falling back.

Offspring variance also moved, though these are early generations in a small
sample and the runs never reached steady state: V_k 18.2 and 35.1 against
chr001's 40.2.
