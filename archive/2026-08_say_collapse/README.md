# Archived: broadcast speech, and a fatal memory bound (2026-08-25/26)

Two runs, archived for opposite reasons. Neither is wrong; one is a decisive
negative result and the other never started.

## `node_4room_7b_chr001_say_6131724` — broadcast kills a dense room

`chr001_evict` with `tell` replaced by `say`. It ran to 400,000 steps in every
room, in two hours, because there was almost nothing to compute.

**129 births in total: 128 seeds and one child.** Population collapsed to 1.0
agent per room, each holding a ~599,000-token context. Maximum generation 1.

A broadcast is paid by every listener (§2.4), so at this run's density one
utterance costs the room roughly sixty times what a directed `tell` does. The
room was consumed by speech before reproduction could establish.

This says nothing about whether agents *want* to communicate — only that they
cannot afford to at ~60 agents per room. Testing that properly needs small
rooms, which is what `node_40room_7b_villages.yaml` does.

## `node_4room_7b_mlp_6129547` — engine never started

Failed after 68 seconds. `_pool_upper_bound` computed the ceiling for
`max_model_len: auto` without subtracting the model weights, on the reasoning
that overshooting only wasted rope-cache memory. It does not: vLLM refuses to
start unless a single request of `max_model_len` fits in the KV cache.

    ValueError: To serve at least one request with the model's max seq len
    (1645057), 87.86 GiB KV cache is needed, which is larger than the
    available KV cache memory (71.03 GiB)

Fixed by subtracting the weights and a reserve for activations and CUDA
graphs, with a regression test that fails on the old logic. Resubmitted.
