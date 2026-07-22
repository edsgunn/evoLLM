import numpy as np

from evollm.blocks import BlockPool, adapter_blocks_needed


def test_kv_blocks_ceil():
    pool = BlockPool(capacity=10, block_size=4)
    assert pool.kv_blocks_for(0) == 0
    assert pool.kv_blocks_for(1) == 1
    assert pool.kv_blocks_for(4) == 1
    assert pool.kv_blocks_for(5) == 2


def test_adapter_reserved_before_kv():
    pool = BlockPool(capacity=3, block_size=4)
    assert pool.try_reserve_adapter("a", 2)
    assert pool.free == 1
    assert not pool.try_reserve_adapter("b", 2)  # birth fails on adapter blocks
    assert pool.free == 1


def test_kv_growth_and_exhaustion():
    pool = BlockPool(capacity=2, block_size=4)
    assert pool.try_reserve_adapter("a", 1)
    # tokens 1..4 fit in one block
    for t in range(1, 5):
        assert pool.try_grow_kv("a", t)
    assert pool.free == 0
    # token 5 needs a second KV block: pool empty -> scarcity event
    assert pool.kv_needs_block("a", 5)
    assert not pool.try_grow_kv("a", 5)
    # failure changes nothing
    assert pool.holdings["a"].kv_blocks == 1


def test_release_returns_blocks():
    pool = BlockPool(capacity=4, block_size=4)
    pool.try_reserve_adapter("a", 2)
    pool.try_grow_kv("a", 8)
    assert pool.free == 0
    pool.release_all("a")
    assert pool.free == 4


def test_random_holder_weighted():
    pool = BlockPool(capacity=100, block_size=4)
    pool.try_reserve_adapter("small", 1)
    pool.try_reserve_adapter("big", 1)
    pool.try_grow_kv("big", 4 * 50)
    rng = np.random.default_rng(0)
    picks = [pool.random_holder(rng) for _ in range(200)]
    # "big" holds 51 of 52 blocks; it should be picked overwhelmingly often
    assert picks.count("big") > 180


def test_adapter_blocks_needed():
    assert adapter_blocks_needed(1, 100) == 1
    assert adapter_blocks_needed(100, 100) == 1
    assert adapter_blocks_needed(101, 100) == 2
