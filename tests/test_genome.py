import numpy as np

from evollm.genome import Genome, spec_from_dims


def make_spec():
    return spec_from_dims(num_layers=3,
                          projections={"q_proj": (16, 16), "v_proj": (16, 8)},
                          rank=4, alpha=8)


def test_random_init_nonzero():
    spec = make_spec()
    g = Genome.random(spec, scale=0.02, rng=np.random.default_rng(0))
    for a, b in g.factors.values():
        assert np.abs(a).sum() > 0 and np.abs(b).sum() > 0


def test_zeros_is_base_model():
    g = Genome.zeros(make_spec())
    for a, b in g.factors.values():
        assert not a.any() and not b.any()


def test_crossover_inherits_wholesale_per_site():
    """§3.2: each site's factor pair comes from exactly one parent (before
    mutation) — no factor-space blending."""
    spec = make_spec()
    rng = np.random.default_rng(1)
    p1 = Genome.random(spec, 0.5, rng)
    p2 = Genome.random(spec, 0.5, rng)
    child = Genome.crossover(p1, p2, mutation_std=0.0, rng=np.random.default_rng(2))
    from_p1 = from_p2 = 0
    for key in child.factors:
        ca, cb = child.factors[key]
        if np.array_equal(ca, p1.factors[key][0]):
            assert np.array_equal(cb, p1.factors[key][1])
            from_p1 += 1
        else:
            assert np.array_equal(ca, p2.factors[key][0])
            assert np.array_equal(cb, p2.factors[key][1])
            from_p2 += 1
    assert from_p1 + from_p2 == len(spec.sites)


def test_crossover_sites_mix_across_parents():
    spec = spec_from_dims(num_layers=20, projections={"q_proj": (8, 8)},
                          rank=2, alpha=4)
    rng = np.random.default_rng(3)
    p1 = Genome.random(spec, 0.5, rng)
    p2 = Genome.random(spec, 0.5, rng)
    child = Genome.crossover(p1, p2, 0.0, np.random.default_rng(4))
    donors = {np.array_equal(child.factors[k][0], p1.factors[k][0])
              for k in child.factors}
    assert donors == {True, False}  # both parents contributed


def test_mutation_perturbs():
    spec = make_spec()
    rng = np.random.default_rng(5)
    p = Genome.random(spec, 0.5, rng)
    child = Genome.crossover(p, p, mutation_std=0.01, rng=rng)
    for key in child.factors:
        diff = child.factors[key][0] - p.factors[key][0]
        assert 0 < np.abs(diff).max() < 0.1


def test_save_load_roundtrip(tmp_path):
    spec = make_spec()
    g = Genome.random(spec, 0.1, np.random.default_rng(6))
    g.save(tmp_path / "g.safetensors")
    loaded = Genome.load(tmp_path / "g.safetensors", spec)
    for key in g.factors:
        assert np.array_equal(g.factors[key][0], loaded.factors[key][0])
        assert np.array_equal(g.factors[key][1], loaded.factors[key][1])


def test_adapter_bytes_uniform():
    spec = make_spec()
    # rank * (in + out) summed over sites, fp16
    expected = sum(4 * (s.in_dim + s.out_dim) for s in spec.sites) * 2
    assert spec.adapter_bytes() == expected
