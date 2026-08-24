import pytest
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


def _spec(layers=4):
    dims = {m: (64, 64) for m in ("q_proj", "k_proj", "v_proj", "o_proj")}
    return spec_from_dims(layers, dims, 4, 8)


def test_site_order_is_layer_major_so_qk_and_vo_are_adjacent():
    """Chromosomal linkage is only meaningful if the site list is already in
    interaction order; this pins that ordering."""
    keys = [s.key for s in _spec(2).sites]
    assert keys == ["0.q_proj", "0.k_proj", "0.v_proj", "0.o_proj",
                    "1.q_proj", "1.k_proj", "1.v_proj", "1.o_proj"]


def test_chromosomal_crossover_switches_parent_once_per_chromosome():
    spec = _spec(28)
    rng = np.random.default_rng(0)
    p1, p2 = Genome.random(spec, 0.02, rng), Genome.random(spec, 0.02, rng)
    n = len(spec.sites)
    for chromosomes in (1, 3, 7):
        for _ in range(20):
            child = Genome.crossover(p1, p2, 0.0, rng, scheme="chromosomal",
                                     chromosomes=chromosomes)
            from_p1 = [np.array_equal(child.factors[s.key][0], p1.factors[s.key][0])
                       for s in spec.sites]
            switches = sum(a != b for a, b in zip(from_p1, from_p1[1:]))
            # each chromosome contributes at most one internal switch, plus at
            # most one at each of the chromosomes-1 boundaries
            assert switches <= 2 * chromosomes - 1
    # and it is genuinely tighter than uniform, which averages n/2
    child = Genome.crossover(p1, p2, 0.0, rng, scheme="chromosomal", chromosomes=3)
    assert n == 112


def test_chromosomal_with_one_chromosome_per_site_matches_uniform_linkage():
    spec = _spec(28)
    rng = np.random.default_rng(1)
    p1, p2 = Genome.random(spec, 0.02, rng), Genome.random(spec, 0.02, rng)
    n = len(spec.sites)
    counts = []
    for _ in range(40):
        child = Genome.crossover(p1, p2, 0.0, rng, scheme="chromosomal",
                                 chromosomes=n)
        from_p1 = [np.array_equal(child.factors[s.key][0], p1.factors[s.key][0])
                   for s in spec.sites]
        counts.append(sum(a != b for a, b in zip(from_p1, from_p1[1:])))
    assert 0.35 * n < np.mean(counts) < 0.65 * n


def test_chromosomal_crossover_is_not_biased_toward_first_parent():
    spec = _spec(28)
    rng = np.random.default_rng(2)
    p1, p2 = Genome.random(spec, 0.02, rng), Genome.random(spec, 0.02, rng)
    share = []
    for _ in range(200):
        child = Genome.crossover(p1, p2, 0.0, rng, scheme="chromosomal", chromosomes=3)
        share.append(np.mean([np.array_equal(child.factors[s.key][0],
                                             p1.factors[s.key][0])
                              for s in spec.sites]))
    assert 0.4 < np.mean(share) < 0.6


def test_multiplicative_mutation_scales_with_factor_magnitude():
    spec = _spec(2)
    rng = np.random.default_rng(3)
    p = Genome.random(spec, 0.02, rng)
    key = spec.sites[0].key
    big = {k: (a * 10, b * 10) for k, (a, b) in p.factors.items()}
    big = Genome(spec, big)

    def step(g, mode):
        child = Genome.crossover(g, g, 0.1, rng, mutation=mode)
        return float(np.std(child.factors[key][0] - g.factors[key][0]))

    # additive: same absolute step regardless of scale
    assert step(p, "additive") == pytest.approx(step(big, "additive"), rel=0.2)
    # multiplicative: step grows with the factor it perturbs
    assert step(big, "multiplicative") == pytest.approx(
        10 * step(p, "multiplicative"), rel=0.2)


def test_multiplicative_mutation_does_not_random_walk_magnitude():
    """The point of the arm: additive noise compounds RMS across generations,
    multiplicative noise (in expectation) leaves it where it was."""
    spec = _spec(8)
    rng = np.random.default_rng(4)
    for mode, grows in (("additive", True), ("multiplicative", False)):
        g = Genome.random(spec, 0.02, rng)
        start = float(np.sqrt(np.mean([np.mean(a ** 2) for a, _ in g.factors.values()])))
        for _ in range(5):
            g = Genome.crossover(g, g, 0.01, rng, mutation=mode)
        end = float(np.sqrt(np.mean([np.mean(a ** 2) for a, _ in g.factors.values()])))
        if grows:
            assert end > 1.3 * start
        else:
            assert end == pytest.approx(start, rel=0.05)


def test_unknown_scheme_and_mode_are_rejected():
    spec = _spec(2)
    rng = np.random.default_rng(5)
    p = Genome.random(spec, 0.02, rng)
    with pytest.raises(ValueError):
        Genome.crossover(p, p, 0.01, rng, scheme="blend")
    with pytest.raises(ValueError):
        Genome.crossover(p, p, 0.01, rng, mutation="lognormal")


def test_crossover_defaults_to_chromosomal():
    """Pinned deliberately: uniform crossover was measured to transmit nothing
    heritable (midparent/single-parent slope ratio 0.39, i.e. shared
    environment), while chromosomal gave 2.78 against a theoretical 2.0."""
    from evollm.config import GenomeConfig
    assert GenomeConfig().crossover == "chromosomal"
    spec = _spec(28)
    rng = np.random.default_rng(7)
    p1, p2 = Genome.random(spec, 0.02, rng), Genome.random(spec, 0.02, rng)
    child = Genome.crossover(p1, p2, 0.0, rng, chromosomes=3)
    from_p1 = [np.array_equal(child.factors[s.key][0], p1.factors[s.key][0])
               for s in spec.sites]
    assert sum(a != b for a, b in zip(from_p1, from_p1[1:])) <= 5
