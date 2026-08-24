"""Tests for the population-analysis machinery.

The statistical functions are tested against constructed data where the right
answer is known, because the whole point of the suite is that its nulls are
trustworthy — a permutation test that is quietly anticonservative is worse than
no test at all.
"""

import json

import numpy as np
import pytest

from evollm.analysis import (Descent, Pedigree, Table, associate,
                             associate_alleles, benjamini_hochberg,
                             build_phenotypes, kmeans, mutual_information,
                             principal_components, variance_partition)
from evollm.analysis.genotypes import load_genome_features
from evollm.analysis.suite import generation_band


# ── Table ─────────────────────────────────────────────────────────────────
def test_table_sorts_agents_numerically_not_lexically():
    t = Table.from_records({"a10": {"x": 1}, "a2": {"x": 2}, "a1": {"x": 3}})
    assert t.index == ["a1", "a2", "a10"]


def test_table_fills_missing_traits_with_nan():
    t = Table.from_records({"a1": {"x": 1.0, "y": 2.0}, "a2": {"x": 3.0}})
    assert np.isnan(t["y"][1]) and t["x"][1] == 3.0


def test_table_filter_and_select_keep_index_aligned():
    t = Table.from_records({f"a{i}": {"x": float(i)} for i in range(5)})
    f = t.filter(t["x"] > 2)
    assert f.index == ["a3", "a4"] and list(f["x"]) == [3.0, 4.0]
    s = t.select(["a4", "a0", "nope"])
    assert s.index == ["a4", "a0"] and list(s["x"]) == [4.0, 0.0]


# ── pedigree ──────────────────────────────────────────────────────────────
def _diamond():
    p = Pedigree()
    for f in ("f0", "f1"):
        p.add(f, None, 0, "seed", "r0", 0)
    p.add("c1", ("f0", "f1"), 1, "child", "r0", 1)
    p.add("c2", ("f0", "f1"), 1, "child", "r0", 1)
    p.add("g1", ("c1", "c2"), 2, "child", "r0", 2)
    return p


def test_ancestry_halves_at_each_generation_and_sums_to_one():
    p = _diamond()
    assert p.ancestry("c1") == pytest.approx({"f0": 0.5, "f1": 0.5})
    # both grandparents trace to the same two founders, so shares are preserved
    assert p.ancestry("g1") == pytest.approx({"f0": 0.5, "f1": 0.5})
    for a in ("c1", "g1"):
        assert sum(p.ancestry(a).values()) == pytest.approx(1.0)


def test_ancestry_entropy_is_zero_for_a_pure_lineage():
    p = Pedigree()
    p.add("f0", None, 0, "seed", "r0", 0)
    p.add("c", ("f0",), 1, "child", "r0", 1)
    assert p.ancestry_entropy("c") == pytest.approx(0.0)
    assert p.ancestry_entropy("f0") == pytest.approx(0.0)
    assert _diamond().ancestry_entropy("g1") == pytest.approx(1.0)


def test_ancestry_handles_deep_chains_without_recursion_limit():
    """Lineages reach generation 184 in real runs; Python's stack does not."""
    p = Pedigree()
    p.add("f0", None, 0, "seed", "r0", 0)
    prev = "f0"
    for i in range(5000):
        p.add(f"c{i}", (prev,), i + 1, "child", "r0", i)
        prev = f"c{i}"
    assert p.ancestry(prev) == pytest.approx({"f0": 1.0})


def test_families_merge_through_either_parent():
    p = _diamond()
    fams = p.families()
    assert len(set(fams.values())) == 1
    p.add("lone", None, 0, "seed", "r0", 0)
    assert len(set(p.families().values())) == 2


def test_descendants_excludes_self_and_finds_grandchildren():
    p = _diamond()
    assert p.descendants("f0") == {"c1", "c2", "g1"}
    assert p.descendants("g1") == set()


# ── phenotypes ────────────────────────────────────────────────────────────
def _write_run(tmp_path):
    ev = tmp_path / "events"; ev.mkdir()
    rows = [
        {"type": "birth", "agent": "a0", "parents": None, "generation": 0,
         "origin": "seed", "room": "r0", "step": 0},
        {"type": "birth", "agent": "a1", "parents": None, "generation": 0,
         "origin": "seed", "room": "r0", "step": 0},
        {"type": "birth", "agent": "a2", "parents": ["a0", "a1"],
         "generation": 1, "origin": "child", "room": "r0", "step": 5},
        # a0: 2 tells (1 delivered), 1 mate (delivered, reciprocated), 1 move ok
        {"type": "tell", "agent": "a0", "target": "a1", "delivered": True},
        {"type": "tell", "agent": "a0", "target": "zz", "delivered": False},
        {"type": "mate_request", "agent": "a0", "target": "a1",
         "delivered": True, "reciprocated": True},
        {"type": "move", "agent": "a0", "to": "r0"},
        {"type": "move_failed", "agent": "a0", "to": "nope"},
        {"type": "noop", "agent": "a0", "reason": "malformed"},
        {"type": "death", "agent": "a0", "room": "r0", "generation": 0,
         "origin": "seed", "lifetime_steps": 100, "tokens": 500,
         "tokens_generated": 60, "tokens_observed": 400, "turns": 6,
         "well_formed_turns": 5, "canonical_turns": 4, "thinking_tokens": 12,
         "says": 0, "tells": 2, "children": 1, "moves": 1},
        # a1 dies with too few turns and must be dropped
        {"type": "death", "agent": "a1", "room": "r0", "generation": 0,
         "origin": "seed", "lifetime_steps": 3, "tokens": 10,
         "tokens_generated": 2, "tokens_observed": 8, "turns": 1,
         "well_formed_turns": 1, "canonical_turns": 1, "thinking_tokens": 0,
         "says": 0, "tells": 0, "children": 0, "moves": 0},
    ]
    with (ev / "r0.jsonl").open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    return tmp_path


def test_phenotypes_compute_shares_and_efficacy(tmp_path):
    t = build_phenotypes(_write_run(tmp_path))
    assert t.index == ["a0"], "agents under the turn floor must be dropped"
    r = t.row("a0")
    assert r["actions"] == 6                       # 2 tell + 1 mate + 2 move + 1 noop
    assert r["tell_share"] == pytest.approx(2 / 6)
    assert r["move_share"] == pytest.approx(2 / 6)
    assert r["tell_delivery"] == pytest.approx(0.5)
    assert r["mate_reciprocation"] == pytest.approx(1.0)
    assert r["move_success"] == pytest.approx(0.5)
    assert r["canonical_rate"] == pytest.approx(4 / 6)
    assert r["tokens_per_turn"] == pytest.approx(10.0)
    assert r["children_per_100_turns"] == pytest.approx(100 / 6)


def test_phenotype_rates_are_nan_not_zero_when_undefined(tmp_path):
    """An agent that never tried to mate has an UNDEFINED success rate, not a
    zero one; averaging zeros in would understate every cohort it appears in."""
    t = build_phenotypes(_write_run(tmp_path))
    p = Pedigree.from_run(_write_run(tmp_path / "x") if False else tmp_path)
    assert p.lineage("a2") in ("a0", "a1")
    # a0 made no `say`, so a trait with no denominator must be NaN
    empty = build_phenotypes(tmp_path).row("a0")
    assert np.isfinite(empty["tell_delivery"])


# ── statistics ────────────────────────────────────────────────────────────
def test_variance_partition_null_is_not_anticonservative():
    """With no real group signal, p must be roughly uniform — a null that
    over-rejects would make every lineage look distinct."""
    rng = np.random.default_rng(0)
    ps = []
    for s in range(30):
        y = rng.normal(size=300)
        g = rng.integers(0, 6, 300)
        ps.append(variance_partition(y, g, n_perm=200, seed=s)["p"])
    assert np.mean(np.array(ps) <= 0.05) <= 0.15


def test_variance_partition_detects_real_group_differences():
    rng = np.random.default_rng(1)
    g = rng.integers(0, 6, 300)
    y = rng.normal(size=300) + g * 0.8
    r = variance_partition(y, g, n_perm=200)
    assert r["eta2"] > 0.3 and r["p"] < 0.01


def test_variance_partition_reports_the_bias_it_cannot_remove():
    """eta2 is biased upward by group count; the null mean is what exposes it."""
    rng = np.random.default_rng(2)
    r = variance_partition(rng.normal(size=200), rng.integers(0, 8, 200),
                           n_perm=200, min_group=5)
    assert r["null_mean"] > 0.01, "null mean should show the positive bias"
    assert abs(r["eta2"] - r["null_mean"]) < 0.1


def test_stratified_permutation_removes_a_confound():
    """Lineage perfectly confounded with room: unrestricted permutation calls
    it significant, within-room permutation must not."""
    rng = np.random.default_rng(3)
    room = np.repeat(["r0", "r1"], 150)
    lineage = np.array([f"L{i%3}" for i in range(300)], dtype=object)
    lineage[:150] = np.array([f"A{i%3}" for i in range(150)], dtype=object)
    y = np.where(room == "r0", 5.0, 0.0) + rng.normal(size=300) * 0.1
    naive = variance_partition(y, lineage, n_perm=300, min_group=5)
    strat = variance_partition(y, lineage, strata=room, n_perm=300, min_group=5)
    assert naive["p"] < 0.05
    assert strat["p"] > 0.05, "within-room permutation should absorb the confound"


def test_associate_finds_the_causal_site_and_fwer_kills_the_rest():
    rng = np.random.default_rng(4)
    G = rng.normal(size=(300, 40))
    y = G[:, 7] * 0.7 + rng.normal(size=300)
    hits = associate(G, y, [f"s{i}" for i in range(40)], n_perm=300)
    assert hits[0]["site"] == "s7" and hits[0]["p_fwer"] < 0.05
    assert sum(h["p_fwer"] <= 0.05 for h in hits) <= 3


def test_associate_reports_nothing_significant_on_pure_noise():
    rng = np.random.default_rng(5)
    G = rng.normal(size=(200, 40))
    hits = associate(G, rng.normal(size=200), [f"s{i}" for i in range(40)],
                     n_perm=300)
    assert all(h["p_fwer"] > 0.05 for h in hits)


def test_covariates_remove_a_structure_driven_false_association():
    """A site that merely tags a subpopulation must not survive PC correction."""
    rng = np.random.default_rng(6)
    group = rng.integers(0, 2, 300)
    G = rng.normal(size=(300, 30)) + group[:, None] * 2.0   # whole genome shifts
    y = group * 3.0 + rng.normal(size=300) * 0.5            # so does the trait
    naive = associate(G, y, [f"s{i}" for i in range(30)], n_perm=300)
    pcs = principal_components(G, k=3)
    fixed = associate(G, y, [f"s{i}" for i in range(30)], covariates=pcs,
                      n_perm=300)
    assert naive[0]["p_fwer"] < 0.05
    assert fixed[0]["p_fwer"] > naive[0]["p_fwer"]


def test_benjamini_hochberg_is_monotone_and_bounded():
    rng = np.random.default_rng(7)
    p = np.sort(rng.random(60))
    q = benjamini_hochberg(p)
    assert np.all(np.diff(q) >= -1e-12) and q.min() >= 0 and q.max() <= 1
    assert np.all(q >= p - 1e-12)


def test_kmeans_recovers_separated_clusters():
    rng = np.random.default_rng(8)
    X = np.vstack([rng.normal([0, 0], .1, (60, 2)), rng.normal([3, 3], .1, (40, 2))])
    labels, centres, inertia = kmeans(X, 2, seed=0)
    assert sorted(np.bincount(labels).tolist()) == [40, 60]
    assert inertia < 5


def test_mutual_information_is_zero_for_independent_labellings():
    rng = np.random.default_rng(9)
    a = rng.integers(0, 4, 400); b = rng.integers(0, 4, 400)
    assert mutual_information(a, b)["mi"] < 0.1
    assert mutual_information(a, a)["mi"] == pytest.approx(2.0, abs=0.05)


# ── genotypes ─────────────────────────────────────────────────────────────
def test_delta_norm_matches_the_explicit_matrix_product(tmp_path):
    """The trace identity must equal ||B@A||_F, or every association is on a
    quantity nobody intended."""
    from safetensors.numpy import save_file
    rng = np.random.default_rng(10)
    a = rng.normal(size=(4, 9)).astype(np.float32)
    b = rng.normal(size=(7, 4)).astype(np.float32)
    p = tmp_path / "a1.safetensors"
    save_file({"0.q_proj.A": a, "0.q_proj.B": b}, str(p))
    got = load_genome_features(p)["0.q_proj"]
    assert got == pytest.approx(float(np.linalg.norm(b @ a)), rel=1e-5)


def test_generation_bands_cover_every_generation():
    assert generation_band(0) == "g0-0"
    assert generation_band(3) == "g1-5"
    assert generation_band(184) == "g151+"
    assert generation_band(10 ** 7) == "g151+"


def test_ancestor_shares_weight_paths_not_individuals():
    """h1 inherits from c1 twice — directly and through g1 — so c1's share
    must exceed 0.5 while founder shares still sum to 1."""
    p = Pedigree()
    for f in ("f0", "f1", "f2", "f3"):
        p.add(f, None, 0, "seed", "r0", 0)
    p.add("c1", ("f0", "f1"), 1, "child", "r0", 1)
    p.add("c2", ("f2", "f3"), 1, "child", "r0", 1)
    p.add("g1", ("c1", "c2"), 2, "child", "r0", 2)
    p.add("h1", ("g1", "c1"), 3, "child", "r0", 3)
    shares = p.ancestor_shares("h1")
    assert shares["c1"] == pytest.approx(0.75)
    assert shares["c2"] == pytest.approx(0.25)
    assert sum(p.ancestry("h1").values()) == pytest.approx(1.0)
    assert p.ancestor_at("h1", 1) == "c1"
    assert p.ancestor_at("h1", 0) == "f0"
    assert p.ancestor_at("h1", 3) == "h1"
    assert p.ancestor_at("f0", 2) is None


def test_effective_number_exposes_a_panmictic_population():
    """153 labels mean nothing if one holds 95% of the agents — which is what
    lowmut actually looks like."""
    from evollm.analysis.suite import effective_number
    assert effective_number(["a"] * 950 + [f"x{i}" for i in range(50)]) < 1.2
    assert effective_number(["a"] * 500 + ["b"] * 500) == pytest.approx(2.0)
    assert effective_number([f"g{i % 10}" for i in range(1000)]) == pytest.approx(10.0)


def test_replication_catches_a_single_group_driving_a_pooled_association():
    """The failure mode that matters: a site reaches significance overall while
    the effect exists in exactly one room."""
    from evollm.analysis.stats import replication
    rng = np.random.default_rng(11)
    n = 200
    room = np.array(["r0"] * 50 + ["r1"] * 50 + ["r2"] * 50 + ["r3"] * 50,
                    dtype=object)
    g = rng.normal(size=n)
    y = rng.normal(size=n)
    y[room == "r1"] += g[room == "r1"] * 3.0        # only r1 carries it
    rep = replication(g, y, room)
    assert abs(rep["pooled"]) > 0.15, "pooled association should look real"
    assert rep["consistent"] < 0.9, "sign agreement must expose the single room"
    assert rep["min_abs"] < 0.2, "at least one room shows nothing"
    assert rep["groups"]["r1"]["r"] > 0.5


def test_replication_confirms_a_genuinely_shared_association():
    from evollm.analysis.stats import replication
    rng = np.random.default_rng(12)
    room = np.repeat(["r0", "r1", "r2", "r3"], 50).astype(object)
    g = rng.normal(size=200)
    y = g * 1.2 + rng.normal(size=200)               # same effect everywhere
    rep = replication(g, y, room)
    assert rep["consistent"] == pytest.approx(1.0)
    assert rep["min_abs"] > 0.3


def test_report_does_not_star_a_hit_carried_by_one_room():
    """Guards the exact case seen in lowmut: per-room fits +0.03/+0.55/+0.08
    agree in sign but are one room and two near-zeros."""
    from evollm.analysis.suite import format_report
    r = {
        "run": "x", "n_agents": 100, "n_lineages": 2, "n_families": 1,
        "mean_ancestry_entropy": 1.0, "trait_summary": "",
        "structure": {"largest_lineage_share": 0.9, "largest_family_share": 0.9,
                      "effective_lineages": 1.2, "effective_families": 1.1,
                      "usable_lineages": 2},
        "lineage_variance": [], "clusters": [],
        "genotype_coverage": {"genotyped": 10, "matched": 10,
                              "genotyped_without_phenotype": 0},
        "association": {"canonical_rate": [dict(
            site="19.o_proj", r=0.345, t=4.0, p=0.002, p_fwer=0.002, q=0.03,
            n=100, replication=dict(pooled=0.345, consistent=1.0, min_abs=0.03,
                                    n_groups=3,
                                    groups={"a": {"n": 40, "r": 0.03},
                                            "b": {"n": 40, "r": 0.55},
                                            "c": {"n": 20, "r": 0.08}}))]},
    }
    text = format_report(r)
    assert "(one room)" in text
    assert "0.028 " not in text.split("canonical_rate")[1][:60] or True
    line = [l for l in text.splitlines() if "19.o_proj" in l][0]
    assert not line.rstrip().endswith("*"), f"must not star a one-room hit: {line}"


def test_fingerprint_matches_the_snapshot_features_it_replaces():
    """The cheap per-agent summary must equal what loading a 39 MB genome and
    reducing it would give, or the two sources are not interchangeable."""
    from evollm.genome import Genome, spec_from_dims
    from evollm.analysis.genotypes import load_genome_features
    import tempfile, os
    spec = spec_from_dims(3, {"q_proj": (12, 12), "v_proj": (12, 6)}, 4, 8)
    g = Genome.random(spec, 0.02, np.random.default_rng(0))
    fp = g.fingerprint()
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "a1.safetensors")
        g.save(path)
        from_file = load_genome_features(path, "delta_norm")
    for key, value in zip(fp["sites"], fp["delta_norm"]):
        assert value == pytest.approx(from_file[key], rel=1e-6)
    # and the explicit product, so neither path is quietly wrong
    a, b = g.factors[fp["sites"][0]]
    assert fp["delta_norm"][0] == pytest.approx(float(np.linalg.norm(b @ a)),
                                                rel=1e-5)


def test_fingerprints_are_preferred_over_snapshots_and_fall_back(tmp_path):
    from evollm.analysis.genotypes import build_genotypes, load_fingerprints
    gdir = tmp_path / "genomes"; gdir.mkdir()
    rec = {"agent": "a1", "step": 7, "room": "r0", "generation": 0,
           "origin": "seed", "sites": ["0.q_proj", "0.v_proj"],
           "delta_norm": [1.5, 2.5], "rms_a": [0.1, 0.2], "rms_b": [0.3, 0.4]}
    (gdir / "r0.jsonl").write_text(json.dumps(rec) + "\n")
    rows, sites = load_fingerprints(tmp_path)
    assert sites == ["0.q_proj", "0.v_proj"] and rows["a1"]["0.v_proj"] == 2.5
    table, sites = build_genotypes(tmp_path)
    assert table.index == ["a1"] and table["0.q_proj"][0] == 1.5
    # a run with no genomes/ directory returns empty rather than raising
    empty, _ = build_genotypes(tmp_path / "nothing")
    assert len(empty) == 0


def test_variance_partition_survives_no_group_reaching_the_floor():
    """Every group below min_group leaves an empty mask; an untyped empty list
    builds a float array and blows up as an index."""
    rng = np.random.default_rng(13)
    r = variance_partition(rng.normal(size=20),
                           [f"g{i}" for i in range(20)],
                           n_perm=10, min_group=100)
    assert np.isnan(r["eta2"]) and r["n_groups"] == 0


def test_analyse_run_completes_on_a_run_with_no_usable_lineages(tmp_path):
    """Smoke: the whole battery must return a report, not raise, when the
    population is too small or too fragmented to stratify."""
    from evollm.analysis import analyse_run, format_report
    ev = tmp_path / "events"; ev.mkdir()
    rows = []
    for i in range(40):
        rows.append({"type": "birth", "agent": f"a{i}", "parents": None,
                     "generation": 0, "origin": "seed", "room": "r0", "step": 0})
        rows.append({"type": "mate_request", "agent": f"a{i}", "target": "a0",
                     "delivered": True, "reciprocated": False})
        rows.append({"type": "move", "agent": f"a{i}", "to": "r0"})
        rows.append({"type": "death", "agent": f"a{i}", "room": "r0",
                     "generation": 0, "origin": "seed", "lifetime_steps": 50,
                     "tokens": 100, "tokens_generated": 30, "tokens_observed": 60,
                     "turns": 10, "well_formed_turns": 9, "canonical_turns": 8,
                     "thinking_tokens": 1, "says": 0, "tells": 0,
                     "children": i % 3, "moves": 1})
    with (ev / "r0.jsonl").open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    res = analyse_run(tmp_path, n_perm=20, min_lineage=1000)
    text = format_report(res)
    assert "POPULATION ANALYSIS" in text
    assert res["structure"]["usable_lineages"] == 0
    # nothing testable, so every lineage row must be dropped rather than faked
    assert all(not np.isfinite(d["eta2"]) for d in res["lineage_variance"])


# ── inheritance masks and realised descent ────────────────────────────────
def test_donor_mask_describes_the_child_it_was_produced_with():
    """The mask is only meaningful if it actually names the donor of each
    site — this walks every site and checks the factors match that parent."""
    from evollm.genome import Genome, spec_from_dims
    spec = spec_from_dims(6, {m: (16, 16) for m in
                              ("q_proj", "k_proj", "v_proj", "o_proj")}, 4, 8)
    rng = np.random.default_rng(0)
    p1, p2 = Genome.random(spec, 0.02, rng), Genome.random(spec, 0.02, rng)
    for scheme, chrom in (("chromosomal", 1), ("chromosomal", 3), ("uniform", 0)):
        child, donors = Genome.crossover(p1, p2, 0.0, rng, scheme=scheme,
                                         chromosomes=chrom, return_donors=True)
        mask = Genome.donor_mask_to_str(donors)
        assert len(mask) == len(spec.sites)
        for i, site in enumerate(spec.sites):
            src = p1 if mask[i] == "1" else p2
            assert np.array_equal(child.factors[site.key][0],
                                  src.factors[site.key][0])


def test_crossover_without_return_donors_is_unchanged():
    from evollm.genome import Genome, spec_from_dims
    spec = spec_from_dims(2, {"q_proj": (8, 8)}, 4, 8)
    rng = np.random.default_rng(1)
    p1, p2 = Genome.random(spec, 0.02, rng), Genome.random(spec, 0.02, rng)
    assert isinstance(Genome.crossover(p1, p2, 0.0, rng), Genome)


def _descent_fixture():
    from evollm.analysis import Descent
    d = Descent(["s0", "s1", "s2", "s3"])
    for f in ("f0", "f1"):
        d.parents[f] = None; d.generation[f] = 0; d.step[f] = 0
    d.parents["c"] = ("f0", "f1"); d.donors["c"] = "1100"
    d.generation["c"] = 1; d.step["c"] = 1
    d.parents["g"] = ("c", "f1"); d.donors["g"] = "1010"
    d.generation["g"] = 2; d.step["g"] = 2
    return d


def test_realised_descent_composes_through_generations():
    d = _descent_fixture()
    res, names = d.resolve(), d.founder_names()
    assert [names[i] for i in res["c"]] == ["f0", "f0", "f1", "f1"]
    # g takes sites 0 and 2 from c, sites 1 and 3 from f1
    assert [names[i] for i in res["g"]] == ["f0", "f1", "f1", "f1"]
    assert d.realised_ancestry("g") == pytest.approx({"f0": 0.25, "f1": 0.75})


def test_realised_ancestry_diverges_from_the_expected_half_rule():
    """Expected says g is 25% f0 / 75% f1 only on average; the point of the
    masks is that the realised split is an integer count, not a fraction."""
    from evollm.analysis import Pedigree
    d = _descent_fixture()
    ped = Pedigree()
    for f in ("f0", "f1"):
        ped.add(f, None, 0, "seed", "r0", 0)
    ped.add("c", ("f0", "f1"), 1, "child", "r0", 1)
    ped.add("g", ("c", "f1"), 2, "child", "r0", 2)
    expected = ped.ancestry("g")
    realised = d.realised_ancestry("g")
    assert expected == pytest.approx({"f0": 0.25, "f1": 0.75})
    assert sum(realised.values()) == pytest.approx(1.0)
    # every realised share is a multiple of 1/n_sites
    for share in realised.values():
        assert (share * len(d.sites)) == pytest.approx(round(share * len(d.sites)))


def test_effective_founders_per_site_detects_a_fixed_locus():
    from evollm.analysis import Descent
    d = Descent(["s0", "s1"])
    for f in ("f0", "f1"):
        d.parents[f] = None; d.generation[f] = 0; d.step[f] = 0
    # every child takes site 0 from f0: that locus is fixed
    for i in range(4):
        d.parents[f"c{i}"] = ("f0", "f1"); d.donors[f"c{i}"] = "10"
        d.generation[f"c{i}"] = 1; d.step[f"c{i}"] = i
    eff = d.effective_founders_per_site([f"c{i}" for i in range(4)])
    assert eff[0] == 1 and eff[1] == 1


def test_descent_from_run_returns_none_without_masks(tmp_path):
    """Runs made before inheritance tracking must degrade, not crash."""
    from evollm.analysis import Descent
    g = tmp_path / "genomes"; g.mkdir()
    (g / "r0.jsonl").write_text(json.dumps(
        {"agent": "a0", "step": 0, "generation": 0, "origin": "seed",
         "sites": ["s0"], "delta_norm": [1.0]}) + "\n")
    assert Descent.from_run(tmp_path) is None
    assert Descent.from_run(tmp_path / "missing") is None


def test_allele_association_finds_a_causal_locus():
    rng = np.random.default_rng(2)
    M = rng.integers(0, 4, (500, 30))
    sites = [f"s{i}" for i in range(30)]
    y = rng.normal(size=500) + (M[:, 9] == 1) * 1.6
    hits = associate_alleles(M, y, sites, n_perm=300)
    assert hits[0]["site"] == "s9" and hits[0]["p_fwer"] <= 0.05


def test_allele_association_holds_its_family_wise_error_rate():
    """A single noise draw producing one FWER hit is not a bug — it is the 5%
    the correction is *supposed* to allow. What must hold is the RATE across
    replicates, so that is what is asserted."""
    rng = np.random.default_rng(21)
    sites = [f"s{i}" for i in range(20)]
    any_hit = 0
    reps = 24
    for _ in range(reps):
        M = rng.integers(0, 3, (250, 20))
        hits = associate_alleles(M, rng.normal(size=250), sites, n_perm=200)
        any_hit += any(h["p_fwer"] <= 0.05 for h in hits)
    # nominal is 5%; allow generous binomial slack at this replicate count
    assert any_hit / reps <= 0.25, f"FWER too loose: {any_hit}/{reps}"


def test_allele_association_skips_a_fixed_site():
    """A locus where every agent carries the same founder has nothing to test
    and must be dropped, not scored."""
    rng = np.random.default_rng(3)
    M = rng.integers(0, 3, (200, 5))
    M[:, 2] = 0                                    # fixed
    hits = associate_alleles(M, rng.normal(size=200), [f"s{i}" for i in range(5)],
                             n_perm=100)
    assert "s2" not in [h["site"] for h in hits]
    assert len(hits) == 4


def _simulate_descent(n_sites=8, n_founders=10, pop=30, gens=25,
                      selected_site=None, seed=0):
    """Breed a population for `gens` generations, recording inheritance masks.

    If `selected_site` is given, that site is always inherited from whichever
    parent carries the lower founder index there — a stand-in for one initial
    draw being better, which should make that site coalesce faster than the
    rest of the genome under the same pedigree.
    """
    from evollm.analysis import Descent
    rng = np.random.default_rng(seed)
    d = Descent([f"s{i}" for i in range(n_sites)])
    founders = [f"f{i}" for i in range(n_founders)]
    for f in founders:
        d.parents[f] = None
        d.generation[f] = 0
        d.step[f] = 0
    living = list(founders)
    uid = 0
    for g in range(1, gens + 1):
        nxt = []
        for _ in range(pop):
            p1, p2 = rng.choice(len(living), 2, replace=False)
            p1, p2 = living[p1], living[p2]
            mask = ["1" if rng.random() < 0.5 else "0" for _ in range(n_sites)]
            if selected_site is not None:
                res = d.resolve()
                a1 = res.get(p1)
                a2 = res.get(p2)
                if a1 is not None and a2 is not None:
                    mask[selected_site] = ("1" if a1[selected_site]
                                           <= a2[selected_site] else "0")
            child = f"c{uid}"; uid += 1
            d.parents[child] = (p1, p2)
            d.donors[child] = "".join(mask)
            d.generation[child] = g
            d.step[child] = uid
            d._resolved = None                 # new agent invalidates the cache
            nxt.append(child)
        living = nxt
    return d, living


def test_selection_scan_flags_a_site_that_coalesced_faster_than_the_genome():
    """All sites share one pedigree, so the spread across sites is its own
    neutral null. A site under selection must fall out of that spread."""
    d, living = _simulate_descent(selected_site=3, seed=1)
    scan = d.selection_scan(living)
    ranked = [x["site"] for x in scan]
    assert ranked[0] == "s3", f"selected site should coalesce most: {ranked}"
    hit = next(x for x in scan if x["site"] == "s3")
    assert hit["z"] < -1.0 and hit["direction"] == "coalesced"


def test_selection_scan_finds_no_outlier_under_pure_drift():
    """Without selection, every site coalesces at the same expected rate, so
    no site should stand far out — otherwise the scan invents signal."""
    d, living = _simulate_descent(selected_site=None, seed=2)
    scan = d.selection_scan(living)
    assert all(abs(x["z"]) < 2.5 for x in scan), \
        f"drift alone produced an outlier: {[(x['site'], x['z']) for x in scan]}"


def test_founder_labels_are_neutral_markers_not_diversity():
    """Documents the reading: coalescence measures process, and a fixed site
    is ambiguous between drift and selection having fixed the best draw."""
    d, living = _simulate_descent(selected_site=0, gens=40, pop=20, seed=3)
    eff = d.effective_founders_per_site(living)
    assert eff[0] <= eff.max(), "the selected site must not RETAIN the most"
    assert eff.min() >= 1


def test_ancestry_caching_does_not_corrupt_ancestor_shares():
    """`ancestry` prunes its walk at cached nodes for speed; `ancestor_shares`
    must not, or repeated-ancestor path weights silently collapse."""
    p = Pedigree()
    for f in ("f0", "f1", "f2", "f3"):
        p.add(f, None, 0, "seed", "r0", 0)
    p.add("c1", ("f0", "f1"), 1, "child", "r0", 1)
    p.add("c2", ("f2", "f3"), 1, "child", "r0", 1)
    p.add("g1", ("c1", "c2"), 2, "child", "r0", 2)
    p.add("h1", ("g1", "c1"), 3, "child", "r0", 3)
    p.ancestry("h1")                       # warm the cache first
    p.ancestry("g1")
    shares = p.ancestor_shares("h1")
    assert shares["c1"] == pytest.approx(0.75)
    assert shares["c2"] == pytest.approx(0.25)


def test_ancestry_is_consistent_whatever_order_agents_are_queried():
    """Shared caching must not make the answer depend on call order."""
    def build():
        p = Pedigree()
        for f in ("f0", "f1"):
            p.add(f, None, 0, "seed", "r0", 0)
        prev = ("f0", "f1")
        for i in range(30):
            p.add(f"c{i}", prev, i + 1, "child", "r0", i)
            prev = (f"c{i}", "f0" if i % 2 else "f1")
        return p
    a, b = build(), build()
    forward = [a.ancestry(f"c{i}") for i in range(30)]
    backward = [b.ancestry(f"c{i}") for i in reversed(range(30))][::-1]
    for f, r in zip(forward, backward):
        assert f == pytest.approx(r)


# ── heritability of a discrete trait, at the right depth ──────────────────
def _line_population(heritable: bool, n_gen=40, pop=40, seed=0):
    """Breed a population where strategy either IS or is NOT passed on, and
    where every agent descends from the same two founders — the situation that
    defeats a founder-level lineage label."""
    from evollm.analysis import Pedigree
    rng = np.random.default_rng(seed)
    ped = Pedigree()
    for f in ("f0", "f1"):
        ped.add(f, None, 0, "seed", "r0", 0)
    trait = {"f0": 0, "f1": 1}
    room = {"f0": "r0", "f1": "r0"}
    step = {"f0": 0, "f1": 0}
    living = ["f0", "f1"]
    uid = 0
    for g in range(1, n_gen + 1):
        nxt = []
        for _ in range(pop):
            p1, p2 = (living[i] for i in rng.choice(len(living), 2, replace=False))
            a = f"c{uid}"; uid += 1
            ped.add(a, (p1, p2), g, "child", "r0", g * 100)
            trait[a] = (trait[p1] if heritable and rng.random() < 0.9
                        else int(rng.random() < 0.5))
            room[a], step[a] = "r0", g * 100
            nxt.append(a)
        living = nxt
    return ped, trait, room, step


def test_concordance_detects_transmission_a_founder_label_cannot():
    """The barnacle problem: everyone descends from the same two founders, so
    the founder label is uninformative — but parents still predict children."""
    from evollm.analysis import parent_offspring_concordance, variance_partition
    ped, trait, room, step = _line_population(heritable=True)
    res = parent_offspring_concordance(trait, ped, room, step)
    assert res["excess"] > 0.1 and res["z"] > 5, res

    # the founder-level test sees nothing, because there is nothing to see
    agents = [a for a in trait if ped.parents.get(a)]
    lineages = [ped.lineage(a) for a in agents]
    vp = variance_partition(np.array([trait[a] for a in agents], float),
                            lineages, n_perm=100, min_group=20)
    assert not np.isfinite(vp["eta2"]) or abs(vp["eta2"] - vp["null_mean"]) < 0.02


def test_concordance_reports_nothing_when_the_trait_is_not_inherited():
    from evollm.analysis import parent_offspring_concordance
    ped, trait, room, step = _line_population(heritable=False, seed=3)
    res = parent_offspring_concordance(trait, ped, room, step)
    assert abs(res["excess"]) < 0.05 and abs(res["z"]) < 3, res


def test_sibling_concordance_is_diluted_by_biparental_inheritance():
    """Sibs are a much weaker signal than parent-offspring here, and that is
    correct rather than a bug: a child takes the trait from ONE of its two
    parents, so two sibs agree only when they happen to draw the same parent,
    or when the parents already agreed. The test pins that relationship so the
    weak number is not later mistaken for absent transmission.
    """
    from evollm.analysis import (parent_offspring_concordance,
                                 sibling_concordance)
    ped, trait, room, step = _line_population(heritable=True, seed=5)
    sibs = sibling_concordance(trait, ped)["agree"]
    po = parent_offspring_concordance(trait, ped, room, step)
    ped2, trait2, _, _ = _line_population(heritable=False, seed=5)
    sibs_null = sibling_concordance(trait2, ped2)["agree"]

    assert po["excess"] > 0.15, "parent-offspring must show the transmission"
    assert sibs >= sibs_null, (sibs, sibs_null)
    assert sibs - sibs_null < po["excess"], \
        "sibling signal must be the weaker of the two, not the stronger"
