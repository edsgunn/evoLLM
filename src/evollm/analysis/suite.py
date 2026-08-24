"""The standard battery: run it on any run directory, get the same questions
answered the same way.

The point of fixing this as a suite rather than writing it fresh each time is
that the confound control is the hard part, and it should not be re-derived
(or re-forgotten) per analysis. Two controls are applied throughout:

  Stratified permutation. Lineages are not spread evenly over rooms or over
  time — a lineage that lived early, in a crowded room, differs from a late one
  for reasons that have nothing to do with inheritance. Every permutation null
  here shuffles WITHIN a (birth room x generation band) stratum, so those
  differences survive into the null and only the excess is counted.

  Genotype PCs as covariates. Related agents share genes with each other
  everywhere in the genome, so a site that merely tags a lineage will correlate
  with anything that lineage happens to do. Associations are fitted after
  removing the top principal components of the genotype matrix, which is what
  broad relatedness looks like in that matrix.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np

from .descent import Descent
from .genotypes import (align, build_genotypes, genotype_matrix,
                        load_fingerprints)
from .pedigree import Pedigree
from .phenotypes import ALL_TRAITS, TRAIT_GROUPS, build_phenotypes, strategy_matrix
from .stats import (associate, associate_alleles, kmeans,
                    mutual_information, principal_components,
                    replication, variance_partition)
from .table import Table

GEN_BANDS = [(0, 0), (1, 5), (6, 20), (21, 60), (61, 150),
             (151, float("inf"))]


def generation_band(g: float) -> str:
    for lo, hi in GEN_BANDS:
        if lo <= g <= hi:
            return f"g{lo}+" if hi == float("inf") else f"g{lo}-{hi}"
    return "g?"


def strata_of(pheno: Table) -> list[str]:
    """Birth room crossed with generation band: the environment an agent had."""
    return [f"{r}|{generation_band(g)}"
            for r, g in zip(pheno["room_born"], pheno["generation"])]


def effective_number(labels) -> float:
    """Inverse Simpson index: how many groups there EFFECTIVELY are.

    A population where one group holds 95% of everyone has ~1.1 effective
    groups no matter how many labels exist, and stratifying it by those labels
    tests nothing. This is the number to look at before believing any
    between-lineage result.
    """
    counts = np.array(list(Counter(labels).values()), dtype=float)
    p = counts / counts.sum()
    return float(1.0 / np.sum(p ** 2))


def analyse_run(run_dir: str | Path, traits=None, n_perm: int = 500,
                min_lineage: int = 20, k_range=(2, 3, 4, 5), n_pcs: int = 5,
                min_turns: int = 5, lineage_generation: int | None = None,
                seed: int = 0) -> dict:
    run_dir = Path(run_dir)
    traits = list(traits or ALL_TRAITS)
    ped = Pedigree.from_run(run_dir)
    pheno = build_phenotypes(run_dir, min_turns=min_turns, pedigree=ped)

    # A deeper cut than founder-level lineage, when asked for. See
    # Pedigree.ancestor_at for why this exists.
    if lineage_generation is not None:
        labels = [ped.ancestor_at(a, lineage_generation) or f"pre-g{lineage_generation}"
                  for a in pheno.index]
        pheno.add("lineage", labels)
    strata = strata_of(pheno)

    families = ped.families()
    fam_labels = [families.get(a, a) for a in pheno.index]
    lin_counts = Counter(pheno["lineage"])
    fam_counts = Counter(fam_labels)
    result = {
        "lineage_generation": lineage_generation,
        "structure": {
            "largest_lineage_share": (max(lin_counts.values()) / max(len(pheno), 1)),
            "largest_family_share": (max(fam_counts.values()) / max(len(pheno), 1)),
            "effective_lineages": effective_number(pheno["lineage"]),
            "effective_families": effective_number(fam_labels),
            "usable_lineages": sum(1 for v in lin_counts.values() if v >= min_lineage),
        },
        "run": run_dir.name,
        "n_agents": len(pheno),
        "n_lineages": len(set(pheno["lineage"])),
        "n_families": len(set(ped.families().values())),
        "mean_ancestry_entropy": float(np.mean(pheno["ancestry_entropy"])),
        "trait_summary": pheno.describe(traits),
    }

    # 1. Do lineages differ? ------------------------------------------------
    result["lineage_variance"] = []
    for trait in traits:
        vp = variance_partition(pheno[trait], pheno["lineage"], strata=strata,
                                n_perm=n_perm, min_group=min_lineage, seed=seed)
        vp["trait"] = trait
        result["lineage_variance"].append(vp)
    result["lineage_variance"].sort(
        key=lambda d: -(d["eta2"] - d["null_mean"] if np.isfinite(d["eta2"]) else -1))

    # 2. Are there distinct strategies, and are they heritable? -------------
    M, strategy_names = strategy_matrix(pheno)
    ok = np.isfinite(M).all(axis=1)
    result["strategy_names"] = strategy_names
    result["n_strategy_rows"] = int(ok.sum())
    result["clusters"] = []
    if ok.sum() >= 50:
        X = M[ok]
        lineages = np.asarray(list(pheno["lineage"]), dtype=object)[ok]
        strat_ok = np.asarray(strata, dtype=object)[ok]
        for k in k_range:
            labels, centres, inertia = kmeans(X, k, seed=seed)
            mi = mutual_information(lineages, labels, n_perm=n_perm,
                                    strata=strat_ok, seed=seed)
            sizes = np.bincount(labels, minlength=k)
            result["clusters"].append(dict(
                k=k, inertia=inertia, sizes=sizes.tolist(),
                centres=[dict(zip(strategy_names, c.round(4))) for c in centres],
                mi=mi["mi"], mi_p=mi.get("p"), mi_null=mi.get("null_mean")))

    # 3. Which sites track which trait? -------------------------------------
    fp_rows, _ = load_fingerprints(run_dir)
    result["genotype_source"] = "fingerprints" if fp_rows else "snapshots"
    geno, sites = build_genotypes(run_dir)
    result["sites"] = sites
    if len(geno):
        p_g, g_g, info = align(pheno, geno)
        result["genotype_coverage"] = info
        if len(p_g) >= 30:
            G = genotype_matrix(g_g, sites)
            pcs = principal_components(G, k=min(n_pcs, len(p_g) - 2))
            g_strata = strata_of(p_g)
            result["association"] = {}
            for trait in traits:
                hits = associate(G, p_g[trait], sites, covariates=pcs,
                                 strata=g_strata, n_perm=n_perm, seed=seed)
                # Every reported hit gets refit per birth room. A p-value alone
                # cannot tell a population-wide effect from a one-room one.
                for h in hits[:8]:
                    j = sites.index(h["site"])
                    h["replication"] = replication(G[:, j], p_g[trait],
                                                   p_g["room_born"])
                result["association"][trait] = hits[:8]
            result["genotype_pc_variance"] = _pc_variance(G, n_pcs)
    else:
        result["genotype_coverage"] = {"genotyped": 0}
    # 4. Realised descent: which founder actually supplied each site? -------
    desc = Descent.from_run(run_dir)
    if desc is not None:
        agents = list(pheno.index)
        M = desc.site_matrix(agents)
        resolved = (M >= 0).all(axis=1)
        eff = desc.effective_founders_per_site([a for a, r in zip(agents, resolved) if r])
        realised = [desc.realised_ancestry(a) for a, r in zip(agents, resolved) if r]
        drift = []
        for a, r in zip(agents, resolved):
            if not r:
                continue
            exp = ped.ancestry(a)
            got = desc.realised_ancestry(a)
            keys = set(exp) | set(got)
            drift.append(0.5 * sum(abs(exp.get(k, 0.0) - got.get(k, 0.0))
                                   for k in keys))
        result["descent"] = {
            "n_resolved": int(resolved.sum()),
            "n_sites": len(desc.sites),
            "n_founders": len(desc.founder_names()),
            "fixed_sites": int((eff <= 1).sum()) if len(eff) else 0,
            "median_founders_per_site": float(np.median(eff)) if len(eff) else 0.0,
            "min_founders_per_site": int(eff.min()) if len(eff) else 0,
            "median_founders_per_agent": float(np.median(
                [len(r) for r in realised])) if realised else 0.0,
            "median_realised_vs_expected": float(np.median(drift)) if drift else 0.0,
            "selection_scan": desc.selection_scan(
                [a for a, r in zip(agents, resolved) if r]),
        }
        if resolved.sum() >= 50:
            sub = pheno.filter(resolved)
            Msub = M[resolved]
            sub_strata = strata_of(sub)
            result["allele_association"] = {}
            for trait in traits:
                hits = associate_alleles(Msub, sub[trait], desc.sites,
                                         strata=sub_strata, n_perm=n_perm,
                                         seed=seed)
                result["allele_association"][trait] = hits[:5]

    result["_pheno"] = pheno
    return result


def _pc_variance(G: np.ndarray, k: int) -> list[float]:
    X = G - G.mean(0)
    sd = X.std(0)
    X = X / np.where(sd > 0, sd, 1.0)
    s = np.linalg.svd(X, compute_uv=False)
    var = s ** 2 / (s ** 2).sum()
    return [float(v) for v in var[:k]]


# ── rendering ─────────────────────────────────────────────────────────────
def _add_descent(L: list, r: dict) -> None:
    d = r.get("descent")
    if not d:
        return
    add = L.append
    add("")
    add("4. REALISED DESCENT — WHICH FOUNDER SUPPLIED EACH SITE?")
    add("-" * 78)
    add(f"   {d['n_resolved']:,} agents resolved across {d['n_sites']} sites "
        f"from {d['n_founders']} founders.")
    add(f"   Founder coalescence per site: median "
        f"{d['median_founders_per_site']:.0f}, minimum {d['min_founders_per_site']}"
        f" — {d['fixed_sites']} site(s) down to one founder.")
    add("   Founders are exchangeable N(0, init_scale) draws, so this measures")
    add("   coalescence of a neutral marker, NOT adaptive diversity: at")
    add("   generation 0 there is none to lose. Fixation is ambiguous on its")
    add("   own — drift, or selection fixing the luckiest draw.")
    scan = d.get("selection_scan") or []
    if scan:
        low = [x for x in scan[:3]]
        high = [x for x in scan[-3:]]
        add(f"   Most coalesced sites: " +
            ", ".join(f"{x['site']} ({x['founders']}, z={x['z']:+.1f})" for x in low))
        add(f"   Most retained sites:  " +
            ", ".join(f"{x['site']} ({x['founders']}, z={x['z']:+.1f})" for x in high))
        add("   All sites share one pedigree, so the spread across sites is its")
        add("   own neutral null; outliers are selection candidates (in blocks,")
        add("   not singly — linked sites travel together).")
    add(f"   Distinct founders contributing to one agent: median "
        f"{d['median_founders_per_agent']:.0f} of its {d['n_sites']} sites.")
    add(f"   Realised vs expected ancestry, median total-variation distance: "
        f"{d['median_realised_vs_expected']:.3f}")
    add("   (0 would mean descent exactly matched the one-half rule; the gap is")
    add("    drift, and it is only visible because inheritance is recorded.)")
    if "allele_association" in r:
        add("")
        add(f"   {'trait':24s} {'top site':14s} {'eta2':>7s} {'alleles':>8s} "
            f"{'p':>7s} {'p_fwer':>8s}")
        for trait, hits in r["allele_association"].items():
            if not hits:
                continue
            h = hits[0]
            star = " *" if h["p_fwer"] <= 0.05 else ""
            add(f"   {trait:24s} {h['site']:14s} {h['eta2']:7.4f} "
                f"{h['n_alleles']:8d} {h['p']:7.3f} {h['p_fwer']:8.3f}{star}")
        add("")
        add("   This asks whether CARRYING one founder's variant at a site")
        add("   predicts behaviour — a marker test on a categorical allele,")
        add("   not a correlation with perturbation magnitude.")


def format_report(r: dict) -> str:
    L = []
    add = L.append
    add(f"POPULATION ANALYSIS — {r['run']}")
    add("=" * 78)
    add(f"{r['n_agents']:,} agents with a measurable lifetime · "
        f"{r['n_lineages']} lineages · {r['n_families']} families · "
        f"mean ancestry entropy {r['mean_ancestry_entropy']:.2f} bits")
    add("")
    st = r["structure"]
    cut = ("founder (generation 0)" if r.get("lineage_generation") is None
           else f"dominant ancestor at generation {r['lineage_generation']}")
    add("POPULATION STRUCTURE")
    add("-" * 78)
    add(f"   lineage cut: {cut}")
    add(f"   largest lineage holds {st['largest_lineage_share'] * 100:5.1f}% of agents"
        f"   effective lineages {st['effective_lineages']:.2f}")
    add(f"   largest family  holds {st['largest_family_share'] * 100:5.1f}% of agents"
        f"   effective families {st['effective_families']:.2f}")
    add(f"   lineages with enough agents to test: {st['usable_lineages']}")
    if st["effective_lineages"] < 2.0:
        add("")
        add("   ** The population is effectively PANMICTIC at this cut: one group")
        add("      holds almost everyone, so there is nothing for a between-lineage")
        add("      test to compare. Any eta2 below is measured against 2-4 residual")
        add("      groups and is underpowered by construction. Re-cut deeper with")
        add("      --lineage-generation to recover structure, if there is any.")
    add("")
    add("TRAIT DISTRIBUTIONS")
    add("-" * 78)
    add(r["trait_summary"])
    add("")

    add("1. DOES LINEAGE EXPLAIN THE TRAIT?")
    add("-" * 78)
    add("   eta2 = share of variance between lineages. Permutation is within")
    add("   (birth room x generation band), so room and era are held fixed.")
    add("")
    add(f"   {'trait':24s} {'eta2':>7s} {'null':>7s} {'excess':>8s} "
        f"{'p':>7s} {'lineages':>9s} {'n':>7s}")
    for d in r["lineage_variance"]:
        if not np.isfinite(d.get("eta2", np.nan)):
            continue
        star = " *" if d["p"] <= 0.05 else ""
        add(f"   {d['trait']:24s} {d['eta2']:7.4f} {d['null_mean']:7.4f} "
            f"{d['eta2'] - d['null_mean']:+8.4f} {d['p']:7.3f} "
            f"{d['n_groups']:9d} {d['n']:7d}{star}")
    add("")

    if r.get("clusters"):
        add("2. ARE THERE DISTINCT STRATEGIES, AND ARE THEY HERITABLE?")
        add("-" * 78)
        add(f"   k-means on the action-composition simplex "
            f"({', '.join(r['strategy_names'])}), n={r['n_strategy_rows']:,}")
        add("   MI = bits of cluster membership predicted by lineage;")
        add("   its null is also permuted within room x generation band.")
        add("")
        for c in r["clusters"]:
            add(f"   k={c['k']}  inertia {c['inertia']:9.1f}  "
                f"MI {c['mi']:.3f} bits (null {c['mi_null']:.3f}, p={c['mi_p']:.3f})")
            for size, centre in zip(c["sizes"], c["centres"]):
                comp = "  ".join(f"{k.replace('_share','')} {v:.2f}"
                                 for k, v in centre.items())
                add(f"        n={size:6d}   {comp}")
        add("")

    cov = r.get("genotype_coverage", {})
    add("3. WHICH SITES TRACK WHICH TRAIT?")
    add("-" * 78)
    if not cov.get("genotyped"):
        add("   No snapshot genomes in this run — association skipped.")
        return "\n".join(L)
    src = r.get("genotype_source", "snapshots")
    add(f"   source: {src}. {cov['matched']} of {cov['genotyped']} genomes matched")
    add(f"   to a phenotype ({cov['genotyped_without_phenotype']} had none: still")
    add("   alive, or died under the turn floor).")
    if src == "snapshots":
        add("   Snapshots sample the LIVING, so this subset is skewed toward the")
        add("   long-lived and is not a random draw from the run. Runs with")
        add("   run.genome_fingerprints on cover every agent instead.")
    if "genotype_pc_variance" in r:
        pcs = ", ".join(f"{v * 100:.1f}%" for v in r["genotype_pc_variance"])
        add(f"   Genotype PC variance explained: {pcs} — removed as covariates.")
    add("")
    if "association" not in r:
        add("   Too few matched agents to fit associations.")
        return "\n".join(L)
    add(f"   {'trait':24s} {'top site':14s} {'r':>7s} {'p_fwer':>8s} "
        f"{'q':>7s} {'agree':>6s} {'per-room r':>28s}")
    for trait, hits in r["association"].items():
        if not hits:
            continue
        h = hits[0]
        rep = h.get("replication", {})
        per = "  ".join(f"{v['r']:+.2f}" for v in rep.get("groups", {}).values())
        agree = rep.get("consistent", float("nan"))
        # A star needs the effect to be PRESENT in more than one room, not
        # merely to point the same way: sign agreement alone passes a hit whose
        # per-room fits are +0.03 / +0.55 / +0.08, which is one room and two
        # near-zeros. Require the weakest room to reach half the pooled effect.
        pooled = abs(rep.get("pooled") or 0.0)
        weakest = rep.get("min_abs", float("nan"))
        flag = ""
        if h["p_fwer"] <= 0.05:
            replicates = (agree >= 0.75 and np.isfinite(weakest)
                          and weakest >= 0.5 * pooled)
            flag = " *" if replicates else " (one room)"
        add(f"   {trait:24s} {h['site']:14s} {h['r']:+7.3f} {h['p_fwer']:8.3f} "
            f"{h['q']:7.3f} {agree:6.2f} {per:>28s}{flag}")
    add("")
    _add_descent(L, r)
    add("   p_fwer is the null of the LARGEST |t| over all 112 sites, so it")
    add("   already pays for having looked at all of them. `agree` is the share")
    add("   of birth rooms whose own fit has the same sign as the pooled one,")
    add("   and per-room r shows the fits themselves. A starred row needs BOTH:")
    add("   an association that survives the multiple-testing null AND one that")
    add("   is present in more than a single room — sign agreement AND a")
    add("   weakest-room effect of at least half the pooled one.")
    return "\n".join(L)
