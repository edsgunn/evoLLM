"""Statistics for structured populations.

Everything here uses permutation for significance rather than a parametric
tail. That is not only to avoid a scipy dependency — these traits are bounded
rates, shares on a simplex, and zero-inflated counts, none of which are normal,
and the agents are related to each other, which breaks the independence every
closed-form test assumes.

The related-agents problem is the one that quietly ruins this kind of analysis.
Two agents from the same lineage share genes AND share a room AND were alive at
the same time, so ANY trait correlates with ANY gene if you let relatedness
supply the correlation. Two defences are provided and both should normally be
used:

  covariates   Regress the trait on genotype principal components first, so the
               association is fitted to what is left after broad population
               structure is removed.
  strata       Permute labels only WITHIN a stratum (lineage, room, time bin),
               so the null preserves the structure instead of destroying it.
               A p-value from unrestricted permutation against structured data
               is anticonservative, often wildly.
"""

from __future__ import annotations

import numpy as np


# ── basic fitting ─────────────────────────────────────────────────────────
def ols(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Least squares with an intercept. Returns (coefficients, residuals)."""
    X = np.column_stack([np.ones(len(X)), X]) if X.ndim > 1 else \
        np.column_stack([np.ones(len(X)), X.reshape(-1, 1)])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta, y - X @ beta


def residualise(y: np.ndarray, covariates: np.ndarray | None) -> np.ndarray:
    """The part of y not explained by the covariates."""
    if covariates is None or covariates.size == 0:
        return y - np.nanmean(y)
    _, resid = ols(covariates, y)
    return resid


def principal_components(M: np.ndarray, k: int = 5) -> np.ndarray:
    """Top-k PCs of a column-standardised matrix; the structure covariates."""
    X = M - M.mean(0)
    sd = X.std(0)
    X = X / np.where(sd > 0, sd, 1.0)
    _, _, vt = np.linalg.svd(X, full_matrices=False)
    return X @ vt[:k].T


# ── variance partition: do groups differ at all? ──────────────────────────
def variance_partition(values: np.ndarray, groups, strata=None,
                       n_perm: int = 1000, min_group: int = 5,
                       seed: int = 0) -> dict:
    """Share of a trait's variance lying BETWEEN groups rather than within.

    This is the "do lineages have different strategies" statistic. It is eta²
    (equivalently an F-test's effect size, and the same quantity population
    genetics calls F_ST when the trait is an allele frequency).

    eta² is biased upward by the number of groups — with one agent per group it
    is 1 by construction — so the permutation null is not optional here. It is
    what turns "groups differ by 4%" into "groups differ by 4% where reshuffled
    labels give 1.2%".
    """
    values = np.asarray(values, float)
    labels = np.asarray(list(groups), dtype=object)
    ok = np.isfinite(values)
    values, labels = values[ok], labels[ok]
    if strata is not None:
        strata = np.asarray(list(strata), dtype=object)[ok]

    keep_labels = {g for g in set(labels) if (labels == g).sum() >= min_group}
    # dtype is explicit: an empty list would otherwise build a float
    # array and fail as an index when no group clears min_group.
    keep = np.array([g in keep_labels for g in labels], dtype=bool)
    values, labels = values[keep], labels[keep]
    if strata is not None:
        strata = strata[keep]
    if len(values) < 3 or len(keep_labels) < 2:
        return dict(eta2=np.nan, p=np.nan, n=len(values),
                    n_groups=len(keep_labels), null_mean=np.nan)

    # Recode once, then score every permutation in a single matmul. The
    # per-permutation Python loop over group masks was the slowest thing in the
    # battery: 100 lineages x 500 permutations x 17 traits of boolean indexing
    # over 17,000 agents.
    codes = np.unique(labels, return_inverse=True)[1][:, None]
    obs = _eta2_matrix(codes, values[:, None])[0, 0]
    rng = np.random.default_rng(seed)
    order = np.arange(len(values))
    perms = np.empty((len(values), n_perm))
    if strata is None:
        for i in range(n_perm):
            perms[:, i] = values[rng.permutation(order)]
    else:
        # Permuting the VALUES within strata is equivalent to permuting the
        # labels and far cheaper, since the group coding can then stay fixed.
        idx = np.arange(len(values))
        blocks = [np.where(strata == sv)[0] for sv in set(strata)]
        for i in range(n_perm):
            take = idx.copy()
            for blk in blocks:
                take[blk] = rng.permutation(blk)
            perms[:, i] = values[take]
    null = _eta2_matrix(codes, perms)[0]
    return dict(eta2=float(obs),
                p=float((np.sum(null >= obs) + 1) / (n_perm + 1)),
                null_mean=float(null.mean()), null_sd=float(null.std()),
                n=int(len(values)), n_groups=len(keep_labels))


def _eta2(values: np.ndarray, labels: np.ndarray) -> float:
    grand = values.mean()
    total = ((values - grand) ** 2).sum()
    if total <= 0:
        return 0.0
    between = 0.0
    for g in set(labels):
        v = values[labels == g]
        between += len(v) * (v.mean() - grand) ** 2
    return between / total


def _shuffle(labels: np.ndarray, strata, rng) -> np.ndarray:
    """Permute labels, within strata when given."""
    if strata is None:
        return rng.permutation(labels)
    out = labels.copy()
    for s in set(strata):
        idx = np.where(strata == s)[0]
        out[idx] = rng.permutation(labels[idx])
    return out


# ── association: which sites track a trait? ───────────────────────────────
def associate(G: np.ndarray, y: np.ndarray, sites: list[str],
              covariates: np.ndarray | None = None, strata=None,
              n_perm: int = 1000, seed: int = 0) -> list[dict]:
    """Per-site association between a genotype column and a trait.

    Both trait and genotype are residualised on the covariates, then each site
    is scored by the correlation of the residuals. Two p-values come back and
    they answer different questions:

      p        per-site, from permuting the trait. Use it to rank sites.
      p_fwer   from the null distribution of the LARGEST |t| across all sites
               at once. This is the one that says whether the top hit is real,
               and with 112 sites it is much more conservative than p.

    Reported sorted by |t| descending.
    """
    G = np.asarray(G, float)
    y = np.asarray(y, float)
    ok = np.isfinite(y) & np.isfinite(G).all(axis=1)
    G, y = G[ok], y[ok]
    cov = None if covariates is None else np.asarray(covariates, float)[ok]
    st = None if strata is None else np.asarray(list(strata), dtype=object)[ok]
    n = len(y)
    if n < 10:
        return []

    Gr = np.column_stack([residualise(G[:, j], cov) for j in range(G.shape[1])])
    yr = residualise(y, cov)
    t_obs = _tstats(Gr, yr)

    rng = np.random.default_rng(seed)
    null_max = np.empty(n_perm)
    exceed = np.zeros(len(t_obs))
    order = np.arange(n)
    for i in range(n_perm):
        perm = _shuffle(order.astype(object), st, rng).astype(int) \
            if st is not None else rng.permutation(order)
        t_null = _tstats(Gr, yr[perm])
        null_max[i] = np.max(np.abs(t_null))
        exceed += np.abs(t_null) >= np.abs(t_obs)

    r = _corr(Gr, yr)
    out = []
    for j, site in enumerate(sites):
        out.append(dict(site=site, r=float(r[j]), t=float(t_obs[j]),
                        p=float((exceed[j] + 1) / (n_perm + 1)),
                        p_fwer=float((np.sum(null_max >= abs(t_obs[j])) + 1)
                                     / (n_perm + 1)),
                        n=int(n)))
    out.sort(key=lambda d: -abs(d["t"]))
    for d, q in zip(out, benjamini_hochberg([d["p"] for d in out])):
        d["q"] = q
    return out


def _corr(G: np.ndarray, y: np.ndarray) -> np.ndarray:
    gs = G.std(0)
    ys = y.std()
    if ys == 0:
        return np.zeros(G.shape[1])
    cov = ((G - G.mean(0)) * (y - y.mean())[:, None]).mean(0)
    return cov / np.where(gs > 0, gs, np.inf) / ys


def _tstats(G: np.ndarray, y: np.ndarray) -> np.ndarray:
    r = _corr(G, y)
    n = len(y)
    denom = np.sqrt(np.maximum(1 - r ** 2, 1e-12) / max(n - 2, 1))
    return r / denom


def benjamini_hochberg(pvals) -> np.ndarray:
    """FDR-adjusted q-values, order preserved."""
    p = np.asarray(list(pvals), float)
    n = len(p)
    if n == 0:
        return p
    order = np.argsort(p)
    q = np.empty(n)
    prev = 1.0
    for rank, idx in enumerate(reversed(order), start=1):
        val = min(prev, p[idx] * n / (n - rank + 1))
        q[idx] = prev = val
    return np.clip(q, 0, 1)


def _eta2_matrix(codes: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """(n sites, P) of eta2 — one row per site, one column per permutation.

    Every permutation is scored in one matmul per site. The naive form (a
    bincount per site per permutation) is 112 x P Python-level calls per trait,
    which on 24,000 agents with 500 permutations turned one run of the battery
    into hours.

    The one-hot for each site is built INSIDE the loop and discarded. Holding
    all 112 at once is 128 founders x 17,000 agents x 8 bytes x 112 sites,
    close to 2 GB, and it was quietly getting the process OOM-killed on a
    login node with no traceback to show for it.
    """
    n, P = Y.shape
    grand = Y.mean(0)
    total = ((Y - grand) ** 2).sum(0)
    total = np.where(total > 0, total, np.inf)
    rows = np.arange(n)
    out = np.empty((codes.shape[1], P))
    for j in range(codes.shape[1]):
        col = codes[:, j]
        k = int(col.max()) + 1
        onehot = np.zeros((k, n))
        onehot[col, rows] = 1.0
        counts = onehot.sum(1)
        sums = onehot @ Y                       # (k, P)
        nz = counts > 0
        means = np.zeros_like(sums)
        means[nz] = sums[nz] / counts[nz, None]
        between = counts[:, None] * (means - grand[None, :]) ** 2
        out[j] = np.where(nz[:, None], between, 0.0).sum(0) / total
    return out


def _eta2_by_column(codes: np.ndarray, y: np.ndarray) -> np.ndarray:
    """eta2 of y against each column of integer group codes."""
    return _eta2_matrix(codes, y[:, None])[:, 0]


def associate_alleles(site_founders: np.ndarray, y: np.ndarray,
                      sites: list[str], strata=None, n_perm: int = 500,
                      min_allele: int = 10, seed: int = 0) -> list[dict]:
    """Per-site association between WHICH FOUNDER supplied a site and a trait.

    This is a genuine marker test. `site_founders` is the realised descent
    matrix from `Descent.site_matrix`: entry (i, j) names the founder that
    actually supplied site j of agent i. Grouping a trait by that label asks
    the question genetics actually asks — do carriers of one ancestral variant
    at this locus behave differently — rather than the weaker question of
    whether a site's perturbation MAGNITUDE correlates with behaviour.

    A site whose alleles have all coalesced to one founder is skipped: it is
    fixed, and there is nothing left to compare.

    `p_fwer` comes from the null of the largest eta2 across all sites at once,
    so it already pays for scanning the genome.
    """
    y = np.asarray(y, float)
    M = np.asarray(site_founders)
    ok = np.isfinite(y) & (M >= 0).all(axis=1)
    y, M = y[ok], M[ok]
    st = None if strata is None else np.asarray(list(strata), dtype=object)[ok]
    if len(y) < 20 or M.size == 0:
        return []

    # Recode each column densely, and drop sites with no usable variation.
    cols, keep_sites = [], []
    for j, site in enumerate(sites):
        vals, inv = np.unique(M[:, j], return_inverse=True)
        counts = np.bincount(inv)
        if (counts >= min_allele).sum() < 2:
            continue
        cols.append(inv.astype(np.int64))
        keep_sites.append(site)
    if not cols:
        return []
    codes = np.column_stack(cols)

    obs = _eta2_matrix(codes, y[:, None])[:, 0]
    rng = np.random.default_rng(seed)
    order = np.arange(len(y))
    # Materialise every permuted trait vector, then score them in one pass.
    perms = np.empty((len(y), n_perm))
    for i in range(n_perm):
        idx = (_shuffle(order.astype(object), st, rng).astype(int)
               if st is not None else rng.permutation(order))
        perms[:, i] = y[idx]
    null = _eta2_matrix(codes, perms)           # (sites, n_perm)
    null_max = null.max(0)
    exceed = (null >= obs[:, None]).sum(1)
    out = [dict(site=site, eta2=float(obs[j]),
                n_alleles=int(codes[:, j].max() + 1),
                p=float((exceed[j] + 1) / (n_perm + 1)),
                p_fwer=float((np.sum(null_max >= obs[j]) + 1) / (n_perm + 1)),
                n=int(len(y)))
           for j, site in enumerate(keep_sites)]
    out.sort(key=lambda d: -d["eta2"])
    for d, q in zip(out, benjamini_hochberg([d["p"] for d in out])):
        d["q"] = q
    return out


def replication(g: np.ndarray, y: np.ndarray, groups, min_n: int = 25) -> dict:
    """Refit one site's association separately in each group.

    A permutation p-value says the association is unlikely under a shuffled
    null. It does NOT say the association exists in more than one place, and in
    a structured population a single room, cohort or clade can carry the whole
    thing. Splitting by group is the cheapest way to find that out, and in
    practice it is what kills most top hits: a site can reach p_fwer = 0.004
    overall while showing r = +0.55 in one room and r = +0.03 in the others.

    Returns per-group correlations plus `consistent`, the share of groups
    agreeing in sign with the pooled estimate. Below ~0.75 the pooled estimate
    is describing one group, not a population.
    """
    g = np.asarray(g, float)
    y = np.asarray(y, float)
    labels = np.asarray(list(groups), dtype=object)
    ok = np.isfinite(g) & np.isfinite(y)
    g, y, labels = g[ok], y[ok], labels[ok]
    pooled = float(np.corrcoef(g, y)[0, 1]) if len(g) > 2 and g.std() > 0 else np.nan

    per = {}
    for grp in sorted(set(labels), key=str):
        m = labels == grp
        if m.sum() < min_n or g[m].std() == 0 or y[m].std() == 0:
            continue
        per[str(grp)] = dict(n=int(m.sum()),
                             r=float(np.corrcoef(g[m], y[m])[0, 1]))
    if not per:
        return dict(pooled=pooled, groups={}, consistent=np.nan, min_abs=np.nan)
    signs = [np.sign(v["r"]) == np.sign(pooled) for v in per.values()]
    return dict(pooled=pooled, groups=per,
                consistent=float(np.mean(signs)),
                min_abs=float(min(abs(v["r"]) for v in per.values())),
                n_groups=len(per))


def parent_offspring_concordance(trait: dict, pedigree, room: dict, step: dict,
                                 window: int = 2000, seed: int = 0) -> dict:
    """Do children share a categorical trait with their PARENTS, above what a
    same-room, same-era stranger shares with them?

    This is the right test for whether a discrete behaviour is inherited, and
    it exists because the obvious alternative is badly wrong at depth. Labelling
    an agent by its generation-0 founder and asking whether that predicts
    behaviour is asking about an ancestor two hundred generations back: in a
    population where everyone descends from the same handful of founders, the
    label is shared by almost everyone and carries no information. A test built
    on it returns "not heritable" no matter how strongly parents resemble their
    children — the same way a human and a barnacle share an ancestor without
    sharing a niche.

    The control is a stranger drawn from the same room within `window` steps,
    so room composition and era are held fixed and only the parental link is
    left.
    """
    rng = np.random.default_rng(seed)
    by_room: dict = {}
    for a in trait:
        by_room.setdefault(room[a], []).append((step[a], a))
    for v in by_room.values():
        v.sort()

    pairs, ctrl = [], []
    for a in trait:
        parents = [p for p in (pedigree.parents.get(a) or ()) if p in trait]
        if not parents:
            continue
        pool = [q for s, q in by_room[room[a]]
                if abs(s - step[a]) < window and q != a]
        for p in parents:
            pairs.append(trait[p] == trait[a])
            if pool:
                ctrl.append(trait[pool[rng.integers(len(pool))]] == trait[a])
    if len(pairs) < 50 or len(ctrl) < 50:
        return {"n": len(pairs)}
    pa, ca = np.array(pairs, float), np.array(ctrl, float)
    agree, control = pa.mean(), ca.mean()
    se = np.sqrt(agree * (1 - agree) / len(pa) + control * (1 - control) / len(ca))
    return dict(agree=float(agree), control=float(control),
                excess=float(agree - control),
                z=float((agree - control) / se) if se > 0 else float("nan"),
                n=len(pa), n_control=len(ca))


def sibling_concordance(trait: dict, pedigree) -> dict:
    """How often full siblings share a categorical trait. Sibs share both
    parents, so this bounds what parent-offspring transmission can deliver."""
    sibs: dict = {}
    for a in trait:
        par = pedigree.parents.get(a)
        if par and len(par) == 2:
            sibs.setdefault(tuple(sorted(par)), []).append(a)
    same = tot = 0
    for group in sibs.values():
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                tot += 1
                same += trait[group[i]] == trait[group[j]]
    if tot < 50:
        return {"n": tot}
    return dict(agree=same / tot, n=tot)


# ── niches: are there distinct strategies at all? ─────────────────────────
def kmeans(X: np.ndarray, k: int, seed: int = 0, iters: int = 100):
    """Plain k-means with k-means++ seeding. Returns (labels, centres, inertia)."""
    rng = np.random.default_rng(seed)
    X = np.asarray(X, float)
    centres = [X[rng.integers(len(X))]]
    for _ in range(k - 1):
        d = np.min(np.stack([((X - c) ** 2).sum(1) for c in centres]), axis=0)
        total = d.sum()
        probs = d / total if total > 0 else np.full(len(X), 1 / len(X))
        centres.append(X[rng.choice(len(X), p=probs)])
    C = np.array(centres)
    labels = np.zeros(len(X), dtype=int)
    for _ in range(iters):
        dist = ((X[:, None, :] - C[None, :, :]) ** 2).sum(-1)
        new = dist.argmin(1)
        if np.array_equal(new, labels):
            break
        labels = new
        for j in range(k):
            if (labels == j).any():
                C[j] = X[labels == j].mean(0)
    inertia = float(((X - C[labels]) ** 2).sum())
    return labels, C, inertia


def mutual_information(a, b, n_perm: int = 0, strata=None, seed: int = 0) -> dict:
    """MI between two labellings, in bits, with an optional permutation null.

    Used to ask whether lineage predicts which behavioural cluster an agent
    lands in — i.e. whether niches are heritable rather than just present.
    """
    # Recode once to integers; the permutation loop then works on plain ints.
    a = np.unique(np.asarray(list(a), dtype=object), return_inverse=True)[1]
    b = np.unique(np.asarray(list(b), dtype=object), return_inverse=True)[1]
    obs = _mi(a, b)
    res = dict(mi=float(obs), n=int(len(a)))
    if n_perm:
        rng = np.random.default_rng(seed)
        st = None if strata is None else np.asarray(list(strata), dtype=object)
        null = np.array([_mi(a, _shuffle(b, st, rng)) for _ in range(n_perm)])
        res.update(p=float((np.sum(null >= obs) + 1) / (n_perm + 1)),
                   null_mean=float(null.mean()))
    return res


def _mi(a: np.ndarray, b: np.ndarray) -> float:
    """Mutual information in bits.

    The joint table is built with bincount rather than a Python loop over
    pairs: at 17,000 agents and 500 permutations the loop version was 8.5M
    iterations per k, and it dominated the whole battery.
    """
    n = len(a)
    if n == 0:
        return 0.0
    ca = a if a.dtype.kind in "iu" else np.unique(a, return_inverse=True)[1]
    cb = b if b.dtype.kind in "iu" else np.unique(b, return_inverse=True)[1]
    ka, kb = int(ca.max()) + 1, int(cb.max()) + 1
    joint = np.bincount(ca * kb + cb, minlength=ka * kb).reshape(ka, kb)
    joint = joint / n
    pa = joint.sum(1, keepdims=True)
    pb = joint.sum(0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        term = joint * np.log2(joint / (pa * pb))
    return float(np.nansum(np.where(joint > 0, term, 0.0)))
