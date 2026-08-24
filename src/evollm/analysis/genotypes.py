"""Snapshot genomes as a numeric matrix, one row per agent, one column per site.

A genome is a LoRA adapter: per (layer, projection) site, factors A (rank x in)
and B (out x rank). What the model actually feels at a site is the product
ΔW = B @ A, so the default per-site feature is ‖B @ A‖_F — the magnitude of the
perturbation that site applies.

That norm is computed WITHOUT forming B @ A. For rank r,

    ‖BA‖_F² = tr(AᵀBᵀBA) = tr((A Aᵀ)(Bᵀ B))

and both AAᵀ and BᵀB are r x r. With r=16 against a 3584-wide model that is the
difference between a 16x16 trace and a 3584x3584 matmul, 112 times per agent.

Only agents alive at a snapshot step have genomes, so genotype coverage is
always a subset — and a biased one, since it excludes anything that died
between snapshots. `align` reports what it dropped rather than silently
inner-joining.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from .table import Table

FEATURES = ("delta_norm", "rms_a", "rms_b")


def _site_key(name: str) -> str | None:
    m = re.fullmatch(r"(\d+\.\w+)\.[AB]", name)
    return m.group(1) if m else None


def load_genome_features(path: str | Path,
                         feature: str = "delta_norm") -> dict[str, float]:
    """One scalar per site for a single agent's `.safetensors` genome."""
    from safetensors.numpy import load_file
    tensors = load_file(str(path))
    sites: dict[str, dict[str, np.ndarray]] = {}
    for name, arr in tensors.items():
        key = _site_key(name)
        if key is not None:
            sites.setdefault(key, {})[name[-1]] = arr
    out = {}
    for key, ab in sites.items():
        a, b = ab.get("A"), ab.get("B")
        if a is None or b is None:
            continue
        if feature == "delta_norm":
            a = a.astype(np.float64); b = b.astype(np.float64)
            val = float(np.sqrt(max(np.trace((a @ a.T) @ (b.T @ b)), 0.0)))
        elif feature == "rms_a":
            val = float(np.sqrt(np.mean(a.astype(np.float64) ** 2)))
        elif feature == "rms_b":
            val = float(np.sqrt(np.mean(b.astype(np.float64) ** 2)))
        else:
            raise ValueError(f"unknown feature {feature!r}; expected one of {FEATURES}")
        out[key] = val
    return out


def snapshot_paths(run_dir: str | Path) -> dict[str, tuple[Path, int]]:
    """agent -> (genome path, snapshot step). Latest snapshot wins."""
    found: dict[str, tuple[Path, int]] = {}
    for p in sorted(Path(run_dir).glob("snapshots/*/step_*/*.safetensors")):
        step = int(p.parent.name.split("_")[-1])
        agent = p.stem
        if agent not in found or step > found[agent][1]:
            found[agent] = (p, step)
    return found


def load_fingerprints(run_dir: str | Path, feature: str = "delta_norm"
                      ) -> tuple[dict[str, dict], list[str]]:
    """Read `<run>/genomes/*.jsonl`, written once per agent at creation.

    Preferred over snapshots wherever present: it covers every agent rather
    than only those alive when a snapshot fired, which removes the survivor
    bias that makes snapshot-based association both underpowered and skewed
    toward the long-lived.
    """
    rows: dict[str, dict] = {}
    sites: list[str] = []
    for path in sorted(Path(run_dir).glob("genomes/*.jsonl")):
        with path.open() as fh:
            for line in fh:
                rec = json.loads(line)
                keys = rec.get("sites") or []
                values = rec.get(feature)
                if not keys or values is None:
                    continue
                if not sites:
                    sites = list(keys)
                rows[rec["agent"]] = {**dict(zip(keys, values)),
                                      "snapshot_step": rec.get("step", 0)}
    return rows, sites


def build_genotypes(run_dir: str | Path, feature: str = "delta_norm",
                    agents=None, source: str = "auto"
                    ) -> tuple[Table, list[str]]:
    """Per-site feature table. Returns (table, site keys in genome order).

    `source`: "fingerprints", "snapshots", or "auto" (fingerprints when the
    run has them, snapshots otherwise). Runs made before fingerprints existed
    fall back automatically.
    """
    if source in ("auto", "fingerprints"):
        rows, sites = load_fingerprints(run_dir, feature)
        if rows:
            if agents is not None:
                keep = set(agents)
                rows = {a: v for a, v in rows.items() if a in keep}
            return Table.from_records(rows), sites
        if source == "fingerprints":
            return Table.from_records({}), []
    paths = snapshot_paths(run_dir)
    if agents is not None:
        keep = set(agents)
        paths = {a: v for a, v in paths.items() if a in keep}
    rows: dict[str, dict] = {}
    sites: list[str] = []
    for agent, (path, step) in sorted(paths.items()):
        feats = load_genome_features(path, feature)
        if not sites:
            sites = list(feats)
        rows[agent] = {**feats, "snapshot_step": step}
    return Table.from_records(rows), sites


def genotype_matrix(geno: Table, sites: list[str]) -> np.ndarray:
    """(n agents, n sites) float matrix in `sites` order."""
    return np.column_stack([geno[s].astype(float) for s in sites])


def align(pheno: Table, geno: Table) -> tuple[Table, Table, dict]:
    """Restrict both tables to agents present in each, reporting the loss.

    Genotype coverage is a snapshot of the living, so the intersection is
    always smaller than either side and is not a random sample of the run.
    """
    shared = [a for a in geno.index if a in set(pheno.index)]
    info = {"phenotyped": len(pheno), "genotyped": len(geno),
            "matched": len(shared),
            "genotyped_without_phenotype": len(geno) - len(shared)}
    return pheno.select(shared), geno.select(shared), info
