"""Genome: a LoRA adapter as heritable material (§3).

Representation: for each adapted (layer, projection) site, a factor pair
(A, B) with A: (rank, in_dim), B: (out_dim, rank). Rank is fixed and uniform
across the population so adapter footprint is identical for every agent and
block accounting stays per-agent-uniform (§3.1).

Reproduction (§3.2): the child inherits (A, B) wholesale from one parent per
site, followed by mutation. Wholesale inheritance sidesteps
basis-arbitrariness: the child's ΔW at each site is exactly one parent's ΔW
at that site. Two axes are configurable, both because the uniform/additive
default measurably degrades children across generations:

  crossover  "uniform" (a coin per site) or "chromosomal" (one crossover
             point per contiguous chromosome of the layer-major site list).
  mutation   "additive" (x + N(0, std), which random-walks magnitude upward)
             or "multiplicative" (x * (1 + N(0, std)), which does not).

Tensors are numpy float32 in the world layer; the vLLM backend converts to
fp16 safetensors in peft layout at registration time.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Site:
    layer: int
    projection: str   # e.g. "q_proj"
    in_dim: int
    out_dim: int

    @property
    def key(self) -> str:
        return f"{self.layer}.{self.projection}"


@dataclass(frozen=True)
class GenomeSpec:
    sites: tuple[Site, ...]
    rank: int
    alpha: int

    def adapter_bytes(self, bytes_per_param: int = 2) -> int:
        """Serialized adapter footprint (fp16 by default) for block accounting."""
        params = sum(self.rank * (s.in_dim + s.out_dim) for s in self.sites)
        return params * bytes_per_param


@dataclass
class Genome:
    spec: GenomeSpec
    # site.key -> (A, B); arrays are owned by this genome and never aliased.
    factors: dict[str, tuple[np.ndarray, np.ndarray]]

    @classmethod
    def random(cls, spec: GenomeSpec, scale: float, rng: np.random.Generator) -> "Genome":
        """Gen-0 initialisation. Both factors are drawn non-zero so ΔW ≠ 0:
        with the conventional B=0 init every gen-0 agent would be functionally
        the base model and crossover would have no variation to work with."""
        factors = {}
        for s in spec.sites:
            a = rng.normal(0.0, scale, size=(spec.rank, s.in_dim)).astype(np.float32)
            b = rng.normal(0.0, scale, size=(s.out_dim, spec.rank)).astype(np.float32)
            factors[s.key] = (a, b)
        return cls(spec, factors)

    @classmethod
    def zeros(cls, spec: GenomeSpec) -> "Genome":
        """Functionally the frozen base model (B = 0). Used for base-rate
        prechecks (§6) and unevolved controls (§5)."""
        factors = {}
        for s in spec.sites:
            a = np.zeros((spec.rank, s.in_dim), dtype=np.float32)
            b = np.zeros((s.out_dim, spec.rank), dtype=np.float32)
            factors[s.key] = (a, b)
        return cls(spec, factors)

    @classmethod
    def crossover(cls, p1: "Genome", p2: "Genome", mutation_std: float,
                  rng: np.random.Generator, scheme: str = "chromosomal",
                  chromosomes: int = 3, mutation: str = "additive",
                  return_donors: bool = False):
        """Recombine two parents. Whole (A, B) pairs travel together per site
        under either scheme, so no factor is ever blended across parents and
        §3.2's basis-arbitrariness argument is untouched. What the scheme
        changes is which sites travel *together*.

        "uniform": every site flips its own coin. With 112 sites that is ~55
            parent switches per child, so a set of k co-adapted sites survives
            intact with probability 2^-(k-1) — co-adaptation cannot persist.
            Measured consequence: children reproduce worse than fresh random
            genomes, which is the signature of recombination outrunning the
            linkage structure.

        "chromosomal": sites are already ordered layer-major as
            [q, k, v, o, q, k, v, o, ...], which puts q beside k (they meet in
            QK^T) and v beside o (o_proj consumes the v-weighted heads), with
            layers in residual-stream order. That list is cut into contiguous
            chromosomes and each takes ONE crossover point, so blocks of
            interacting sites are inherited whole. `chromosomes` is the linkage
            dial: 1 is a single cut over the whole genome, len(sites)
            reproduces "uniform" exactly.

        With `return_donors`, also returns the per-site inheritance mask —
        True where the site came from p1. That mask IS the act of inheritance:
        one bit per site per child, and the only record of which parent
        actually supplied each part of the genome. Without it, descent can
        only be estimated by the expected one-half rule, which is an average
        over births that never happened rather than the one that did.
        """
        assert p1.spec == p2.spec
        sites = p1.spec.sites
        if scheme == "chromosomal":
            donors = cls._chromosomal_donors(len(sites), chromosomes, rng)
        elif scheme == "uniform":
            donors = [rng.random() < 0.5 for _ in sites]
        else:
            raise ValueError(f"unknown crossover scheme {scheme!r}")

        factors = {}
        for site, from_p1 in zip(sites, donors):
            donor = p1 if from_p1 else p2
            a, b = donor.factors[site.key]
            factors[site.key] = (cls._mutate(a, mutation_std, mutation, rng),
                                 cls._mutate(b, mutation_std, mutation, rng))
        child = cls(p1.spec, factors)
        return (child, donors) if return_donors else child

    @staticmethod
    def _chromosomal_donors(n: int, chromosomes: int,
                            rng: np.random.Generator) -> list[bool]:
        """One crossover point per chromosome, and a coin for which parent
        leads it — otherwise every child would be biased toward parent one."""
        chromosomes = max(1, min(chromosomes, n))
        edges = [round(i * n / chromosomes) for i in range(chromosomes + 1)]
        donors: list[bool] = []
        for lo, hi in zip(edges, edges[1:]):
            if hi <= lo:
                continue
            point = int(rng.integers(lo, hi + 1))   # may sit at either end
            lead = bool(rng.random() < 0.5)
            donors.extend(lead if i < point else not lead for i in range(lo, hi))
        return donors

    @staticmethod
    def _mutate(x: np.ndarray, std: float, mode: str,
                rng: np.random.Generator) -> np.ndarray:
        """Additive noise is what every archived run used, and it random-walks:
        factor RMS grew 0.0200 -> 0.0267 by generation 5 and the perturbation
        the adapter applies, ||B@A||, nearly doubled — a growing push in a
        random direction that no recombination scheme can undo. Multiplicative
        noise scales with each factor's own magnitude, so the *relative*
        perturbation per mating stays constant instead of compounding.
        """
        if mode == "multiplicative":
            return (x * (1.0 + rng.normal(0.0, std, size=x.shape))).astype(np.float32)
        if mode != "additive":
            raise ValueError(f"unknown mutation mode {mode!r}")
        return x + rng.normal(0.0, std, size=x.shape).astype(np.float32)

    @staticmethod
    def donor_mask_to_str(donors) -> str:
        """One character per site: '1' from parent one, '0' from parent two.

        Stored as plain text rather than packed bits. 112 characters per child
        is ~1.3 MB across a 12,000-birth run — small enough that being able to
        read, grep and diff the masks directly is worth more than a 4x saving.
        Positions correspond to `spec.sites` order, which is layer-major
        [q, k, v, o] and is what the chromosomal scheme cuts.
        """
        return "".join("1" if d else "0" for d in donors)

    # ── cheap summary ─────────────────────────────────────────────────────
    def fingerprint(self) -> dict[str, list[float]]:
        """Per-site scalars: what analysis actually consumes.

        A full genome is ~10.1M float32 parameters — 39 MB on disk — while
        every genotype-phenotype analysis reduces it to a handful of numbers
        per site. Writing the reduction for EVERY agent costs about 1.3 KB and
        removes the two limits that made the lowmut association underpowered:
        coverage (165 genomes from 11,741 births) and survivor bias (a periodic
        snapshot can only ever capture agents that were alive when it fired).

        ‖B@A‖_F is computed through the rank-sized identity
        ‖BA‖² = tr((AAᵀ)(BᵀB)), so it never forms the full ΔW.
        """
        keys, dnorm, rms_a, rms_b = [], [], [], []
        for site in self.spec.sites:
            a, b = self.factors[site.key]
            # The Gram matrices are rank x rank, so the float32 matmul is a
            # 3584-term reduction into 16x16 — accurate to ~1e-5 relative,
            # which is far below the resolution any analysis reads. Promoting
            # the factors to float64 first tripled the cost of this call for
            # digits nothing consumes.
            keys.append(site.key)
            gram = (a @ a.T).astype(np.float64) @ (b.T @ b).astype(np.float64)
            dnorm.append(float(np.sqrt(max(np.trace(gram), 0.0))))
            rms_a.append(round(float(np.sqrt(np.mean(
                np.square(a, dtype=np.float64)))), 8))
            rms_b.append(round(float(np.sqrt(np.mean(
                np.square(b, dtype=np.float64)))), 8))
        dnorm = [round(v, 8) for v in dnorm]
        return {"sites": keys, "delta_norm": dnorm,
                "rms_a": rms_a, "rms_b": rms_b}

    # ── persistence ───────────────────────────────────────────────────────
    def save(self, path: str | Path) -> None:
        from safetensors.numpy import save_file
        tensors = {}
        for key, (a, b) in self.factors.items():
            tensors[f"{key}.A"] = a
            tensors[f"{key}.B"] = b
        save_file(tensors, str(path))

    @classmethod
    def load(cls, path: str | Path, spec: GenomeSpec) -> "Genome":
        from safetensors.numpy import load_file
        tensors = load_file(str(path))
        factors = {}
        for s in spec.sites:
            factors[s.key] = (tensors[f"{s.key}.A"], tensors[f"{s.key}.B"])
        return cls(spec, factors)


def spec_from_dims(num_layers: int, projections: dict[str, tuple[int, int]],
                   rank: int, alpha: int) -> GenomeSpec:
    """Build a spec from explicit per-projection (in_dim, out_dim)."""
    sites = tuple(
        Site(layer, proj, dims[0], dims[1])
        for layer in range(num_layers)
        for proj, dims in projections.items()
    )
    return GenomeSpec(sites, rank, alpha)


def spec_from_hf_config(hf_config, target_modules: list[str], rank: int,
                        alpha: int) -> GenomeSpec:
    """Derive site shapes from a HuggingFace model config (llama/qwen family
    attention + MLP projections)."""
    hidden = hf_config.hidden_size
    n_heads = hf_config.num_attention_heads
    n_kv = getattr(hf_config, "num_key_value_heads", n_heads)
    head_dim = getattr(hf_config, "head_dim", None) or hidden // n_heads
    inter = getattr(hf_config, "intermediate_size", 4 * hidden)
    dims = {
        "q_proj": (hidden, n_heads * head_dim),
        "k_proj": (hidden, n_kv * head_dim),
        "v_proj": (hidden, n_kv * head_dim),
        "o_proj": (n_heads * head_dim, hidden),
        "gate_proj": (hidden, inter),
        "up_proj": (hidden, inter),
        "down_proj": (inter, hidden),
    }
    unknown = [m for m in target_modules if m not in dims]
    if unknown:
        raise ValueError(f"unsupported target modules: {unknown}")
    return spec_from_dims(
        hf_config.num_hidden_layers,
        {m: dims[m] for m in target_modules},
        rank, alpha,
    )
