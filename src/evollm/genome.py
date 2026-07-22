"""Genome: a LoRA adapter as heritable material (§3).

Representation: for each adapted (layer, projection) site, a factor pair
(A, B) with A: (rank, in_dim), B: (out_dim, rank). Rank is fixed and uniform
across the population so adapter footprint is identical for every agent and
block accounting stays per-agent-uniform (§3.1).

Reproduction (§3.2): per-site uniform crossover — the child inherits (A, B)
wholesale from one parent per site, 50/50 — followed by small Gaussian
mutation. Wholesale inheritance sidesteps basis-arbitrariness: the child's
ΔW at each site is exactly one parent's ΔW at that site.

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
                  rng: np.random.Generator) -> "Genome":
        assert p1.spec == p2.spec
        factors = {}
        for s in p1.spec.sites:
            donor = p1 if rng.random() < 0.5 else p2
            a, b = donor.factors[s.key]
            a = a + rng.normal(0.0, mutation_std, size=a.shape).astype(np.float32)
            b = b + rng.normal(0.0, mutation_std, size=b.shape).astype(np.float32)
            factors[s.key] = (a, b)
        return cls(p1.spec, factors)

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
