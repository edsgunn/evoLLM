"""Experiment configuration.

A single YAML file describes a run. Everything the environment does is
parameterised here; anything not in this file is a property of the hardware
(room capacity read from the device) or of the base model (genome shapes read
from its config).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# Eviction policies (§2.5): who dies when a room's pool is exhausted.
EVICT_REQUESTER = "requester"          # the agent whose KV growth hit the wall
EVICT_RANDOM_HOLDER = "random_holder"  # a random held block's owner


@dataclass
class RoomConfig:
    id: str
    gpu: int | None = None            # CUDA device index (vllm backend)
    capacity_blocks: int | None = None  # None => derive from the engine (§4.2)


@dataclass
class WorldConfig:
    rooms: list[RoomConfig] = field(default_factory=list)
    # Edges as [room_id, room_id] pairs, or the string "complete".
    edges: Any = "complete"
    block_size: int = 16              # tokens per KV block
    eviction: str = EVICT_REQUESTER
    initial_population_per_room: int = 8
    max_action_tokens: int = 256      # hard cap per action turn; hitting it forces turn end
    mate_window_tokens: int = 64      # acceptance window, in the target's own tokens (§2.4)
    viability_probe_turns: int = 3    # k of §3.2: well-formed action within first k turns


@dataclass
class GenomeConfig:
    rank: int = 16
    alpha: int = 32
    target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    init_scale: float = 0.02          # std of gen-0 factor entries
    mutation_std: float = 0.01        # std of Gaussian mutation after crossover (§3.2)


@dataclass
class ModelConfig:
    name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    dtype: str = "bfloat16"
    max_model_len: int = 8192


@dataclass
class EngineConfig:
    gpu_memory_utilization: float = 0.85
    max_loras: int = 8                # GPU-resident adapter slots (working set)
    max_cpu_loras: int = 128          # CPU-side cache; must hold a room's population
    max_num_seqs: int = 64
    enable_prefix_caching: bool = True
    enable_chunked_prefill: bool = True
    # Blocks held back from the authoritative pool so the engine can never be
    # asked for memory the controller believes exists (§4.2/§4.3). Engine-side
    # preemption is an integrity violation, not a game event.
    safety_margin_blocks: int = 64
    adapter_dir: str = "/dev/shm/evollm"  # tmpfs for adapter registration (§4.1 path 1)


@dataclass
class SamplingConfig:
    temperature: float = 1.0
    top_p: float = 1.0


@dataclass
class MockConfig:
    """Parameters that only exist for the mock backend (tests / dry runs)."""
    adapter_blocks: int = 2           # per-agent adapter footprint in blocks
    policy: str = "random"            # default scripted policy name
    system_prompt_tokens: int | None = None  # override prompt length (tests)


@dataclass
class RunConfig:
    max_world_steps: int = 100_000
    snapshot_every_steps: int = 5_000
    occupancy_every_steps: int = 500
    out_dir: str = "runs"


@dataclass
class Config:
    run_name: str = "dev"
    seed: int = 0
    backend: str = "mock"             # "mock" | "vllm"
    model: ModelConfig = field(default_factory=ModelConfig)
    world: WorldConfig = field(default_factory=WorldConfig)
    genome: GenomeConfig = field(default_factory=GenomeConfig)
    engine: EngineConfig = field(default_factory=EngineConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    mock: MockConfig = field(default_factory=MockConfig)
    run: RunConfig = field(default_factory=RunConfig)

    def room_ids(self) -> list[str]:
        return [r.id for r in self.world.rooms]

    def adjacency(self) -> dict[str, list[str]]:
        """Room graph as adjacency lists. 'complete' connects every pair."""
        ids = self.room_ids()
        if self.world.edges == "complete":
            return {r: [s for s in ids if s != r] for r in ids}
        adj: dict[str, list[str]] = {r: [] for r in ids}
        for a, b in self.world.edges:
            if a not in adj or b not in adj:
                raise ValueError(f"edge ({a}, {b}) references unknown room")
            adj[a].append(b)
            adj[b].append(a)
        return adj


def _build(cls, data: dict) -> Any:
    """Recursively build a dataclass from a plain dict, rejecting unknown keys."""
    kwargs = {}
    fields = {f.name: f for f in cls.__dataclass_fields__.values()}
    for key, value in data.items():
        if key not in fields:
            raise ValueError(f"unknown config key {key!r} for {cls.__name__}")
        ftype = fields[key].type
        sub = _NESTED.get((cls, key))
        if sub is not None:
            kwargs[key] = _build(sub, value)
        elif (cls, key) == (WorldConfig, "rooms"):
            kwargs[key] = [_build(RoomConfig, r) for r in value]
        else:
            kwargs[key] = value
    return cls(**kwargs)


_NESTED = {
    (Config, "model"): ModelConfig,
    (Config, "world"): WorldConfig,
    (Config, "genome"): GenomeConfig,
    (Config, "engine"): EngineConfig,
    (Config, "sampling"): SamplingConfig,
    (Config, "mock"): MockConfig,
    (Config, "run"): RunConfig,
}


def load_config(path: str | Path) -> Config:
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    cfg = _build(Config, data)
    if not cfg.world.rooms:
        raise ValueError("config must declare at least one room")
    if cfg.world.eviction not in (EVICT_REQUESTER, EVICT_RANDOM_HOLDER):
        raise ValueError(f"unknown eviction policy {cfg.world.eviction!r}")
    return cfg
