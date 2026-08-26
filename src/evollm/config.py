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
    # Parents carry their children's adapter blocks (§3.2). Reproduction is
    # otherwise free: the child's 22 adapter blocks come out of the room pool
    # and neither parent pays anything, which is why offspring number has
    # variance 40-61 against a mean of 3 and effective population size sits at
    # 2-6.
    #
    # This is deliberately NOT a tax. No extra memory is allocated — the
    # blocks charged are the same ones the engine really registered for the
    # child's adapter, and the child stops paying for them. Only the owner
    # changes, so the incentive costs no device memory to create.
    #
    # It only bites under eviction: random_holder, where the victim is drawn
    # in proportion to blocks held. Under `requester` a parent's extra
    # holdings do not raise its own hazard, so the charge would be inert.
    parental_investment: bool = False
    initial_population_per_room: int = 8
    # How an agent absorbs a queued utterance (§2.3's clock).
    #
    #   "utterance" — the whole utterance enters context in one step. Hearing
    #       a 35-token message costs 35 tokens of context at once, while
    #       acting costs one token per step. This is what makes speech
    #       genuinely expensive to listeners (§2.4) and lifetime dependent on
    #       behaviour at all.
    #   "token" — one token per step, the literal reading of §2.3. Measured
    #       consequence: every agent's context grew at exactly one token per
    #       step regardless of what it did, so speech cost listeners nothing
    #       and survival was independent of behaviour.
    #
    # Both keep the world clock uniform — every living agent advances exactly
    # one step — and neither refers to wall-clock time.
    # Which actions exist in this world (§2.4). The say/tell choice is the
    # experimental variable: `say` broadcasts, so one generated token becomes
    # N-1 observation tokens and the observation economy diverges with room
    # size; `tell` is directed, closing that economy at 1:1 but giving up the
    # common knowledge a public channel creates. An action an agent emits that
    # is not in this list is refused, and the agent is told so — it learns the
    # repertoire by attempting it, as it learns the room by moving in it.
    tools: list[str] = field(
        default_factory=lambda: ["say", "mate", "go"])
    # Frame every block in the model's trained chat format:
    #   <|im_start|>system\n ...prompt... <|im_end|>
    #   <|im_start|>user\n ...observation... <|im_end|>
    #   <|im_start|>assistant\n ...action... <|im_end|>
    # Instruct models are trained on nothing else, and running them on a bare
    # stream put every token they produced out of distribution. The markers
    # are ordinary tokens: metered on the clock and charged in blocks like
    # anything else. Set false to reproduce the original bare stream.
    chat_format: bool = True
    # When an agent stops reading and starts acting.
    #
    #   "one"   — read one utterance, then act (§2.3's literal swap rule).
    #   "drain" — read until the queue is empty, then act.
    #
    # Under "one" an agent with a deep queue answers an action ~one utterance
    # at a time, so a response is read many turns after the action that caused
    # it: measured mean backlog was 7,333 tokens on 7B, roughly 120-180 queued
    # utterances, which makes it impossible for an agent to associate any
    # outcome with its cause. "drain" empties the queue first, so an agent
    # always acts on the present.
    #
    # The cost is that a flooded agent cannot act at all: if observations
    # arrive faster than it drains, it reads until it dies. That is bounded
    # (context grows either way) but it makes speech genuinely weaponisable
    # under `say`, where one utterance becomes N-1. Under `tell` the economy
    # is 1:1 and starvation cannot compound.
    read_policy: str = "one"
    observation_absorption: str = "utterance"
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

    # How sites are recombined. "chromosomal" cuts the layer-major site list
    # into contiguous chromosomes and takes one crossover point in each, so
    # interacting projections stay together (~4 parent switches per child);
    # "uniform" flips a coin per site (~55 switches across 112 sites).
    #
    # Chromosomal is the default because it is the only scheme measured to
    # transmit anything: run 6071674 gave h2 = +0.61 with a midparent /
    # single-parent slope ratio of 2.78 -- close to the 2.0 that additive
    # genetic transmission predicts -- while uniform gave 0.39 on the same
    # trait, indistinguishable from the shared-environment control and so
    # not inheritance at all.
    crossover: str = "chromosomal"
    # Linkage dial for the chromosomal scheme: 1 = one cut over the whole
    # genome, len(sites) = identical to uniform.
    chromosomes: int = 3
    # "additive" adds N(0, std) — which random-walks, and measurably did.
    # "multiplicative" scales by (1 + N(0, std)), holding relative
    # perturbation constant instead of letting magnitude compound.
    mutation: str = "additive"


@dataclass
class ModelConfig:
    name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    dtype: str = "bfloat16"

    # "auto" derives the ceiling from the largest room so that pool exhaustion
    # always binds first — i.e. the context limit stops being part of the
    # experiment. An int pins it explicitly.
    max_model_len: int | str = "auto"

    # Let agents run past the base model's trained window. The rope cos/sin
    # cache is sized from the HF config, not from max_model_len, and
    # `cos_sin_cache.index_select(0, positions)` is unchecked — so exceeding
    # the window without growing the cache is an out-of-bounds CUDA read, not
    # graceful degradation. Setting this passes
    # hf_overrides={"max_position_embeddings": max_model_len}, which grows the
    # cache so out-of-distribution positions degrade quality rather than
    # corrupting memory. (VLLM_ALLOW_LONG_MAX_MODEL_LEN is deliberately NOT
    # used: it lifts the length check without growing the cache.)
    extend_context: bool = True

    # Optional RoPE scaling (e.g. YaRN) to retain quality past the window:
    #   rope_scaling: {rope_type: yarn, factor: 4.0, original_max_position_embeddings: 32768}
    rope_scaling: dict | None = None


@dataclass
class EngineConfig:
    gpu_memory_utilization: float = 0.85
    max_loras: int = 8                # GPU-resident adapter slots (working set)
    max_cpu_loras: int = 128          # CPU-side cache; must hold a room's population
    max_num_seqs: int = 64
    enable_prefix_caching: bool = True
    enable_chunked_prefill: bool = True
    # Blocks held back from the authoritative pool so the engine can never be
    # asked for memory the controller believes exists (§4.2/§4.3). Only binds
    # when capacity_blocks is left unset; a room that names its own capacity
    # sets the headroom directly. See preemption_budget for what happens when
    # the engine runs short anyway.
    safety_margin_blocks: int = 64
    # How many engine preemptions a run tolerates before aborting.
    #
    # This used to abort on the FIRST one, on the reasoning that a preemption
    # made subsequent deaths unattributable. That reasoning was wrong. vLLM's
    # _preempt_request frees the request's KV blocks and resets
    # num_computed_tokens to 0, but leaves _output_token_ids untouched: the
    # request goes back to the waiting queue, recomputes its prefix and
    # continues from exactly where it was. No token is lost, none is emitted
    # twice, and no agent dies of it, because deaths are decided by BlockPool
    # and never by the engine. Aborting on one transient event cost four
    # multi-hour runs, two of them past ten hours.
    #
    # A budget keeps the signal without the brittleness: every preemption is
    # logged as an event so any analysis can see them, and a room that is
    # genuinely over-subscribed still stops rather than quietly degrading.
    # Set to 0 to restore the old abort-on-first behaviour.
    preemption_budget: int | None = 200
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
class RefillConfig:
    """Keep the arena populated so selection has something to act on (§6, §7).

    Measured base rates put reproduction far below replacement, so a room runs
    down to extinction — §7's "uninformative" outcome — before any lineage can
    establish. Refill admits fresh immigrants, perturbed from the base model,
    whenever the room falls below a floor. The population then random-walks
    until some lineage reproduces faster than it dies, after which the floor
    stops being reached and refill stops firing on its own.

    This is not scaffolding in §6's sense: it lowers no acceptance bar, scores
    nobody, and is blind to what any agent did — it only tops up numbers. What
    it changes is the question. Instead of "can 16 founders bootstrap", the
    experiment asks "given a continuous supply of variation, does a
    self-sustaining lineage emerge". The answer is read off the refill rate:
    `births / (births + refills)` rising toward 1 is takeoff, and a flat rate
    is a population being kept alive by immigration rather than by descent.
    """
    enabled: bool = False

    # The trigger is **blocks**, not head-count: admit immigrants while more
    # than this fraction of the room's pool is free. Expressed in the world's
    # own currency (§2.2) — empty space invites colonisation.
    #
    # A head-count floor was the trigger first, and it was measurably wrong.
    # With `min_population: 8` the rooms sat pinned at exactly 8 agents while
    # the pool was 92-94% full, so every immigrant was inserted into an almost
    # full room, took its 22 adapter blocks, and brought everyone else's death
    # forward. One 4000-block room needed 3,441 refills on that treadmill
    # while an 8000-block room needed 30.
    max_free_fraction: float | None = None
    # Optional head-count floor, kept only as a backstop. 0 disables it.
    min_population: int = 0
    # Immigrants admitted per check. Without a cap the block trigger would
    # admit hundreds at once — an adapter is only 22 blocks, so filling half a
    # 48,000-block room by adapters alone would mean ~1,000 agents in a single
    # check. The room is meant to fill with *contexts*, not with bodies.
    max_per_check: int = 4
    check_every_steps: int = 200
    # Immigrants are drawn the same way generation zero was, unless overridden.
    perturbation_scale: float | None = None
    # Hard ceiling on immigrants per run; None for unlimited. A run that only
    # survives by immigration should be visibly that, not quietly propped up.
    max_total: int | None = None

    # ── takeoff detection ─────────────────────────────────────────────────
    # A room is judged self-sustaining when it has gone `takeoff_window_steps`
    # without needing an immigrant while producing at least `takeoff_min_births`
    # children: the population replaced itself by descent. Finding that state
    # is the expensive part of a run — a random walk over initialisations — so
    # it is checkpointed the moment it is reached, and a later run can start
    # from the checkpoint instead of searching again (`seed_from`).
    takeoff_window_steps: int = 5000
    takeoff_min_births: int = 3
    checkpoint_on_takeoff: bool = True


@dataclass
class RunConfig:
    max_world_steps: int = 100_000
    # Per-site genome summaries for EVERY agent at creation (~1.3 KB each),
    # written to <run>/genomes/<room>.jsonl.
    #
    # A full genome is 39 MB, so snapshotting often enough to genotype most of
    # a population would cost hundreds of GB per run — and would still miss
    # every agent born and dead between two snapshots, because a snapshot can
    # only capture the living. lowmut genotyped 165 agents out of 11,741
    # births that way, which is what left its gene-behaviour associations
    # underpowered. Fingerprints cover everyone for the price of a rounding
    # error, and full snapshots stay for work that needs the actual factors.
    genome_fingerprints: bool = True
    snapshot_every_steps: int = 5_000
    occupancy_every_steps: int = 500
    out_dir: str = "runs"
    # Log the raw text of this many action turns per room, with how each was
    # parsed. Costs log volume only, and is the only way to see what agents
    # actually emit — without it a malformed turn is a bare counter.
    trace_turns: int = 0

    # Periodically dump raw agent contexts, verbatim, exactly as the model
    # reads them — special tokens and all.
    #
    # trace_turns records what an agent *emitted*; this records what it *saw*.
    # For a long time nothing recorded the latter, and the consequence was that
    # every model ran for weeks on a bare token stream with no <|im_start|>
    # role markers — completely out of its trained distribution — while the
    # symptoms (empty turns, malformed actions) were repeatedly misattributed
    # to the models. One look at a raw context would have shown it.
    context_snapshot_every_steps: int = 0
    context_snapshot_agents: int = 3      # agents sampled per room per dump
    context_head_tokens: int = 600        # the opening, where framing shows
    context_tail_tokens: int = 900        # the present, where behaviour shows


@dataclass
class Config:
    run_name: str = "dev"
    seed: int = 0
    # Start from a checkpointed population instead of random initialisations.
    # Path to a checkpoint directory (or one of its per-room subdirectories);
    # genomes are drawn from it to fill each room. This is how a run skips the
    # random search that a previous run already paid for.
    seed_from: str | None = None
    backend: str = "mock"             # "mock" | "vllm"
    model: ModelConfig = field(default_factory=ModelConfig)
    world: WorldConfig = field(default_factory=WorldConfig)
    genome: GenomeConfig = field(default_factory=GenomeConfig)
    engine: EngineConfig = field(default_factory=EngineConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    mock: MockConfig = field(default_factory=MockConfig)
    refill: RefillConfig = field(default_factory=RefillConfig)
    run: RunConfig = field(default_factory=RunConfig)

    def room_ids(self) -> list[str]:
        return [r.id for r in self.world.rooms]

    def _clustered_adjacency(self) -> dict[str, list[str]]:
        """Complete within a device, sparse between devices.

        Every run so far has been panmictic — 1.04-1.47 EFFECTIVE lineages and
        95-99.5% of agents in a single family — because one big room per GPU
        under `edges: complete` mixes everything. Nothing could diverge, so
        founder labels carried no information and no niche ever formed.

        This makes each device a cluster of small, fully-connected rooms, and
        joins the devices in a ring. Migration within a cluster is easy and
        migration between clusters has to pass through a single room, which is
        isolation by distance: the structure a metapopulation needs before
        lineages can differentiate at all.

        A ring rather than a denser graph deliberately. `go` is the only
        migration channel and it succeeds 65-70% of the time, so migration is
        already generous; anything denser reproduces the panmictic pool with
        extra steps.
        """
        rooms = self.world.rooms
        by_device: dict = {}
        for r in rooms:
            by_device.setdefault(r.gpu if r.gpu is not None else r.id,
                                 []).append(r.id)
        devices = list(by_device)
        adj: dict[str, list[str]] = {r.id: [] for r in rooms}
        for members in by_device.values():           # complete inside a device
            for a in members:
                adj[a].extend(m for m in members if m != a)
        if len(devices) > 1:                         # ring across devices
            for i, dev in enumerate(devices):
                nxt = devices[(i + 1) % len(devices)]
                if len(devices) == 2 and i == 1:
                    break                            # a 2-ring is one edge
                a, b = by_device[dev][0], by_device[nxt][0]
                adj[a].append(b)
                adj[b].append(a)
        return {k: sorted(set(v)) for k, v in adj.items()}

    def adjacency(self) -> dict[str, list[str]]:
        """Room graph as adjacency lists. 'complete' connects every pair."""
        ids = self.room_ids()
        if self.world.edges == "complete":
            return {r: [s for s in ids if s != r] for r in ids}
        if self.world.edges == "clustered":
            return self._clustered_adjacency()
        adj: dict[str, list[str]] = {r: [] for r in ids}
        for a, b in self.world.edges:
            if a not in adj or b not in adj:
                raise ValueError(f"edge ({a}, {b}) references unknown room")
            adj[a].append(b)
            adj[b].append(a)
        return adj


def resolve_max_model_len(cfg: "Config", adapter_blocks: int) -> int:
    """Resolve model.max_model_len, expanding "auto".

    The context window is a property of the base model's training, not of the
    world. Where it binds before the room's pool does, every death becomes an
    infrastructure artefact (§4.3). "auto" removes it from the experiment by
    sizing the ceiling to the most any single agent could ever hold — the
    largest room's whole pool — so scarcity is always what kills.

    Agents will run far past the base model's trained window and behave worse
    there. That is intended: degradation is a selection pressure like any
    other, and nothing in the environment refers to it.
    """
    block_size = cfg.world.block_size
    if cfg.model.max_model_len != "auto":
        return int(cfg.model.max_model_len)
    capacities = [r.capacity_blocks for r in cfg.world.rooms]
    if any(c is None for c in capacities):
        # Rooms that take their capacity from the engine create a circularity:
        # capacity comes from the engine's pool, but the engine needs
        # max_model_len before it can be built. Broken with an UPPER BOUND on
        # the pool, computed from device memory without starting anything. The
        # real pool is always smaller (the weights come out of the same
        # budget), so the ceiling derived here can never bind — which is the
        # point: scarcity, not the context window, decides who dies.
        bound = _pool_upper_bound(cfg)
        if bound is None:
            raise ValueError(
                'model.max_model_len: "auto" needs either explicit '
                "capacity_blocks on every room, or a visible CUDA device to "
                "size the ceiling from")
        # A room gets its DEVICE's pool divided by the rooms sharing that
        # device, so the ceiling is sized to a room's share rather than the
        # whole card. Sizing it to the card would leave the context window far
        # above anything an agent could reach, which is the situation the
        # ceiling exists to prevent — and would stretch rope ten times further
        # than the run needs.
        per_device: dict = {}
        for r in cfg.world.rooms:
            key = r.gpu if r.gpu is not None else r.id
            per_device[key] = per_device.get(key, 0) + 1
        busiest = max(per_device.values()) if per_device else 1
        return (bound // busiest - adapter_blocks) * block_size + 1
    # +1 because vLLM needs the prompt plus at least one output token to fit.
    return (max(capacities) - adapter_blocks) * block_size + 1


def _pool_upper_bound(cfg: "Config") -> int | None:
    """Most KV blocks the engine could possibly hold on one device.

    Erring low is safe: capacity clamps and the context window comes back into
    play. Erring HIGH is fatal, which an earlier version of this got wrong. It
    ignored the model weights on the reasoning that overshooting only wasted
    rope-cache memory. It does not: vLLM refuses to start unless a single
    request of max_model_len fits in the KV cache, so a bound that is too
    generous kills the engine at init:

        ValueError: To serve at least one request with the model's max seq len
        (1645057), 87.86 GiB KV cache is needed, which is larger than the
        available KV cache memory (71.03 GiB)

    So the weights are subtracted, and a further reserve for activations,
    captured CUDA graphs and fragmentation, all of which come out of the same
    utilisation budget.
    """
    try:
        import pynvml
        from transformers import AutoConfig
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        total = int(pynvml.nvmlDeviceGetMemoryInfo(h).total)
    except Exception:
        return None
    hf = AutoConfig.from_pretrained(cfg.model.name)
    layers = hf.num_hidden_layers
    n_kv = getattr(hf, "num_key_value_heads", hf.num_attention_heads)
    n_heads = hf.num_attention_heads
    head_dim = getattr(hf, "head_dim", None) or hf.hidden_size // n_heads
    dtype_bytes = 2                       # bfloat16 / fp16
    block_bytes = (cfg.world.block_size * 2 * layers * n_kv * head_dim
                   * dtype_bytes)

    hidden, vocab = hf.hidden_size, hf.vocab_size
    inter = getattr(hf, "intermediate_size", 4 * hidden)
    per_layer = (hidden * n_heads * head_dim          # q
                 + 2 * hidden * n_kv * head_dim       # k, v
                 + n_heads * head_dim * hidden        # o
                 + 3 * hidden * inter)                # gate, up, down
    params = vocab * hidden + layers * per_layer
    if not getattr(hf, "tie_word_embeddings", False):
        params += vocab * hidden                      # lm_head
    weight_bytes = params * dtype_bytes
    RESERVE = 3 * 1024 ** 3       # activations, CUDA graphs, fragmentation

    budget = cfg.engine.gpu_memory_utilization * total - weight_bytes - RESERVE
    if budget <= 0:
        return None
    return int(budget / block_bytes)


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
    (Config, "refill"): RefillConfig,
    (Config, "run"): RunConfig,
}


def _validate_tools(cfg: "Config") -> None:
    from .actions import ALL_TOOLS

    unknown = [t for t in cfg.world.tools if t not in ALL_TOOLS]
    if unknown:
        raise ValueError(f"unknown tools {unknown}; known: {list(ALL_TOOLS)}")
    if not cfg.world.tools:
        raise ValueError("world.tools must enable at least one action")
    if "accept" in cfg.world.tools and "mate" not in cfg.world.tools:
        raise ValueError("world.tools: 'accept' is meaningless without 'mate'")


def load_config(path: str | Path) -> Config:
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    cfg = _build(Config, data)
    if not cfg.world.rooms:
        raise ValueError("config must declare at least one room")
    if isinstance(cfg.world.edges, str) and \
            cfg.world.edges not in ("complete", "clustered"):
        raise ValueError(
            f"world.edges must be 'complete', 'clustered', or a list of "
            f"[room, room] pairs; got {cfg.world.edges!r}")
    if cfg.world.eviction not in (EVICT_REQUESTER, EVICT_RANDOM_HOLDER):
        raise ValueError(f"unknown eviction policy {cfg.world.eviction!r}")
    if cfg.genome.crossover not in ("uniform", "chromosomal"):
        raise ValueError("genome.crossover must be 'uniform' or 'chromosomal', "
                         f"got {cfg.genome.crossover!r}")
    if cfg.genome.mutation not in ("additive", "multiplicative"):
        raise ValueError("genome.mutation must be 'additive' or "
                         f"'multiplicative', got {cfg.genome.mutation!r}")
    _validate_tools(cfg)
    if cfg.world.read_policy not in ("one", "drain"):
        raise ValueError("world.read_policy must be 'one' or 'drain', got "
                         f"{cfg.world.read_policy!r}")
    if cfg.world.observation_absorption not in ("utterance", "token"):
        raise ValueError("world.observation_absorption must be 'utterance' or "
                         f"'token', got {cfg.world.observation_absorption!r}")
    return cfg
