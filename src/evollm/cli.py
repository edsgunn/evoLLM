"""Command-line entry points.

    evollm run                 -c config.yaml        the main experiment loop
    evollm precheck-handshake  -c config.yaml        §6 base-rate measurement
    evollm measure-throughput  -c config.yaml        §4.1 adapter-residency sweep
    evollm eval-surprise       -c config.yaml ...    §5 held-out surprise vs controls
    evollm report              <run_dir>             aggregate a run's event logs
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np

from .blocks import adapter_blocks_needed
from .config import Config, load_config, resolve_max_model_len
from .genome import Genome, GenomeSpec, spec_from_dims
from .world import World

# Tiny genome for the mock backend: the world layer only needs shapes.
_MOCK_SPEC = dict(num_layers=2, projections={"q_proj": (8, 8)})


def build_spec(cfg: Config) -> GenomeSpec:
    if cfg.backend == "mock":
        return spec_from_dims(rank=cfg.genome.rank, alpha=cfg.genome.alpha,
                              **_MOCK_SPEC)
    from transformers import AutoConfig
    from .genome import spec_from_hf_config
    hf_config = AutoConfig.from_pretrained(cfg.model.name)
    return spec_from_hf_config(hf_config, cfg.genome.target_modules,
                               cfg.genome.rank, cfg.genome.alpha)


def _run_stamp() -> str:
    """Slurm job id when there is one, else a timestamp — so every run gets
    its own event log."""
    return os.environ.get("SLURM_JOB_ID") or time.strftime("%Y%m%d-%H%M%S")


def _adapter_blocks(cfg: Config, spec: GenomeSpec) -> int:
    """Per-agent adapter footprint, uniform across the population (§3.1)."""
    if cfg.backend == "mock":
        return cfg.mock.adapter_blocks
    from transformers import AutoConfig
    from .engines.vllm_engine import kv_block_bytes
    hf_config = AutoConfig.from_pretrained(cfg.model.name)
    block_bytes = kv_block_bytes(hf_config, cfg.world.block_size)
    return adapter_blocks_needed(spec.adapter_bytes(), block_bytes)


def prepare(cfg: Config) -> tuple[GenomeSpec, int]:
    """Resolve everything an engine needs before it can be constructed.

    Every entry point that builds an engine must go through this — the engine
    needs a concrete max_model_len to size both its scheduler limit and the
    rope cache, and "auto" is only resolvable once the adapter footprint is
    known. Idempotent, so calling it twice is harmless.
    """
    spec = build_spec(cfg)
    adapter_blocks = _adapter_blocks(cfg, spec)
    cfg.model.max_model_len = resolve_max_model_len(cfg, adapter_blocks)
    return spec, adapter_blocks


async def build_world(cfg: Config) -> World:
    spec, adapter_blocks = prepare(cfg)
    if cfg.backend == "mock":
        from .engines.mock import POLICIES, MockEngine, WordTokenizer
        policy = POLICIES[cfg.mock.policy]
        tokenizer = WordTokenizer()
        engines = {room.id: MockEngine(default_policy=policy, seed=cfg.seed + i,
                                       tokenizer=tokenizer)
                   for i, room in enumerate(cfg.world.rooms)}
    elif cfg.backend == "vllm":
        from .engines.vllm_engine import VLLMEngine
        engines = {}
        # Rooms are constructed sequentially: each engine core inherits the
        # CUDA_VISIBLE_DEVICES set for its room at spawn time (§2.1).
        for room in cfg.world.rooms:
            engine = VLLMEngine(cfg, room, spec)
            await engine.start()
            engines[room.id] = engine
    else:
        raise ValueError(f"unknown backend {cfg.backend!r}")

    world = World(cfg, engines, spec, adapter_blocks)
    for room_id, controller in world.controllers.items():
        print(f"[{room_id}] capacity {controller.pool.capacity} blocks, "
              f"adapter footprint {adapter_blocks} blocks/agent")
    return world


async def _run(cfg: Config, zero_genomes: bool = False,
               max_steps: int | None = None) -> Path:
    run_dir = Path(cfg.run.out_dir) / cfg.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    world = await build_world(cfg)
    try:
        await world.seed(zero_genomes=zero_genomes)
        t0 = time.time()
        await world.run(max_steps=max_steps)
        elapsed = time.time() - t0
        steps = {r: c.step_count for r, c in world.controllers.items()}
        print(f"ran {steps} room steps in {elapsed:.1f}s; "
              f"final population {world.population}")
        world.final_snapshot()
    finally:
        world.close()
        for controller in world.controllers.values():
            await controller.engine.stop()
    return run_dir


def cmd_run(args) -> None:
    cfg = load_config(args.config)
    if args.name:
        cfg.run_name = args.name
    if getattr(args, "seed_from", None):
        cfg.seed_from = args.seed_from
    run_dir = Path(cfg.run.out_dir) / cfg.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(args.config, run_dir / "config.yaml")
    asyncio.run(_run(cfg, max_steps=args.steps))
    _print_report(run_dir)


def cmd_precheck_handshake(args) -> None:
    """§6: instantiate base-model agents (zero adapters) and measure the
    handshake completion rate directly, before committing the two weeks."""
    cfg = load_config(args.config)
    # Unique by default: a fixed name would append to (and silently merge
    # with) the previous precheck's event log.
    cfg.run_name = args.name or f"{cfg.run_name}-precheck-{_run_stamp()}"
    cfg.run.snapshot_every_steps = 0
    # The precheck is a diagnostic run: capture what agents actually emit.
    cfg.run.trace_turns = max(cfg.run.trace_turns, args.trace)
    cfg.run.context_snapshot_every_steps = \
        cfg.run.context_snapshot_every_steps or 2000
    # §6 asks for the *unassisted* base rate, so the precheck never refills —
    # and refill immigrants are drawn at random, which would contaminate a
    # zero-genome measurement anyway. Whether the run being gated refills
    # changes what counts as a fatal result, below.
    refill_in_run = cfg.refill.enabled
    cfg.refill.enabled = False
    asyncio.run(_run(cfg, zero_genomes=True, max_steps=args.steps))
    run_dir = Path(cfg.run.out_dir) / cfg.run_name
    from .report import aggregate
    stats = aggregate(run_dir)
    h = stats["handshake"]
    print("\n── handshake precheck (§6) ──")
    print(f"steps: {args.steps}  requests: {h['requests']}  "
          f"delivered: {h['delivered']}  valid accepts: {h['valid_accepts']}  "
          f"gen>0 births: {h['births']}")
    print(f"request→birth rate: {h['request_to_birth_rate']}")
    deaths = stats["deaths_by_cause"]
    if stats["invalid_deaths"]:
        print(f"VERDICT: FAIL — {len(stats['invalid_deaths'])} deaths were not "
              "scarcity events (§4.3). The substrate is leaking into the "
              "experiment; fix that before measuring anything.")
        raise SystemExit(3)
    print(f"deaths by cause: {deaths}")
    if not h["requests"] or not h["births"]:
        print("VERDICT: FAIL — handshake base rate is negligible; the "
              "population will die out before selection can act. Prepare the "
              "scaffolded easier-agreement variant before committing to the "
              "main run.")
        # Non-zero exit so an `sbatch --dependency=afterok` chain does not
        # launch the main run on a bootstrap that cannot happen (§6, §7).
        raise SystemExit(2)

    # §3.2: "birth rate must substantially exceed replacement rate for the
    # population to survive its first generations". A single lucky handshake
    # is not a viable bootstrap, and testing only for non-zero births once
    # passed a population that was collapsing 63 deaths to 1 birth.
    total_deaths = sum(deaths.values())
    print(f"births (gen>0): {h['births']}  deaths: {total_deaths}  "
          f"ratio: {h['births'] / max(total_deaths, 1):.2f}")
    if refill_in_run:
        # The run this gates admits immigrants when it drops below its floor,
        # so a sub-replacement rate is the condition refill exists to survive,
        # not a reason to abandon the run. Extinction is off the table; what
        # remains fatal is a handshake that never completes at all, which the
        # check above already caught.
        print("NOTE: the run being gated has refill enabled, so extinction is "
              "prevented and sub-replacement reproduction is expected — the "
              "replacement criterion is not applied. Whether the population "
              "becomes self-sustaining is measured in the run itself, as "
              "self-sufficiency = births / (births + refills).")
        print("VERDICT: PASS — the handshake completes; refill carries the "
              "population while selection searches.")
        return
    if h["births"] <= total_deaths:
        print("VERDICT: FAIL — births are at or below replacement, so the "
              "population shrinks every generation and reaches extinction "
              "before selection has anything to act on (§3.2, §7). Either "
              "scaffold the acceptance condition (§6) or raise the birth "
              "opportunity rate before committing to the main run.")
        raise SystemExit(4)
    print("VERDICT: PASS — handshake bootstraps and births exceed "
          "replacement; selection has something to work with.")


def cmd_measure_throughput(args) -> None:
    """§4.1: measure achievable concurrent-adapter throughput before fixing a
    population size. Sweeps the number of distinct resident adapters."""
    cfg = load_config(args.config)
    asyncio.run(_measure_throughput(cfg, args))


async def _measure_throughput(cfg: Config, args) -> None:
    from .engines.vllm_engine import VLLMEngine
    spec, _ = prepare(cfg)
    room = cfg.world.rooms[0]
    engine = VLLMEngine(cfg, room, spec)
    await engine.start()
    rng = np.random.default_rng(cfg.seed)
    prompt = engine.tokenize("The agents live in rooms of finite memory. ")

    async def one_agent(agent_id: str, n_tokens: int) -> int:
        handle = engine.start_turn(agent_id, prompt, n_tokens)
        produced = 0
        while True:
            event = await handle.next_event()
            from .engines.base import TurnEnded
            if isinstance(event, TurnEnded):
                return produced
            produced += 1

    print(f"engine up: {engine.capacity_blocks()} authoritative blocks "
          f"(max_loras={cfg.engine.max_loras})")
    for n_adapters in [int(x) for x in args.sweep.split(",")]:
        ids = [f"bench{n_adapters}_{i}" for i in range(n_adapters)]
        for aid in ids:
            await engine.register_adapter(
                aid, Genome.random(spec, cfg.genome.init_scale, rng))
        t0 = time.time()
        counts = await asyncio.gather(*(one_agent(a, args.tokens) for a in ids))
        dt = time.time() - t0
        total = sum(counts)
        print(f"adapters={n_adapters:4d}  tokens={total:6d}  "
              f"time={dt:6.1f}s  tok/s={total / dt:8.1f}  "
              f"tok/s/agent={total / dt / n_adapters:6.1f}")
        for aid in ids:
            await engine.unregister_adapter(aid)
    await engine.stop()


def cmd_eval_surprise(args) -> None:
    cfg = load_config(args.config)
    from .evaluate import evaluate_snapshots, load_streams
    spec, _ = prepare(cfg)   # eval builds an LLM too, so it needs the same
    streams = load_streams(args.streams)
    snapshot_dirs = sorted(Path(p) for p in args.snapshots)
    results = evaluate_snapshots(cfg, spec, snapshot_dirs, streams,
                                 n_controls=args.controls, seed=cfg.seed,
                                 prefix=args.prefix)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"base model NLL/token:      {results['controls']['base_model']:.4f}")
    ri = results["controls"]["random_init"]
    if ri["mean"] is not None:
        print(f"random-init controls:      {ri['mean']:.4f} ± {ri['std']:.4f} "
              f"(n={ri['n']})")
    for snap in results["snapshots"]:
        print(f"{snap['path']}: population {snap['population_mean']:.4f}  "
              f"by generation {snap['by_generation']}")
    print(f"written to {out}")


def cmd_plot(args) -> None:
    from .plots import build_html
    out = args.out or (Path(args.run_dir) / "report.html")
    path = build_html(args.run_dir, out, max_families=args.max_families)
    print(f"wrote {path}  ({path.stat().st_size / 1024:.0f} KB)")


def cmd_report(args) -> None:
    _print_report(Path(args.run_dir))


def cmd_analyse(args) -> None:
    """Population analysis: lineage stratification, strategy niches, and
    per-site genotype-phenotype association."""
    from .analysis import analyse_run, format_report as fmt
    result = analyse_run(args.run_dir, n_perm=args.permutations,
                         min_lineage=args.min_lineage, min_turns=args.min_turns,
                         n_pcs=args.pcs, seed=args.seed,
                         lineage_generation=args.lineage_generation)
    text = fmt(result)
    print()
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n")
        print(f"\nwrote {args.out}")
    if args.traits_csv:
        result["_pheno"].to_csv(args.traits_csv)
        print(f"wrote {args.traits_csv}  "
              f"({len(result['_pheno']):,} agents x "
              f"{len(result['_pheno'].columns)} traits)")


def _print_report(run_dir: Path) -> None:
    from .report import aggregate, format_report
    print()
    print(format_report(aggregate(run_dir)))


def main() -> None:
    parser = argparse.ArgumentParser(prog="evollm")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("run", help="run the experiment")
    p.add_argument("-c", "--config", required=True)
    p.add_argument("--name", help="override run_name")
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--seed-from", dest="seed_from",
                   help="checkpoint directory to start the population from, "
                        "skipping the random search for a viable one")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("precheck-handshake",
                       help="measure the mate-handshake base rate (§6)")
    p.add_argument("-c", "--config", required=True)
    p.add_argument("--name")
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--trace", type=int, default=4000,
                   help="raw action turns to log per room, for diagnosis")
    p.set_defaults(func=cmd_precheck_handshake)

    p = sub.add_parser("measure-throughput",
                       help="concurrent-adapter throughput sweep (§4.1)")
    p.add_argument("-c", "--config", required=True)
    p.add_argument("--sweep", default="1,4,8,16,32,64")
    p.add_argument("--tokens", type=int, default=256)
    p.set_defaults(func=cmd_measure_throughput)

    p = sub.add_parser("eval-surprise",
                       help="held-out surprise vs unevolved controls (§5)")
    p.add_argument("-c", "--config", required=True)
    p.add_argument("--snapshots", nargs="+", required=True,
                   help="snapshot directories (runs/<name>/snapshots/<room>/step_*)")
    p.add_argument("--streams", required=True,
                   help="JSONL file of held-out observation streams")
    p.add_argument("--controls", type=int, default=8)
    p.add_argument("--prefix", default="",
                   help="optional shared context prefix for scoring")
    p.add_argument("--out", default="surprise_results.json")
    p.set_defaults(func=cmd_eval_surprise)

    p = sub.add_parser("plot", help="visualise a run: lineage, occupancy, blocks")
    p.add_argument("run_dir")
    p.add_argument("-o", "--out", default=None,
                   help="output HTML (default: <run_dir>/report.html)")
    p.add_argument("--max-families", type=int, default=28,
                   help="kinship groups drawn, largest first")
    p.set_defaults(func=cmd_plot)

    p = sub.add_parser("analyse",
                       help="population analysis: lineages, strategies, genes")
    p.add_argument("run_dir")
    p.add_argument("--permutations", type=int, default=500,
                   help="permutation draws for every null (default 500)")
    p.add_argument("--min-lineage", type=int, default=20,
                   help="drop lineages with fewer agents than this (default 20)")
    p.add_argument("--min-turns", type=int, default=5,
                   help="drop agents with fewer turns than this (default 5)")
    p.add_argument("--pcs", type=int, default=5,
                   help="genotype PCs used as structure covariates (default 5)")
    p.add_argument("--lineage-generation", type=int, default=None,
                   help="cut lineages at this generation's dominant ancestor "
                        "instead of at founders; use when the population has "
                        "gone panmictic and founder labels separate nothing")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", help="also write the report to this path")
    p.add_argument("--traits-csv", help="write the per-agent trait table here")
    p.set_defaults(func=cmd_analyse)

    p = sub.add_parser("report", help="aggregate a run's event logs")
    p.add_argument("run_dir")
    p.set_defaults(func=cmd_report)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
