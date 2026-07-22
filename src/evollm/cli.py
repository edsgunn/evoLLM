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
import shutil
import time
from pathlib import Path

import numpy as np

from .blocks import adapter_blocks_needed
from .config import Config, load_config
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


async def build_world(cfg: Config) -> World:
    spec = build_spec(cfg)
    if cfg.backend == "mock":
        from .engines.mock import POLICIES, MockEngine, WordTokenizer
        policy = POLICIES[cfg.mock.policy]
        tokenizer = WordTokenizer()
        engines = {room.id: MockEngine(default_policy=policy, seed=cfg.seed + i,
                                       tokenizer=tokenizer)
                   for i, room in enumerate(cfg.world.rooms)}
        adapter_blocks = cfg.mock.adapter_blocks
    elif cfg.backend == "vllm":
        from transformers import AutoConfig
        from .engines.vllm_engine import VLLMEngine, kv_block_bytes
        hf_config = AutoConfig.from_pretrained(cfg.model.name)
        block_bytes = kv_block_bytes(hf_config, cfg.world.block_size)
        adapter_blocks = adapter_blocks_needed(spec.adapter_bytes(), block_bytes)
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
    run_dir = Path(cfg.run.out_dir) / cfg.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(args.config, run_dir / "config.yaml")
    asyncio.run(_run(cfg, max_steps=args.steps))
    _print_report(run_dir)


def cmd_precheck_handshake(args) -> None:
    """§6: instantiate base-model agents (zero adapters) and measure the
    handshake completion rate directly, before committing the two weeks."""
    cfg = load_config(args.config)
    cfg.run_name = args.name or f"{cfg.run_name}-precheck"
    cfg.run.snapshot_every_steps = 0
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
    if not h["requests"] or not h["births"]:
        print("VERDICT: handshake base rate is negligible — the population "
              "will die out before selection can act. Prepare the scaffolded "
              "easier-agreement variant before committing to the main run.")
    else:
        print("VERDICT: non-zero handshake base rate; selection has something "
              "to work with.")


def cmd_measure_throughput(args) -> None:
    """§4.1: measure achievable concurrent-adapter throughput before fixing a
    population size. Sweeps the number of distinct resident adapters."""
    cfg = load_config(args.config)
    asyncio.run(_measure_throughput(cfg, args))


async def _measure_throughput(cfg: Config, args) -> None:
    from .engines.vllm_engine import VLLMEngine
    spec = build_spec(cfg)
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
    spec = build_spec(cfg)
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


def cmd_report(args) -> None:
    _print_report(Path(args.run_dir))


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
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("precheck-handshake",
                       help="measure the mate-handshake base rate (§6)")
    p.add_argument("-c", "--config", required=True)
    p.add_argument("--name")
    p.add_argument("--steps", type=int, default=5000)
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

    p = sub.add_parser("report", help="aggregate a run's event logs")
    p.add_argument("run_dir")
    p.set_defaults(func=cmd_report)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
