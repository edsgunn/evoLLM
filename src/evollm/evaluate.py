"""Held-out surprise measurement (§5, §7).

Scores population snapshots against held-out observation streams: mean NLL
per token of the stream under base+adapter. The load-bearing comparison is
population vs unevolved controls — random-initialisation genomes drawn the
same way generation zero was, plus the zero adapter (the frozen base) —
because the confound this experiment is most exposed to is that the base
model already exhibits the behaviour.

There are no weight updates anywhere in this codebase, so any decline across
generations relative to controls is attributable to selection over
initialisations expressed in context.

Runs offline (no world, no economy): plain teacher-forced scoring via
prompt_logprobs.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .config import Config
from .genome import Genome, GenomeSpec


def load_streams(path: str | Path) -> list[str]:
    streams = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                record = json.loads(line)
                streams.append(record.get("text") or record["stream"])
    return streams


def snapshot_genomes(snapshot_dir: str | Path, spec: GenomeSpec
                     ) -> list[tuple[dict, Genome]]:
    """Load (metadata, genome) pairs from a snapshot directory."""
    snapshot_dir = Path(snapshot_dir)
    with open(snapshot_dir / "population.json") as f:
        population = json.load(f)
    out = []
    for meta in population["agents"]:
        genome = Genome.load(snapshot_dir / f"{meta['id']}.safetensors", spec)
        out.append((meta, genome))
    return out


class SurpriseScorer:
    """Teacher-forced NLL of held-out streams under base+adapter."""

    def __init__(self, cfg: Config, spec: GenomeSpec, prefix: str = ""):
        from vllm import LLM
        self.cfg = cfg
        self.spec = spec
        self.llm = LLM(
            model=cfg.model.name,
            dtype=cfg.model.dtype,
            max_model_len=cfg.model.max_model_len,
            gpu_memory_utilization=cfg.engine.gpu_memory_utilization,
            enable_lora=True,
            max_lora_rank=spec.rank,
            max_loras=cfg.engine.max_loras,
            max_cpu_loras=cfg.engine.max_cpu_loras,
            enable_prefix_caching=cfg.engine.enable_prefix_caching,
        )
        self.tokenizer = self.llm.get_tokenizer()
        self._lora_counter = 0
        self._tmp = Path(cfg.engine.adapter_dir) / cfg.run_name / "eval"
        self.prefix_ids = self.tokenizer.encode(prefix, add_special_tokens=False) \
            if prefix else []

    def score(self, genome: Genome | None, streams: list[str]) -> float:
        """Mean NLL per stream token. genome=None scores the frozen base."""
        from vllm import SamplingParams
        from vllm.lora.request import LoRARequest
        from .engines.vllm_engine import write_peft_adapter

        lora_request = None
        if genome is not None:
            self._lora_counter += 1
            path = self._tmp / f"g{self._lora_counter}"
            write_peft_adapter(genome, self.cfg.model.name, path)
            lora_request = LoRARequest(
                lora_name=f"eval-g{self._lora_counter}",
                lora_int_id=self._lora_counter, lora_path=str(path))

        prompts = []
        boundaries = []
        for text in streams:
            stream_ids = self.tokenizer.encode(text, add_special_tokens=False)
            ids = list(self.prefix_ids) + stream_ids
            prompts.append({"prompt_token_ids": ids})
            boundaries.append(len(self.prefix_ids))

        params = SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=0)
        outputs = self.llm.generate(prompts, params, lora_request=lora_request,
                                    use_tqdm=False)
        nlls, counts = 0.0, 0
        for out, boundary in zip(outputs, boundaries):
            token_ids = out.prompt_token_ids
            # prompt_logprobs[i] is the logprob of token i given tokens < i;
            # entry 0 is None. Score only the stream suffix.
            for i in range(max(boundary, 1), len(token_ids)):
                entry = out.prompt_logprobs[i]
                logprob = entry[token_ids[i]].logprob
                nlls -= logprob
                counts += 1
        return nlls / max(counts, 1)


def evaluate_snapshots(cfg: Config, spec: GenomeSpec, snapshot_dirs: list[Path],
                       streams: list[str], n_controls: int, seed: int,
                       prefix: str = "") -> dict:
    scorer = SurpriseScorer(cfg, spec, prefix=prefix)
    rng = np.random.default_rng(seed)

    results: dict = {"snapshots": [], "controls": {}}

    base_nll = scorer.score(None, streams)
    results["controls"]["base_model"] = base_nll

    control_nlls = []
    for _ in range(n_controls):
        control = Genome.random(spec, cfg.genome.init_scale, rng)
        control_nlls.append(scorer.score(control, streams))
    results["controls"]["random_init"] = {
        "n": n_controls,
        "mean": float(np.mean(control_nlls)) if control_nlls else None,
        "std": float(np.std(control_nlls)) if control_nlls else None,
        "all": control_nlls,
    }

    for snap_dir in snapshot_dirs:
        pairs = snapshot_genomes(snap_dir, spec)
        per_agent = []
        for meta, genome in pairs:
            nll = scorer.score(genome, streams)
            per_agent.append({"id": meta["id"], "generation": meta["generation"],
                              "nll": nll})
        by_gen: dict[int, list[float]] = {}
        for r in per_agent:
            by_gen.setdefault(r["generation"], []).append(r["nll"])
        results["snapshots"].append({
            "path": str(snap_dir),
            "population_mean": float(np.mean([r["nll"] for r in per_agent]))
            if per_agent else None,
            "by_generation": {g: float(np.mean(v)) for g, v in sorted(by_gen.items())},
            "agents": per_agent,
        })
    return results
