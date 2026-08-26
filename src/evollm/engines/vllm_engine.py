"""vLLM multi-LoRA serving backend (§4).

One AsyncLLM engine per room/GPU: a single frozen base resident on the
device, many adapters alongside it, requests tagged by adapter. Adapters are
registered by writing peft-format directories to tmpfs (§4.1 path 1).

Division of authority (§4.2/§4.3): the room controller owns the block
economy; the engine is given strictly more KV memory than the controller's
authoritative pool (safety_margin_blocks), so the engine can never legitimately
run out. Engine-side preemption is therefore an infrastructure artefact, and a
stat-logger watchdog turns it into an ExperimentIntegrityError instead of
letting it silently absorb what should have been a death (§4.3). Chunked
prefill spreads observation/prompt prefills across engine steps (§4.4); the
world clock itself is token-metered by the controller and exact regardless.

Imports of vllm/transformers/torch stay inside methods so the module is
importable (for tests of everything else) on machines without the GPU stack.
"""

from __future__ import annotations

import asyncio
import itertools
import json
import os
import shutil
from collections import deque
from pathlib import Path

import numpy as np

from ..config import Config, RoomConfig
from ..events import ExperimentIntegrityError
from ..genome import Genome, GenomeSpec
from .base import EngineBackend, TurnEnded, TurnHandle, TurnToken

# peft module paths for the projections the genome supports.
_MODULE_PARENT = {
    "q_proj": "self_attn", "k_proj": "self_attn",
    "v_proj": "self_attn", "o_proj": "self_attn",
    "gate_proj": "mlp", "up_proj": "mlp", "down_proj": "mlp",
}


def kv_block_bytes(hf_config, block_size: int, dtype_bytes: int = 2) -> int:
    """Bytes of KV cache per block: the exchange rate between adapter bytes
    and blocks in the unified pool (§2.2)."""
    n_kv = getattr(hf_config, "num_key_value_heads",
                   hf_config.num_attention_heads)
    head_dim = getattr(hf_config, "head_dim", None) or \
        hf_config.hidden_size // hf_config.num_attention_heads
    return block_size * 2 * n_kv * head_dim * hf_config.num_hidden_layers * dtype_bytes


def write_peft_adapter(genome: Genome, model_name: str, path: Path) -> None:
    """Serialize a genome as a peft LoRA directory vLLM can load."""
    import torch
    from safetensors.torch import save_file

    path.mkdir(parents=True, exist_ok=True)
    target_modules = sorted({s.projection for s in genome.spec.sites})
    config = {
        "peft_type": "LORA",
        "base_model_name_or_path": model_name,
        "r": genome.spec.rank,
        "lora_alpha": genome.spec.alpha,
        "lora_dropout": 0.0,
        "target_modules": target_modules,
        "bias": "none",
        "fan_in_fan_out": False,
        "modules_to_save": None,
        "task_type": "CAUSAL_LM",
    }
    with open(path / "adapter_config.json", "w") as f:
        json.dump(config, f)
    tensors = {}
    for site in genome.spec.sites:
        a, b = genome.factors[site.key]
        prefix = (f"base_model.model.model.layers.{site.layer}."
                  f"{_MODULE_PARENT[site.projection]}.{site.projection}")
        tensors[f"{prefix}.lora_A.weight"] = torch.from_numpy(a).to(torch.float16)
        tensors[f"{prefix}.lora_B.weight"] = torch.from_numpy(b).to(torch.float16)
    save_file(tensors, str(path / "adapter_model.safetensors"))


class _PreemptionWatchdog:
    """StatLogger that counts engine preemptions.

    A preemption is a PERFORMANCE event, not a corruption. vLLM's
    `_preempt_request` frees the request's KV blocks and resets
    `num_computed_tokens` to 0, but leaves `_output_token_ids` untouched: the
    request returns to the waiting queue, recomputes its prefix, and carries on
    from exactly where it was. No token is lost, none is emitted twice, and no
    agent dies of it — deaths are decided by this project's BlockPool, never by
    the engine. What a preemption does mean is that the room's capacity claim
    is close enough to the engine's real pool for the scheduler to run out of
    working room, which is worth knowing and worth logging.

    Instances are handed to a per-engine sink rather than a class-level list.
    Four rooms share one process, and a shared counter meant a single
    preemption in one room aborted all four.
    """

    def __init__(self, vllm_config, engine_index: int = 0):
        self.preempted = 0

    def record(self, scheduler_stats=None, iteration_stats=None,
               *args, **kwargs):
        if iteration_stats is not None:
            self.preempted += getattr(iteration_stats, "num_preempted_reqs", 0)

    def log_engine_initialized(self):
        pass

    def log(self):
        pass

    def record_sleep_state(self, *args, **kwargs):
        pass


class VLLMTurnHandle(TurnHandle):
    def __init__(self, engine, request_id: str, generator, turn_end_id: int):
        self._engine = engine
        self.request_id = request_id
        self._gen = generator
        self._turn_end_id = turn_end_id
        self._buffer: deque[int] = deque()
        self._finish_reason: str | None = None

    async def next_event(self) -> TurnToken | TurnEnded:
        while not self._buffer:
            if self._finish_reason is not None:
                return TurnEnded(natural=self._finish_reason == "stop")
            try:
                out = await anext(self._gen)
            except StopAsyncIteration:
                return TurnEnded(natural=self._finish_reason == "stop")
            completion = out.outputs[0]
            self._buffer.extend(completion.token_ids)
            if out.finished:
                self._finish_reason = completion.finish_reason or "stop"
        tok = self._buffer.popleft()
        if tok == self._turn_end_id:
            # The stop token itself: the controller charges and appends it.
            return TurnEnded(natural=True)
        return TurnToken(tok)

    async def abort(self) -> None:
        try:
            await self._engine.abort(self.request_id)
        finally:
            await self._gen.aclose()


class VLLMEngine(EngineBackend):
    def __init__(self, cfg: Config, room: RoomConfig, spec: GenomeSpec):
        self.cfg = cfg
        self.room = room
        self.spec = spec
        self.engine = None
        self.tokenizer = None
        self.turn_end_id = -1
        self._lora_ids = itertools.count(1)  # LoRARequest ids must be > 0
        self._active: dict[str, tuple[int, Path]] = {}  # agent -> (lora_id, dir)
        self._turn_counter = itertools.count()
        self._adapter_root = Path(cfg.engine.adapter_dir) / cfg.run_name / room.id
        self._num_gpu_blocks: int | None = None
        self._watchdogs: list[_PreemptionWatchdog] = []
        self._preempted_seen = 0

    async def start(self) -> None:
        # Checked before the heavy imports so a config error costs nothing.
        # This reached vLLM as the literal string "auto" once, surfacing as an
        # opaque TypeError deep in engine construction. Every entry point that
        # builds an engine must call cli.prepare() first.
        if not isinstance(self.cfg.model.max_model_len, int):
            raise ValueError(
                f"model.max_model_len is {self.cfg.model.max_model_len!r}, not "
                "resolved to an int — call evollm.cli.prepare(cfg) before "
                "constructing an engine")

        from transformers import AutoTokenizer
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.usage.usage_lib import UsageContext
        from vllm.v1.engine.async_llm import AsyncLLM

        if self.room.gpu is not None:
            # The engine-core subprocess inherits the environment at spawn
            # time, so setting this before construction pins the room to its
            # device. Rooms are constructed sequentially (see cli.build_world).
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.room.gpu)

        self._adapter_root.mkdir(parents=True, exist_ok=True)
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model.name)
        self.turn_end_id = self._resolve_turn_end_id()

        ecfg = self.cfg.engine
        args = AsyncEngineArgs(
            model=self.cfg.model.name,
            dtype=self.cfg.model.dtype,
            max_model_len=self.cfg.model.max_model_len,
            hf_overrides=self._hf_overrides() or None,
            gpu_memory_utilization=ecfg.gpu_memory_utilization,
            block_size=self.cfg.world.block_size,
            enable_lora=True,
            max_loras=ecfg.max_loras,
            max_lora_rank=self.spec.rank,
            max_cpu_loras=ecfg.max_cpu_loras,
            max_num_seqs=ecfg.max_num_seqs,
            enable_prefix_caching=ecfg.enable_prefix_caching,
            enable_chunked_prefill=ecfg.enable_chunked_prefill,
            # No CPU swap: nothing recoverable is supposed to happen when
            # memory runs out (§4.3).
            swap_space=0,
            disable_log_stats=False,
        )
        sink = self._watchdogs

        class _BoundWatchdog(_PreemptionWatchdog):
            def __init__(self, vllm_config, engine_index: int = 0):
                super().__init__(vllm_config, engine_index)
                sink.append(self)

        self.engine = AsyncLLM.from_engine_args(
            args,
            usage_context=UsageContext.ENGINE_CONTEXT,
            stat_loggers=[_BoundWatchdog],
        )
        cache_cfg = self.engine.vllm_config.cache_config
        self._num_gpu_blocks = cache_cfg.num_gpu_blocks
        claimed = self.room.capacity_blocks
        if claimed:
            head = self._num_gpu_blocks - claimed
            print(f"[{self.room.id}] engine pool {self._num_gpu_blocks:,} blocks; "
                  f"economy claims {claimed:,}; headroom {head:,} "
                  f"({head / self._num_gpu_blocks * 100:.0f}%)")

    def _hf_overrides(self) -> dict:
        """Let agents live past the base model's trained context window.

        The rope cos/sin cache is built to config.max_position_embeddings
        (rotary_embedding/base.py: torch.arange(max_position_embeddings)), and
        lookup is an unchecked index_select. Raising max_model_len alone —
        e.g. via VLLM_ALLOW_LONG_MAX_MODEL_LEN — leaves the cache short and
        turns an over-long context into an out-of-bounds CUDA read. Overriding
        max_position_embeddings grows the cache with it, so positions beyond
        the trained window degrade quality (which selection is free to act on)
        instead of corrupting memory.
        """
        from transformers import AutoConfig

        overrides: dict = {}
        model_cfg = self.cfg.model
        if model_cfg.rope_scaling:
            overrides["rope_scaling"] = model_cfg.rope_scaling
        if not model_cfg.extend_context:
            return overrides
        hf_config = AutoConfig.from_pretrained(model_cfg.name)
        native = getattr(hf_config, "max_position_embeddings", None)
        if native is not None and model_cfg.max_model_len > native:
            overrides["max_position_embeddings"] = model_cfg.max_model_len
            head_dim = getattr(hf_config, "head_dim", None) or \
                hf_config.hidden_size // hf_config.num_attention_heads
            cache_gib = model_cfg.max_model_len * head_dim * 4 / 1024**3
            print(f"[{self.room.id}] extending context {native} -> "
                  f"{model_cfg.max_model_len} tokens "
                  f"({model_cfg.max_model_len / native:.1f}x the trained "
                  f"window; rope cache ~{cache_gib:.2f} GiB). Behaviour past "
                  f"{native} is out of distribution by design.")
        return overrides

    def _resolve_turn_end_id(self) -> int:
        for token in ("<|im_end|>",):
            tid = self.tokenizer.convert_tokens_to_ids(token)
            if tid is not None and tid >= 0:
                return tid
        return self.tokenizer.eos_token_id

    async def stop(self) -> None:
        if self.engine is not None:
            self.engine.shutdown()
        shutil.rmtree(self._adapter_root, ignore_errors=True)

    # ── capacity (§4.2) ───────────────────────────────────────────────────
    def pool_blocks(self) -> int | None:
        return self._num_gpu_blocks

    def device_memory(self) -> dict | None:
        """Actual bytes in use on this room's GPU, from NVML.

        The block economy accounts for KV and adapters; the device also carries
        the model weights, the CUDA context, cuBLAS workspaces, captured CUDA
        graphs, activation buffers and fragmentation. None of that is in the
        ledger, which is why capacity has to be set against a measured number
        rather than a computed one.

        NVML is used rather than torch.cuda because it needs no CUDA context in
        this process: the engine runs in its own subprocess, and initialising a
        context here purely to ask a question would itself consume a few
        hundred MB of what we are trying to measure.
        """
        if self.room.gpu is None:
            return None
        try:
            import pynvml
            if not getattr(self, "_nvml_ready", False):
                pynvml.nvmlInit()
                self._nvml_ready = True
            h = pynvml.nvmlDeviceGetHandleByIndex(self.room.gpu)
            info = pynvml.nvmlDeviceGetMemoryInfo(h)
            return {"used_mb": int(info.used) // 2**20,
                    "free_mb": int(info.free) // 2**20,
                    "total_mb": int(info.total) // 2**20}
        except Exception:
            return None

    def capacity_blocks(self) -> int | None:
        """Authoritative pool size for the room.

        The engine's measured KV pool is only an upper bound. On a 96 GB GH200
        a 1.5B model yields ~2.9M tokens of KV, which is hundreds of times the
        model's context window — so an agent would hit max_model_len long
        before the room could ever starve, and death by scarcity would be
        unreachable. Capacity is therefore also clamped so that even a lone
        agent holding the entire pool stays under the context ceiling; World
        re-checks this invariant for every room (§4.3).
        """
        if self._num_gpu_blocks is None:
            return None
        capacity = self._num_gpu_blocks - self.cfg.engine.safety_margin_blocks
        if capacity <= 0:
            raise ValueError(
                f"room {self.room.id}: engine pool of {self._num_gpu_blocks} "
                "blocks leaves no capacity after the safety margin")
        return capacity

    # ── tokenizer ─────────────────────────────────────────────────────────
    def block_prefix(self, role: str, first: bool = False) -> list[int]:
        if not self.cfg.world.chat_format:
            return []
        # "<|im_end|>\n<|im_start|>{role}\n" is the trained boundary; the
        # turn-end token itself is appended by the controller, so the newline
        # that follows it opens the next block here.
        lead = "" if first else "\n"
        return self.tokenize(f"{lead}<|im_start|>{role}\n")

    def tokenize(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def detokenize(self, ids: list[int]) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=False)

    # ── adapters (§4.1) ───────────────────────────────────────────────────
    async def register_adapter(self, agent_id: str, genome: Genome) -> None:
        from vllm.lora.request import LoRARequest

        lora_id = next(self._lora_ids)
        path = self._adapter_root / agent_id
        write_peft_adapter(genome, self.cfg.model.name, path)
        self._active[agent_id] = (lora_id, path)
        ok = await self.engine.add_lora(
            LoRARequest(lora_name=agent_id, lora_int_id=lora_id,
                        lora_path=str(path)))
        if not ok:
            raise ExperimentIntegrityError(
                f"engine refused adapter for {agent_id}")

    async def unregister_adapter(self, agent_id: str) -> None:
        entry = self._active.pop(agent_id, None)
        if entry is None:
            return
        lora_id, path = entry
        try:
            await self.engine.remove_lora(lora_id)
        finally:
            shutil.rmtree(path, ignore_errors=True)

    # ── turns ─────────────────────────────────────────────────────────────
    def start_turn(self, agent_id: str, context: list[int],
                   max_tokens: int) -> TurnHandle:
        from vllm import SamplingParams
        from vllm.lora.request import LoRARequest
        from vllm.sampling_params import RequestOutputKind

        lora_id, path = self._active[agent_id]
        request_id = f"{agent_id}-t{next(self._turn_counter)}"
        params = SamplingParams(
            temperature=self.cfg.sampling.temperature,
            top_p=self.cfg.sampling.top_p,
            max_tokens=max_tokens,
            stop_token_ids=[self.turn_end_id],
            output_kind=RequestOutputKind.DELTA,
        )
        generator = self.engine.generate(
            {"prompt_token_ids": list(context)},
            params,
            request_id,
            lora_request=LoRARequest(lora_name=agent_id, lora_int_id=lora_id,
                                     lora_path=str(path)),
        )
        return VLLMTurnHandle(self.engine, request_id, generator, self.turn_end_id)

    # ── integrity (§4.3) ──────────────────────────────────────────────────
    def poll_preemptions(self) -> int:
        """Preemptions by THIS engine since the last poll."""
        total = sum(w.preempted for w in self._watchdogs)
        new = total - self._preempted_seen
        self._preempted_seen = total
        return new
