"""Engine backends.

The controller is engine-agnostic: it needs a tokenizer, per-agent adapter
registration, and a way to drive one action turn as a stream of tokens. The
mock backend implements this with scripted policies for tests and dry runs;
the vLLM backend implements it with multi-LoRA serving (§4.1).
"""

from .base import EngineBackend, TurnEnded, TurnHandle, TurnToken

__all__ = ["EngineBackend", "TurnHandle", "TurnToken", "TurnEnded"]
