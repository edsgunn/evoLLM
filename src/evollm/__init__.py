"""evoLLM: gradient-free evolution as a substrate for in-context surprise minimisation.

Agents are LoRA adapters over a shared frozen base model. They act, observe,
reproduce and die inside a block economy whose currency is device memory.
No weights change within a lifetime; the only thing that changes across
generations is the distribution of initial adapter weights.
"""

__version__ = "0.1.0"
