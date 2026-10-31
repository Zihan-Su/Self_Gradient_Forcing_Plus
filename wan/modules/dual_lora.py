"""PEFT LoRA parameters with explicit, checkpoint-safe token-role routing."""

import math

import torch
from torch import nn

try:
    from peft import LoraConfig
    from peft.tuners.lora.layer import Linear as PeftLinear
except ImportError:
    # The original single/dual-full paths do not require the optional dependency.
    LoraConfig = None
    PeftLinear = nn.Module


class DualLoRALinear(PeftLinear):
    """Route a clean prefix to Context Writer and remaining tokens to Denoiser.

    Use PEFT's native A/B modules, initialization and state-dict names. Select
    adapters per token, never by mutating the active adapter during forward.
    """

    def __init__(self, linear, rank=256, alpha=256):
        if LoraConfig is None:
            raise ImportError("Dual LoRA requires peft==0.19.1; install the repository requirements.txt")
        if not isinstance(rank, int) or isinstance(rank, bool) or rank <= 0:
            raise ValueError("LoRA rank must be a positive integer")
        if not math.isfinite(alpha) or alpha <= 0:
            raise ValueError("LoRA alpha must be finite and positive")
        config = LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=0.0,
                            bias="none", init_lora_weights=True)
        linear.requires_grad_(False)
        super().__init__(linear, "denoiser", config=config, r=rank, lora_alpha=alpha)
        self.update_layer("context_writer", r=rank, lora_alpha=alpha, config=config)
        # PEFT freezes inactive adapters. Both sets must stay trainable, but our
        # forward below selects exactly one set for each token, not their sum.
        self.set_adapter(["denoiser", "context_writer"])

    def _adapter(self, x, adapter):
        x = self._cast_input_dtype(x, self.lora_A[adapter].weight.dtype)
        return self.lora_B[adapter](self.lora_A[adapter](self.lora_dropout[adapter](x))) * self.scaling[adapter]

    def merge(self, *args, **kwargs):
        raise RuntimeError("Do not merge role-specific LoRAs into the shared base")

    def forward(self, x, memory_token_count=0):
        if not 0 <= memory_token_count <= x.shape[1]:
            raise ValueError(
                f"memory_token_count={memory_token_count} is incompatible with input {x.shape}"
            )
        if self.disable_adapters:
            raise RuntimeError("Dual LoRA requires both role-specific adapters")
        base = self.base_layer(x)
        if memory_token_count == 0:
            return base + self._adapter(x, "denoiser").to(base.dtype)
        if memory_token_count == x.shape[1]:
            return base + self._adapter(x, "context_writer").to(base.dtype)
        residual = torch.cat([
            self._adapter(x[:, :memory_token_count], "context_writer"),
            self._adapter(x[:, memory_token_count:], "denoiser"),
        ], dim=1)
        return base + residual.to(base.dtype)
