"""Runtime-independent step contract and lifetime of an SGF+ world."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from sgf_plus_backend import SGFBackend


@dataclass(frozen=True)
class Settings:
    """Resolved assets and native rollout length passed to one GPU worker."""

    source: Path
    weights: Path
    output_latents: int = 963


@dataclass(frozen=True)
class SGFInput:
    """One step of a world; all arrays crossing this boundary are on the CPU."""

    world_id: int
    """Application-assigned identity, changed only for a fresh rollout."""
    prompt: str
    """Text for this chunk; changes retain the current world's video history."""
    seed: int
    """Native random seed used when starting the world."""
    image_conditioned: bool
    """Whether a fresh world requires an anchor, distinct from text-only mode."""
    image: np.ndarray | None = None
    """RGB uint8 [480,832,3], supplied until the world ID is acknowledged."""


@dataclass(frozen=True)
class SGFResult:
    """Facts produced by one completed chunk."""

    frames: np.ndarray
    """Contiguous RGB uint8 [F,480,832,3] containing only new output frames."""
    world_id: int
    """Identity actually applied by the model."""
    chunk_index: int
    """One-based number of successfully generated chunks in this world."""
    complete: bool
    """Whether the native configured rollout length has been reached."""


class NotSeeded(RuntimeError):
    """An image-conditioned world arrived without its reference."""


class RolloutComplete(RuntimeError):
    """The configured native video length has been generated."""


class SGFModel:
    """Own one world behind the world-ID/anchor acknowledgment contract.

    A changed ID starts at chunk 1. Missing image conditioning raises NotSeeded;
    stepping beyond the configured video length raises RolloutComplete.
    """

    device: str = "cuda:0"

    def __init__(self) -> None:
        self._backend: SGFBackend | None = None
        self._world_id: int | None = None

    def load(self, settings: Settings) -> None:
        """Load resolved assets on the device assigned by the worker runner."""
        from sgf_plus_backend import SGFBackend

        self._backend = SGFBackend(settings, self.device)

    def generate(self, input: SGFInput) -> SGFResult:
        """Apply a fresh identity if necessary, then produce one native chunk."""
        assert self._backend is not None, "Load the model before generating"
        if input.world_id != self._world_id:
            if input.image_conditioned and input.image is None:
                raise NotSeeded("A fresh image-conditioned world needs its anchor")
            self._backend.start(input.prompt, input.seed, input.image)
            self._world_id = input.world_id
        else:
            self._backend.update_prompt(input.prompt)
        frames, index, complete = self._backend.step()
        return SGFResult(
            frames=frames, world_id=self._world_id, chunk_index=index, complete=complete
        )

    def reset(self) -> None:
        """Release world caches while retaining the loaded weights."""
        assert self._backend is not None, "Load the model before resetting"
        self._backend.reset()
        self._world_id = None
