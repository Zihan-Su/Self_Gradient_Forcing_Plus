"""Client-visible controls, stream and progress for SGF+."""

import numpy as np
from reactor_runtime import (
    InputField,
    InputState,
    MessageField,
    ModelMessage,
    Output,
    Video,
)


class SGFState(InputState):
    """Session controls and private rollout selection."""

    paused: bool = InputField(
        default=False,
        description="Hold generation on the last frame; resume without restarting.",
    )
    _prompt: str = ""
    _seed: int = 0
    _image: np.ndarray | None = None
    _world_id: int = 0
    _applied_world_id: int | None = None
    _chunks: int = 0
    _complete: bool = False
    _last_seconds: float | None = None


class SGFOutput(Output):
    """Silent generated video at the upstream inference export rate of 16 FPS."""

    main_video: Video


class StateUpdate(ModelMessage):
    """Shared selection and progress, broadcast after commands and chunks."""

    prompt: str = MessageField(description="Selected prompt; empty before starting.")
    seed: int = MessageField(description="Seed for the selected rollout.")
    has_image: bool = MessageField(
        description="Whether generation uses an uploaded image."
    )
    paused: bool = MessageField(description="Whether generation is held.")
    world_id: int = MessageField(
        description="Rollout identity, changed when starting or restarting."
    )
    completed_chunks: int = MessageField(
        description="Successfully generated chunks in this rollout."
    )
    complete: bool = MessageField(
        description="The video reached its configured length; start or reset to continue."
    )
    last_chunk_seconds: float | None = MessageField(
        description="Generation time for the last completed chunk, or null before output."
    )
