"""Application commands and Runtime step hooks for SGF+."""

import asyncio
from pathlib import Path

from reactor_runtime import (
    ApplicationError,
    ClientInfo,
    CommandError,
    InputField,
    ReactorApp,
    StepOutcome,
    TrackPayload,
    UploadedFile,
    connected,
    event,
    session_ended,
)
from reactor_runtime.distributed import DistributedRunner

from sgf_plus_assets import prepare
from sgf_plus_images import fit_image
from sgf_plus_model import SGFInput, SGFModel, SGFResult
from sgf_plus_types import SGFOutput, SGFState, StateUpdate


class SGFPlus(ReactorApp):
    """Generate a continuous silent video from text and an optional uploaded image."""

    # Match the upstream inference export rate without speeding up fast chunks.
    fps = 16
    state: SGFState

    def load(self, config_path: Path) -> None:
        settings = prepare(config_path)
        self._engine = DistributedRunner(
            SGFModel,
            world_size=1,
            load_kwargs={"settings": settings},
            call_timeout=300,
        )
        self._engine.start()

    @connected
    async def on_connected(self, client: ClientInfo) -> None:
        await client.send(self._status())

    @session_ended
    def on_session_ended(self) -> None:
        self._engine.reset()
        self.output.flush()

    @event(
        name="start",
        description="Start a fresh text-only video. Requires a non-empty prompt; clears any selected image. Returns and broadcasts state_update.",
    )
    async def start(
        self,
        prompt: str = InputField(
            moderate=True,
            max_length=4096,
            description="Describe the video to generate.",
        ),
        seed: int = InputField(
            default=0, ge=0, le=2147483647, description="Reproducible random seed."
        ),
    ) -> StateUpdate:
        prompt = self._validate_prompt(prompt)
        self.state._image = None
        self.state._prompt, self.state._seed = prompt, seed
        return await self._restart()

    @event(
        name="set_image",
        description="Start a fresh video from an uploaded image and non-empty prompt. Returns and broadcasts state_update; invalid images or empty text return command_error.",
    )
    async def set_image(
        self,
        image: UploadedFile = InputField(
            moderate=True,
            description="PNG, JPEG, WebP or BMP; up to 25 MiB and 40 million pixels. Resized to 832x480 without cropping.",
        ),
        prompt: str = InputField(
            moderate=True,
            max_length=4096,
            description="Describe the motion and scene to generate.",
        ),
        seed: int = InputField(
            default=0, ge=0, le=2147483647, description="Reproducible random seed."
        ),
    ) -> StateUpdate:
        prompt = self._validate_prompt(prompt)
        if image.mime_type not in (
            "image/png",
            "image/jpeg",
            "image/webp",
            "image/bmp",
        ):
            raise CommandError("invalid_image", "Upload PNG, JPEG, WebP or BMP")
        try:
            pixels = await asyncio.to_thread(fit_image, image.data)
        except ValueError as error:
            raise CommandError("invalid_image", str(error)) from error
        self.state._image = pixels
        self.state._prompt, self.state._seed = prompt, seed
        return await self._restart()

    @event(
        name="set_prompt",
        description="Update the prompt for the next generated chunk, retaining video history and pause state. Requires a started, unfinished video; returns and broadcasts state_update. Already buffered frames keep playing.",
    )
    async def set_prompt(
        self,
        prompt: str = InputField(
            moderate=True,
            max_length=4096,
            description="Non-empty description of the scene and motion for subsequent chunks.",
        ),
    ) -> StateUpdate:
        self._require_started()
        if self.state._complete:
            raise CommandError(
                "rollout_complete", "Start or reset before changing the prompt"
            )
        self.state._prompt = self._validate_prompt(prompt)
        status = self._status()
        await self.send(status)
        return status

    @event(
        name="reset",
        description="Restart the selected prompt and optional image with a new seed. Requires a started rollout; returns and broadcasts state_update.",
    )
    async def reset(
        self,
        seed: int = InputField(
            default=0, ge=0, le=2147483647, description="Seed for the restarted video."
        ),
    ) -> StateUpdate:
        self._require_started()
        self.state._seed = seed
        return await self._restart()

    @event(
        name="set_paused",
        description="Pause or resume generation without restarting. Returns and broadcasts state_update.",
    )
    async def set_paused(
        self,
        paused: bool = InputField(
            description="True to hold generation; false to resume."
        ),
    ) -> StateUpdate:
        self.state.paused = paused
        status = self._status()
        await self.send(status)
        return status

    async def process_input(self) -> SGFInput:
        if not self.state._prompt:
            raise ApplicationError("Start with a prompt and optional image")
        if self.state.paused:
            raise ApplicationError("Paused")
        if self.state._complete:
            raise ApplicationError("Video complete; start or reset to continue")
        return SGFInput(
            world_id=self.state._world_id,
            prompt=self.state._prompt,
            seed=self.state._seed,
            image_conditioned=self.state._image is not None,
            image=self.state._image
            if self.state._world_id != self.state._applied_world_id
            else None,
        )

    def generate(self, input: SGFInput) -> SGFResult:
        return self._engine.generate(input)

    async def process_output(self, outcome: StepOutcome) -> SGFOutput | None:
        if outcome.error is not None:
            # Refusals are handled before inference. NotSeeded/RolloutComplete
            # here indicate a contract bug; CUDA/native failures end the session.
            raise outcome.error
        result: SGFResult = outcome.result
        self.state._applied_world_id = result.world_id
        self.state._chunks = result.chunk_index
        self.state._complete = result.complete
        self.state._last_seconds = outcome.elapsed
        await self.send(self._status())
        metadata = [
            {"world_id": result.world_id, "chunk_index": result.chunk_index}
        ] * len(result.frames)
        return SGFOutput(main_video=TrackPayload(result.frames, metadata=metadata))

    def _status(self) -> StateUpdate:
        state = self.state
        return StateUpdate(
            prompt=state._prompt,
            seed=state._seed,
            has_image=state._image is not None,
            paused=state.paused,
            world_id=state._world_id,
            completed_chunks=state._chunks,
            complete=state._complete,
            last_chunk_seconds=state._last_seconds,
        )

    async def _restart(self) -> StateUpdate:
        self.state._world_id += 1
        self.state._chunks = 0
        self.state._complete = False
        self.state._last_seconds = None
        self.state.paused = False
        self.output.flush()
        status = self._status()
        await self.send(status)
        return status

    def _require_started(self) -> None:
        if not self.state._prompt:
            raise CommandError("not_started", "Start a video first")

    @staticmethod
    def _validate_prompt(prompt: str) -> str:
        if not prompt.strip():
            raise CommandError("empty_prompt", "Enter a non-empty prompt")
        return prompt.strip()
