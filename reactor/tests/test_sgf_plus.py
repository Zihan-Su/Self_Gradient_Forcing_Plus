"""Exercise application hooks and model lifetime without torch or weights."""

import asyncio
import subprocess
import sys
from contextlib import nullcontext
from dataclasses import replace
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from PIL import Image
from reactor_runtime import ApplicationError, CommandError, StepOutcome, UploadedFile
from reactor_runtime.schema import render

from sgf_plus import SGFPlus
from sgf_plus_backend import Run, SGFBackend
from sgf_plus_images import fit_image
from sgf_plus_model import NotSeeded, RolloutComplete, SGFInput, SGFModel, SGFResult
from sgf_plus_types import SGFState


def app():
    model = SGFPlus()
    model.state = SGFState()
    model._engine = Mock()
    messages = []

    async def send(message):
        messages.append(message)

    model.send = send
    return model, messages


def upload():
    data = BytesIO()
    Image.new("RGB", (64, 64), "green").save(data, format="PNG")
    return UploadedFile(name="image.png", mime_type="image/png", data=data.getvalue())


def result(world=1, index=1, complete=False):
    return SGFResult(
        frames=np.zeros((2, 480, 832, 3), dtype=np.uint8),
        world_id=world,
        chunk_index=index,
        complete=complete,
    )


def test_refused_steps_never_reach_model():
    model, _ = app()
    for prompt, paused, complete in [
        ("", False, False),
        ("test", True, False),
        ("test", False, True),
    ]:
        model.state._prompt, model.state.paused, model.state._complete = (
            prompt,
            paused,
            complete,
        )
        with pytest.raises(ApplicationError):
            asyncio.run(model.process_input())
    model._engine.generate.assert_not_called()


def test_generate_reads_only_passed_input():
    model, _ = app()
    request = SGFInput(world_id=7, prompt="old", seed=0, image_conditioned=False)
    model.state = None
    model.generate(request)
    model._engine.generate.assert_called_once_with(request)


def test_anchor_repeated_until_acknowledged_then_omitted():
    model, messages = app()
    asyncio.run(model.set_image(upload(), "A forest", 42))
    first = asyncio.run(model.process_input())
    assert first.image.shape == (480, 832, 3)
    assert asyncio.run(model.process_input()).image is first.image
    asyncio.run(model.process_output(StepOutcome(result=result(), elapsed=0.25)))
    assert asyncio.run(model.process_input()).image is None
    assert model.state._chunks == 1 and messages[-1].last_chunk_seconds == 0.25
    asyncio.run(model.reset(7))
    assert model.state._chunks == 0 and model.state._world_id == 2
    assert asyncio.run(model.process_input()).image is not None


def test_failure_is_not_completion_or_acknowledgement():
    model, _ = app()
    asyncio.run(model.start("Test", 0))
    failure = RuntimeError("native failure")
    model._engine.generate.side_effect = failure
    with pytest.raises(RuntimeError):
        model.generate(asyncio.run(model.process_input()))
    with pytest.raises(RuntimeError, match="native failure"):
        asyncio.run(model.process_output(StepOutcome(error=failure, elapsed=1)))
    assert model.state._chunks == 0 and model.state._applied_world_id is None


def test_prompt_update_keeps_world_history_reference_seed_and_pause():
    model, messages = app()
    model.output = Mock()
    asyncio.run(model.set_image(upload(), "Forest", 3))
    image = model.state._image
    asyncio.run(model.process_output(StepOutcome(result=result(index=4), elapsed=0.5)))
    asyncio.run(model.set_paused(True))
    model.output.flush.reset_mock()
    reply = asyncio.run(model.set_prompt("Snow falls"))
    assert model.state._image is image and model.state._world_id == 1
    assert model.state._seed == 3 and model.state._chunks == 4
    assert model.state._applied_world_id == 1 and model.state._last_seconds == 0.5
    assert reply.paused and reply.prompt == "Snow falls" and messages[-1] == reply
    model.output.flush.assert_not_called()
    model._engine.generate.assert_not_called()
    model._engine.reset.assert_not_called()
    with pytest.raises(ApplicationError):
        asyncio.run(model.process_input())
    asyncio.run(model.set_paused(False))
    request = asyncio.run(model.process_input())
    assert request.prompt == "Snow falls" and request.image is None
    asyncio.run(model.process_output(StepOutcome(result=result(index=5), elapsed=0.6)))
    assert model.state._chunks == 5
    asyncio.run(model.start("A desert", 4))
    assert (
        model.state._image is None
        and not asyncio.run(model.process_input()).image_conditioned
    )


def test_prompt_refusal_does_not_mutate_or_restart_the_video():
    model, _ = app()
    asyncio.run(model.start("Forest", 3))
    with pytest.raises(CommandError):
        asyncio.run(model.set_prompt(" \n "))
    assert model.state._prompt == "Forest" and model.state._world_id == 1
    model.state._complete = True
    with pytest.raises(CommandError):
        asyncio.run(model.set_prompt("Rain"))
    assert model.state._complete and model.state._prompt == "Forest"


def test_prompt_change_and_failed_step_do_not_acknowledge_completion():
    model, _ = app()
    asyncio.run(model.start("Forest", 3))
    asyncio.run(model.process_output(StepOutcome(result=result(index=3), elapsed=0.5)))
    asyncio.run(model.set_prompt("Rain"))
    with pytest.raises(RuntimeError, match="text encoder failed"):
        asyncio.run(
            model.process_output(
                StepOutcome(error=RuntimeError("text encoder failed"), elapsed=1)
            )
        )
    assert model.state._world_id == 1 and model.state._chunks == 3


def test_pause_resume_and_completion():
    model, _ = app()
    asyncio.run(model.start("Test", 0))
    asyncio.run(model.set_paused(True))
    with pytest.raises(ApplicationError):
        asyncio.run(model.process_input())
    asyncio.run(model.set_paused(False))
    assert asyncio.run(model.process_input()).world_id == 1
    asyncio.run(
        model.process_output(StepOutcome(result=result(complete=True), elapsed=1))
    )
    with pytest.raises(ApplicationError):
        asyncio.run(model.process_input())


def test_session_state_does_not_leak_and_end_resets_model():
    model, _ = app()
    asyncio.run(model.start("Test", 987))
    model.on_session_ended()
    model._engine.reset.assert_called_once_with()
    model.state = SGFState()
    assert (
        model.state._seed == 0
        and model.state._world_id == 0
        and model.state._prompt == ""
    )
    # No per-viewer disconnect hook resets the shared model.
    assert "on_disconnected" not in SGFPlus.__dict__


@pytest.mark.parametrize(
    "command,args", [("start", (" ", 0)), ("set_prompt", ("Test",)), ("reset", (0,))]
)
def test_command_refusals(command, args):
    model, _ = app()
    with pytest.raises(CommandError):
        asyncio.run(getattr(model, command)(*args))
    assert model.state._world_id == 0


def test_bad_uploads_do_not_change_state(monkeypatch):
    model, _ = app()
    with pytest.raises(CommandError):
        asyncio.run(
            model.set_image(
                UploadedFile(name="bad.png", mime_type="image/png", data=b"bad"),
                "Test",
                0,
            )
        )
    with pytest.raises(ValueError):
        fit_image(b"x" * (25 * 1024 * 1024 + 1))
    monkeypatch.setattr(
        Image, "open", Mock(side_effect=Image.DecompressionBombError("large"))
    )
    with pytest.raises(CommandError):
        asyncio.run(
            model.set_image(
                UploadedFile(name="large.png", mime_type="image/png", data=b"x"),
                "Test",
                0,
            )
        )
    assert model.state._world_id == 0


def test_model_half_owns_world_and_reset():
    model = SGFModel()
    backend = Mock()
    model._backend = backend
    backend.step.return_value = (result().frames, 1, False)
    request = SGFInput(world_id=1, prompt="Test", seed=0, image_conditioned=True)
    with pytest.raises(NotSeeded):
        model.generate(request)
    backend.start.assert_not_called()
    request = replace(request, image=np.zeros((480, 832, 3), dtype=np.uint8))
    assert model.generate(request).world_id == 1
    model.generate(replace(request, image=None))
    assert backend.start.call_count == 1
    backend.update_prompt.assert_called_once_with("Test")
    backend.step.return_value = (result().frames, 3, False)
    updated = model.generate(replace(request, prompt="Rain", image=None))
    backend.update_prompt.assert_called_with("Rain")
    assert backend.start.call_count == 1
    assert updated.world_id == 1 and updated.chunk_index == 3
    backend.step.side_effect = RolloutComplete()
    with pytest.raises(RolloutComplete):
        model.generate(request)
    model.reset()
    with pytest.raises(NotSeeded):
        model.generate(replace(request, image=None))
    backend.reset.assert_called_once_with()


def fake_text_backend(monkeypatch):
    # No Torch import is needed: the operation under test only stages text and
    # invalidates native cache flags. The GPU forward is tested separately.
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(no_grad=nullcontext))
    backend = SGFBackend.__new__(SGFBackend)
    caches = [
        {
            "is_init": True,
            "generation_expert": {"is_init": True, "k": object(), "v": object()},
            "memory_expert": {"is_init": True, "k": object(), "v": object()},
        }
        for _ in range(2)
    ]
    backend.pipeline = SimpleNamespace(
        crossattn_cache=caches,
        kv_cache1=object(),
        vae=Mock(),
        text_encoder=Mock(return_value={"prompt_embeds": object()}),
    )
    backend.run = Run(
        conditioning={"old": object()},
        prompt="Forest",
        noise=object(),
        position=15,
        input_frames=3,
        prefix=None,
        index=4,
    )
    return backend


def test_backend_refreshes_both_text_experts_and_keeps_temporal_state(monkeypatch):
    backend = fake_text_backend(monkeypatch)
    run = backend.run
    noise, kv = run.noise, backend.pipeline.kv_cache1
    old_k = [block["memory_expert"]["k"] for block in backend.pipeline.crossattn_cache]
    backend.update_prompt("Rain")
    backend.pipeline.text_encoder.assert_called_once_with(text_prompts=["Rain"])
    assert (
        backend.run is run and run.noise is noise and backend.pipeline.kv_cache1 is kv
    )
    assert (run.position, run.index, run.input_frames, run.prefix) == (15, 4, 3, None)
    assert (
        run.prompt == "Rain"
        and run.conditioning is backend.pipeline.text_encoder.return_value
    )
    for index, block in enumerate(backend.pipeline.crossattn_cache):
        assert not block["is_init"]
        assert not block["generation_expert"]["is_init"]
        assert not block["memory_expert"]["is_init"]
        assert block["memory_expert"]["k"] is old_k[index]
    backend.pipeline.vae.model.clear_cache.assert_not_called()
    backend.update_prompt("Rain")
    assert backend.pipeline.text_encoder.call_count == 1


def test_unchanged_prompt_is_a_noop_and_encoding_failure_keeps_old_text(monkeypatch):
    backend = fake_text_backend(monkeypatch)
    conditioning = backend.run.conditioning
    backend.update_prompt("Forest")
    backend.pipeline.text_encoder.assert_not_called()
    backend.pipeline.text_encoder.side_effect = RuntimeError("text encoder failed")
    with pytest.raises(RuntimeError, match="text encoder failed"):
        backend.update_prompt("Rain")
    assert backend.run.prompt == "Forest" and backend.run.conditioning is conditioning
    for block in backend.pipeline.crossattn_cache:
        assert (
            block["is_init"]
            and block["generation_expert"]["is_init"]
            and block["memory_expert"]["is_init"]
        )


def test_model_modules_import_without_runtime_or_torch():
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
class Block:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'reactor_runtime', 'torch'}:
            raise ImportError('Forbidden dependency: ' + fullname)
sys.meta_path.insert(0, Block())
import sgf_plus_model
import sgf_plus_backend
""",
        ],
        check=True,
        cwd=Path(__file__).resolve().parents[1],
    )


def test_schema_moderation_and_real_command_surface():
    schema = render(Path(__file__).resolve().parents[1])
    assert set(schema["paths"]) == {
        "/events/start",
        "/events/set_image",
        "/events/set_prompt",
        "/events/reset",
        "/events/set_paused",
    }
    for name in ("start", "set_image", "set_prompt"):
        props = schema["paths"][f"/events/{name}"]["post"]["requestBody"]["content"][
            "application/json"
        ]["schema"]["properties"]
        assert props["prompt"]["x-reactor-moderate"]
        if name == "set_image":
            assert props["image"]["x-reactor-moderate"]
    assert SGFPlus.__dict__["fps"] == 16
