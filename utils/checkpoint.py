"""Normalize checkpoint keys and initialize memory experts."""

import hashlib
from pathlib import Path

import torch


DUAL_LORA_FORMAT = "sgf_plus_peft_dual_lora_v1"


def normalize_state_dict(state):
    return {k.replace("_fsdp_wrapped_module.", "").replace("_checkpoint_wrapped_module.", "").replace("_orig_mod.", ""): v
            for k, v in state.items()}


def initialize_memory_expert(state, parameter_names):
    """Copy generation weights when the memory expert is absent."""
    pairs = {}
    for name in parameter_names:
        for part in ("patch_embedding", "text_embedding", "time_embedding",
                     "time_projection", "head", "modulation", "weight", "bias"):
            marker = ".memory_" + part
            if marker in name:
                pairs[name] = name.replace(marker, "." + part, 1)
                break
    if not any(name in state for name in pairs):
        for memory, generation in pairs.items():
            state[memory] = state[generation]
    return state


def is_lora_parameter(name):
    return ".lora_A." in name or ".lora_B." in name


def dual_lora_state_dict(state):
    """Extract both named adapters from an ordinary or gathered FSDP state."""
    return {n: p for n, p in normalize_state_dict(state).items() if is_lora_parameter(n)}


def _model_keys(model):
    return set(normalize_state_dict(dict(model.named_parameters()))) | set(
        normalize_state_dict(dict(model.named_buffers())))


def _load_partial_state(model, state, allowed_missing):
    expected = _model_keys(model) - allowed_missing
    if set(state) != expected:
        raise RuntimeError("Invalid dual LoRA checkpoint: "
                           f"missing={sorted(expected - set(state))}, "
                           f"unexpected={sorted(set(state) - expected)}")
    result = model.load_state_dict(state, strict=False)
    if set(normalize_state_dict(dict.fromkeys(result.missing_keys))) != allowed_missing or result.unexpected_keys:
        raise RuntimeError(f"Invalid dual LoRA load: {result}")


def load_dual_lora_state_dict(model, state):
    """Load a single-expert base or a complete PEFT base-plus-two-adapter state.

    Translate old Linear weight/bias keys to PEFT base_layer keys. A published
    dual-full checkpoint is not a rank-256 LoRA initialization and is rejected.
    """
    keys = _model_keys(model)
    adapters = {n for n in keys if is_lora_parameter(n)}
    state = normalize_state_dict(state)
    if any(is_lora_parameter(n) for n in state):
        _load_partial_state(model, state, set())
    else:
        base_key_map = {n.replace(".base_layer.", "."): n for n in keys - adapters}
        state = {base_key_map.get(n, n): p for n, p in state.items()}
        _load_partial_state(model, state, adapters)


def load_dual_lora_adapters(model, state):
    """Reject a missing role, missing tensor, or accidentally included base."""
    state = normalize_state_dict(state)
    base_keys = {n for n in _model_keys(model) if not is_lora_parameter(n)}
    _load_partial_state(model, state, base_keys)


def _file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _generator_state(checkpoint, use_ema=False):
    if use_ema:
        if "generator_ema" not in checkpoint:
            raise RuntimeError("Checkpoint has no generator_ema")
        return checkpoint["generator_ema"]
    for key in ("generator", "model", "generator_ema"):
        if key in checkpoint:
            return checkpoint[key]
    return checkpoint


def load_dual_lora_checkpoint(model, path, use_ema=False, base_checkpoint=None):
    """Restore adapters and their exact frozen TF base; return save metadata.

    base_checkpoint is an explicit path override for moving the example to a
    different machine. The recorded digest must still match. This restores
    generator weights only, not optimizer/critic state or the training step.
    """
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    config = getattr(model, "model", model).dual_lora_config
    metadata = checkpoint.get("dual_lora")
    if metadata is None:
        state = _generator_state(checkpoint, use_ema)
        if any(is_lora_parameter(n) for n in state):
            raise RuntimeError("Frozen base must be a single-expert TF checkpoint")
        load_dual_lora_state_dict(model, state)
        return {"format": DUAL_LORA_FORMAT, **config,
                "base": {"path": str(path), "sha256": _file_sha256(path)}}

    if metadata.get("format") != DUAL_LORA_FORMAT or any(metadata.get(k) != v for k, v in config.items()):
        raise RuntimeError("Dual LoRA checkpoint configuration does not match the model")
    base = metadata["base"]
    base_path = Path(base_checkpoint or base["path"])
    if _file_sha256(base_path) != base["sha256"]:
        raise RuntimeError("Frozen base checkpoint SHA256 does not match")
    base_state = torch.load(base_path, map_location="cpu", weights_only=True)
    # A compact checkpoint cannot recursively serve as the frozen base.
    if "dual_lora" in base_state:
        raise RuntimeError("Frozen base must be a single-expert TF checkpoint")
    load_dual_lora_state_dict(model, _generator_state(base_state))
    del base_state
    load_dual_lora_adapters(model, _generator_state(checkpoint, use_ema))
    return {**metadata, "base": {**base, "path": str(base_path)}}
