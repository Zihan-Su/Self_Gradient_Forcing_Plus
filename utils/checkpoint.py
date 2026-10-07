"""Normalize checkpoint keys and initialize memory experts."""


def normalize_state_dict(state):
    return {k.replace("_fsdp_wrapped_module.", "").replace("_checkpoint_wrapped_module.", ""): v
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
