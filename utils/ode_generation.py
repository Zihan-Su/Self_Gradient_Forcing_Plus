from typing import Dict, Iterable, Optional

import torch


def merge_cfg_prompt_embeds(
    conditional_dict: dict,
    unconditional_dict: dict,
) -> dict:
    cond = conditional_dict["prompt_embeds"]
    uncond = unconditional_dict["prompt_embeds"]

    if isinstance(cond, torch.Tensor):
        prompt_embeds = torch.cat([cond, uncond], dim=0)
    else:
        prompt_embeds = list(cond) + list(uncond)

    return {"prompt_embeds": prompt_embeds}


def normalize_trajectory_indices(
    trajectory_indices: Iterable[int],
    num_inference_steps: int,
) -> list[int]:
    total = num_inference_steps + 2
    normalized = []
    for idx in trajectory_indices:
        norm_idx = idx if idx >= 0 else total + idx
        if norm_idx < 0 or norm_idx >= total:
            raise IndexError(
                f"trajectory index {idx} is out of range for a trajectory of length {total}"
            )
        normalized.append(norm_idx)
    return normalized


class CausalODETrajectoryGenerator:
    def __init__(
        self,
        model,
        scheduler,
        num_frame_per_block: int,
        num_inference_steps: int,
        guidance_scale: float,
    ) -> None:
        self.model = model
        self.scheduler = scheduler
        self.num_frame_per_block = num_frame_per_block
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.frame_seq_length = 1560
        self.num_transformer_blocks = len(self.model.model.blocks)
        self.local_attn_size = self.model.model.local_attn_size

    def _make_kv_cache(self, batch_size: int, device: torch.device) -> list[dict]:
        if self.local_attn_size != -1:
            kv_cache_size = self.local_attn_size * self.frame_seq_length
        else:
            kv_cache_size = 32760

        kv_cache = []
        for _ in range(self.num_transformer_blocks):
            kv_cache.append(
                {
                    "k": torch.zeros(
                        [batch_size, kv_cache_size, 12, 128],
                        dtype=torch.float32,
                        device=device,
                    ),
                    "v": torch.zeros(
                        [batch_size, kv_cache_size, 12, 128],
                        dtype=torch.float32,
                        device=device,
                    ),
                    "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                    "local_end_index": torch.tensor([0], dtype=torch.long, device=device),
                }
            )
        return kv_cache

    def _make_crossattn_cache(self, batch_size: int, device: torch.device) -> list[dict]:
        crossattn_cache = []
        for _ in range(self.num_transformer_blocks):
            crossattn_cache.append(
                {
                    "k": torch.zeros(
                        [batch_size, 512, 12, 128],
                        dtype=torch.float32,
                        device=device,
                    ),
                    "v": torch.zeros(
                        [batch_size, 512, 12, 128],
                        dtype=torch.float32,
                        device=device,
                    ),
                    "is_init": False,
                }
            )
        return crossattn_cache

    def _batched_cfg_step(
        self,
        latents: torch.Tensor,
        paired_conditional_dict: dict,
        timestep: torch.Tensor,
        clean_x: Optional[torch.Tensor] = None,
        kv_cache: Optional[list[dict]] = None,
        crossattn_cache: Optional[list[dict]] = None,
        current_start: Optional[int] = None,
    ) -> torch.Tensor:
        latents_pair = latents.repeat(2, 1, 1, 1, 1)
        timestep_pair = timestep.repeat(2, 1)

        clean_pair = None
        if clean_x is not None:
            clean_pair = clean_x.repeat(2, 1, 1, 1, 1)

        flow_pair, _ = self.model(
            latents_pair,
            paired_conditional_dict,
            timestep_pair,
            kv_cache=kv_cache,
            crossattn_cache=crossattn_cache,
            current_start=current_start,
            clean_x=clean_pair,
        )

        flow_cond = flow_pair[:1].float()
        flow_uncond = flow_pair[1:2].float()
        return flow_uncond + self.guidance_scale * (flow_cond - flow_uncond)

    def _update_clean_cache(
        self,
        clean_x: torch.Tensor,
        paired_conditional_dict: dict,
        kv_cache: list[dict],
        crossattn_cache: list[dict],
        current_start: int,
    ) -> None:
        timestep = torch.full(
            [1, clean_x.shape[1]],
            0.0,
            device=clean_x.device,
            dtype=torch.float32,
        )
        with torch.no_grad():
            self._batched_cfg_step(
                latents=clean_x,
                paired_conditional_dict=paired_conditional_dict,
                timestep=timestep,
                kv_cache=kv_cache,
                crossattn_cache=crossattn_cache,
                current_start=current_start,
            )

    def _generate_full(
        self,
        clean_latent: torch.Tensor,
        paired_conditional_dict: dict,
        normalized_indices: list[int],
        initial_noise: torch.Tensor,
    ) -> torch.Tensor:
        latents = initial_noise.clone()
        selected_steps = {idx for idx in normalized_indices if idx < self.num_inference_steps}
        step_snapshots: Dict[int, torch.Tensor] = {}

        frame_count = latents.shape[1]
        for step_idx, t in enumerate(self.scheduler.timesteps):
            if step_idx in selected_steps:
                step_snapshots[step_idx] = latents.clone()

            timestep = t * torch.ones(
                [1, frame_count],
                device=latents.device,
                dtype=torch.float32,
            )
            flow_pred = self._batched_cfg_step(
                latents=latents,
                paired_conditional_dict=paired_conditional_dict,
                timestep=timestep,
                clean_x=clean_latent,
            )
            latents = self.scheduler.step(
                flow_pred.flatten(0, 1),
                timestep.flatten(0, 1),
                latents.flatten(0, 1),
            ).unflatten(dim=0, sizes=flow_pred.shape[:2])

        return self._assemble_selected_trajectory(
            clean_latent=clean_latent,
            final_latent=latents,
            normalized_indices=normalized_indices,
            step_snapshots=step_snapshots,
        )

    def _generate_blockwise_kv(
        self,
        clean_latent: torch.Tensor,
        paired_conditional_dict: dict,
        normalized_indices: list[int],
        initial_noise: torch.Tensor,
    ) -> torch.Tensor:
        num_frames = clean_latent.shape[1]
        if num_frames % self.num_frame_per_block != 0:
            raise ValueError(
                f"num_frames={num_frames} must be divisible by num_frame_per_block={self.num_frame_per_block}"
            )

        kv_cache = self._make_kv_cache(batch_size=2, device=clean_latent.device)
        crossattn_cache = self._make_crossattn_cache(batch_size=2, device=clean_latent.device)

        selected_steps = {idx for idx in normalized_indices if idx < self.num_inference_steps}
        step_snapshots = {
            idx: torch.empty_like(clean_latent)
            for idx in selected_steps
        }
        final_latent = torch.empty_like(clean_latent)

        num_blocks = num_frames // self.num_frame_per_block
        for block_idx in range(num_blocks):
            start = block_idx * self.num_frame_per_block
            end = start + self.num_frame_per_block
            current_start = start * self.frame_seq_length

            block_clean = clean_latent[:, start:end].contiguous()
            block_latents = initial_noise[:, start:end].clone()

            for step_idx, t in enumerate(self.scheduler.timesteps):
                if step_idx in selected_steps:
                    step_snapshots[step_idx][:, start:end] = block_latents

                timestep = t * torch.ones(
                    [1, block_latents.shape[1]],
                    device=block_latents.device,
                    dtype=torch.float32,
                )
                flow_pred = self._batched_cfg_step(
                    latents=block_latents,
                    paired_conditional_dict=paired_conditional_dict,
                    timestep=timestep,
                    kv_cache=kv_cache,
                    crossattn_cache=crossattn_cache,
                    current_start=current_start,
                )
                block_latents = self.scheduler.step(
                    flow_pred.flatten(0, 1),
                    timestep.flatten(0, 1),
                    block_latents.flatten(0, 1),
                ).unflatten(dim=0, sizes=flow_pred.shape[:2])

            final_latent[:, start:end] = block_latents
            self._update_clean_cache(
                clean_x=block_clean,
                paired_conditional_dict=paired_conditional_dict,
                kv_cache=kv_cache,
                crossattn_cache=crossattn_cache,
                current_start=current_start,
            )

        return self._assemble_selected_trajectory(
            clean_latent=clean_latent,
            final_latent=final_latent,
            normalized_indices=normalized_indices,
            step_snapshots=step_snapshots,
        )

    def _assemble_selected_trajectory(
        self,
        clean_latent: torch.Tensor,
        final_latent: torch.Tensor,
        normalized_indices: list[int],
        step_snapshots: Dict[int, torch.Tensor],
    ) -> torch.Tensor:
        selected = []
        final_index = self.num_inference_steps
        clean_index = self.num_inference_steps + 1

        for idx in normalized_indices:
            if idx < self.num_inference_steps:
                selected.append(step_snapshots[idx])
            elif idx == final_index:
                selected.append(final_latent)
            elif idx == clean_index:
                selected.append(clean_latent)
            else:
                raise RuntimeError(f"Unexpected normalized trajectory index: {idx}")

        return torch.stack(selected, dim=1)

    def generate(
        self,
        clean_latent: torch.Tensor,
        paired_conditional_dict: dict,
        trajectory_indices: Iterable[int],
        generation_mode: str = "blockwise_kv",
        initial_noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if generation_mode not in {"full", "blockwise_kv"}:
            raise ValueError(f"Unsupported generation_mode: {generation_mode}")

        normalized_indices = normalize_trajectory_indices(
            trajectory_indices=trajectory_indices,
            num_inference_steps=self.num_inference_steps,
        )

        if initial_noise is None:
            initial_noise = torch.randn_like(clean_latent, dtype=torch.float32)
        else:
            initial_noise = initial_noise.to(
                device=clean_latent.device,
                dtype=torch.float32,
            )

        with torch.no_grad():
            if generation_mode == "full":
                return self._generate_full(
                    clean_latent=clean_latent,
                    paired_conditional_dict=paired_conditional_dict,
                    normalized_indices=normalized_indices,
                    initial_noise=initial_noise,
                )
            return self._generate_blockwise_kv(
                clean_latent=clean_latent,
                paired_conditional_dict=paired_conditional_dict,
                normalized_indices=normalized_indices,
                initial_noise=initial_noise,
            )
