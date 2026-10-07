from typing import List, Optional

import torch
import torch.distributed as dist

from utils.scheduler import SchedulerInterface
from utils.wan_wrapper import WanDiffusionWrapper


class SelfGradientForcingTrainingPipeline:
    """Training-time rollout for Self Gradient Forcing.

    Self Gradient Forcing performs a no-grad autoregressive cache construction
    followed by one teacher-forced gradient pass at the sampled training timestep.
    The cache target can be the final clean prediction or the sampled exit prediction.
    """

    def __init__(
        self,
        denoising_step_list: List[int],
        scheduler: SchedulerInterface,
        generator: WanDiffusionWrapper,
        num_frame_per_block: int = 3,
        independent_first_frame: bool = False,
        same_step_across_blocks: bool = False,
        last_step_only: bool = False,
        num_max_frames: int = 21,
        context_noise: int = 0,
        denoising_step_list_first_chunk: Optional[List[int]] = None,
        grad_mode: str = "self_gradient_forcing",
        per_rank_exit_step: bool = True,
        self_gradient_forcing_match_context: bool = True,
        self_gradient_forcing_cache_mode: str = "exit",
        **kwargs,
    ):
        super().__init__()
        if grad_mode != "self_gradient_forcing":
            raise ValueError(
                "This release only supports grad_mode='self_gradient_forcing'. "
                f"Got {grad_mode!r}."
            )

        self.scheduler = scheduler
        self.generator = generator
        self.denoising_step_list = denoising_step_list
        if self.denoising_step_list[-1] == 0:
            self.denoising_step_list = self.denoising_step_list[:-1]

        self.denoising_step_list_first_chunk = denoising_step_list_first_chunk
        if self.denoising_step_list_first_chunk is not None and self.denoising_step_list_first_chunk[-1] == 0:
            self.denoising_step_list_first_chunk = self.denoising_step_list_first_chunk[:-1]

        self.num_transformer_blocks = 30
        self.frame_seq_length = 1560
        self.num_frame_per_block = num_frame_per_block
        self.context_noise = context_noise
        self.i2v = False

        self.kv_cache1 = None
        self.crossattn_cache = None
        self.independent_first_frame = independent_first_frame
        self.same_step_across_blocks = same_step_across_blocks
        self.last_step_only = last_step_only
        self.kv_cache_size = num_max_frames * self.frame_seq_length
        self.per_rank_exit_step = per_rank_exit_step
        self.self_gradient_forcing_match_context = self_gradient_forcing_match_context
        if self_gradient_forcing_cache_mode not in ("exit", "final_clean"):
            raise ValueError(
                "self_gradient_forcing_cache_mode must be 'exit' or 'final_clean'. "
                f"Got {self_gradient_forcing_cache_mode!r}."
            )
        self.self_gradient_forcing_cache_mode = self_gradient_forcing_cache_mode

    def generate_and_sync_list(self, num_blocks, num_denoising_steps, device):
        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
            if rank == 0:
                indices = torch.randint(
                    low=0,
                    high=num_denoising_steps,
                    size=(num_blocks,),
                    device=device,
                )
                if self.last_step_only:
                    indices = torch.ones_like(indices) * (num_denoising_steps - 1)
            else:
                indices = torch.empty(num_blocks, dtype=torch.long, device=device)
            dist.broadcast(indices, src=0)
            return indices.tolist()

        indices = torch.randint(
            low=0,
            high=num_denoising_steps,
            size=(num_blocks,),
            device=device,
        )
        if self.last_step_only:
            indices = torch.ones_like(indices) * (num_denoising_steps - 1)
        return indices.tolist()

    def inference_with_trajectory(
        self,
        noise: torch.Tensor,
        clean_image_or_video: torch.Tensor = None,
        initial_latent: Optional[torch.Tensor] = None,
        return_sim_step: bool = False,
        **conditional_dict,
    ) -> torch.Tensor:
        return self._self_gradient_forcing_inference(
            noise=noise,
            clean_image_or_video=clean_image_or_video,
            initial_latent=initial_latent,
            return_sim_step=return_sim_step,
            **conditional_dict,
        )

    def _self_gradient_forcing_inference(
        self,
        noise: torch.Tensor,
        clean_image_or_video: torch.Tensor = None,
        initial_latent: Optional[torch.Tensor] = None,
        return_sim_step: bool = False,
        **conditional_dict,
    ) -> torch.Tensor:
        """Run Self Gradient Forcing for one training sample.

        Pass 1 constructs the autoregressive KV cache without gradients and
        records the noisy input and x0 prediction at the sampled training timestep.
        Pass 2 performs one parallel teacher-forced forward with gradients so
        cross-frame attention parameters receive the recovered signal.
        """
        assert initial_latent is None, "Self Gradient Forcing training assumes text-to-video inputs."
        batch_size, num_frames, num_channels, height, width = noise.shape
        assert num_frames % self.num_frame_per_block == 0
        num_blocks = num_frames // self.num_frame_per_block
        device, dtype = noise.device, noise.dtype

        self._initialize_kv_cache(batch_size=batch_size, dtype=dtype, device=device)
        self._initialize_crossattn_cache(batch_size=batch_size, dtype=dtype, device=device)

        num_denoising_steps = len(self.denoising_step_list)
        if self.self_gradient_forcing_cache_mode == "exit" and self.per_rank_exit_step:
            idx = int(torch.randint(low=0, high=num_denoising_steps, size=(1,), device=device).item())
        else:
            idx = int(self.generate_and_sync_list(1, num_denoising_steps, device=device)[0])
        train_t = self.denoising_step_list[idx]

        x_hat = torch.zeros_like(noise)
        x_ctx_hat = torch.zeros_like(noise)
        noisy_at_t = torch.zeros_like(noise)

        all_num_frames = [self.num_frame_per_block] * num_blocks
        current_start_frame = 0
        with torch.no_grad():
            for current_num_frames in all_num_frames:
                noisy_input = noise[:, current_start_frame:current_start_frame + current_num_frames]
                x0_exit = None
                timestep = None

                for index, current_timestep in enumerate(self.denoising_step_list):
                    timestep = torch.ones(
                        [batch_size, current_num_frames], device=device, dtype=torch.int64
                    ) * current_timestep

                    if index == idx:
                        noisy_at_t[:, current_start_frame:current_start_frame + current_num_frames] = noisy_input

                    _, x0 = self.generator(
                        noisy_image_or_video=noisy_input,
                        conditional_dict=conditional_dict,
                        timestep=timestep,
                        kv_cache=self.kv_cache1,
                        crossattn_cache=self.crossattn_cache,
                        current_start=current_start_frame * self.frame_seq_length,
                        use_memory_expert=False,
                    )

                    if index == idx:
                        x0_exit = x0
                        if self.self_gradient_forcing_cache_mode == "exit" and not self.per_rank_exit_step:
                            break

                    if index < num_denoising_steps - 1:
                        next_timestep = self.denoising_step_list[index + 1]
                        noisy_input = self.scheduler.add_noise(
                            x0.flatten(0, 1),
                            torch.randn_like(x0.flatten(0, 1)),
                            next_timestep * torch.ones(
                                [batch_size * current_num_frames], device=device, dtype=torch.long
                            ),
                        ).unflatten(0, x0.shape[:2])

                if x0_exit is None:
                    raise RuntimeError("Failed to capture the Self Gradient Forcing exit state.")

                x0_cache = x0 if self.self_gradient_forcing_cache_mode == "final_clean" else x0_exit
                x_hat[:, current_start_frame:current_start_frame + current_num_frames] = x0_cache

                context_timestep = torch.ones_like(timestep) * self.context_noise
                x_ctx = self.scheduler.add_noise(
                    x0_cache.flatten(0, 1),
                    torch.randn_like(x0_cache.flatten(0, 1)),
                    context_timestep.flatten(0, 1).long(),
                ).unflatten(0, x0_cache.shape[:2])
                x_ctx_hat[:, current_start_frame:current_start_frame + current_num_frames] = x_ctx

                self.generator(
                    noisy_image_or_video=x_ctx,
                    conditional_dict=conditional_dict,
                    timestep=context_timestep,
                    kv_cache=self.kv_cache1,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length,
                    use_memory_expert=True,
                )
                current_start_frame += current_num_frames

        tf_timestep = train_t * torch.ones([batch_size, num_frames], device=device, dtype=torch.int64)
        if self.self_gradient_forcing_match_context:
            pass2_clean_x = x_ctx_hat.detach()
            aug_t = torch.ones([batch_size, num_frames], device=device, dtype=torch.int64) * self.context_noise
        else:
            pass2_clean_x = x_hat.detach()
            aug_t = torch.zeros([batch_size, num_frames], device=device, dtype=torch.int64)

        _, output = self.generator(
            noisy_image_or_video=noisy_at_t,
            conditional_dict=conditional_dict,
            timestep=tf_timestep,
            clean_x=pass2_clean_x,
            aug_t=aug_t,
        )

        if idx == num_denoising_steps - 1:
            denoised_timestep_to = 0
            denoised_timestep_from = 1000 - torch.argmin(
                (self.scheduler.timesteps.to(device) - self.denoising_step_list[idx].to(device)).abs(), dim=0
            ).item()
        else:
            denoised_timestep_to = 1000 - torch.argmin(
                (self.scheduler.timesteps.to(device) - self.denoising_step_list[idx + 1].to(device)).abs(), dim=0
            ).item()
            denoised_timestep_from = 1000 - torch.argmin(
                (self.scheduler.timesteps.to(device) - self.denoising_step_list[idx].to(device)).abs(), dim=0
            ).item()

        if return_sim_step:
            return output, denoised_timestep_from, denoised_timestep_to, idx + 1
        return output, denoised_timestep_from, denoised_timestep_to

    def _initialize_kv_cache(self, batch_size, dtype, device):
        kv_cache1 = []
        for _ in range(self.num_transformer_blocks):
            kv_cache1.append({
                "k": torch.zeros([batch_size, self.kv_cache_size, 12, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, self.kv_cache_size, 12, 128], dtype=dtype, device=device),
                "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                "local_end_index": torch.tensor([0], dtype=torch.long, device=device),
            })
        self.kv_cache1 = kv_cache1

    def _initialize_crossattn_cache(self, batch_size, dtype, device):
        crossattn_cache = []
        for _ in range(self.num_transformer_blocks):
            crossattn_cache.append({
                "k": torch.zeros([batch_size, 512, 12, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, 512, 12, 128], dtype=dtype, device=device),
                "is_init": False,
            })
        self.crossattn_cache = crossattn_cache
