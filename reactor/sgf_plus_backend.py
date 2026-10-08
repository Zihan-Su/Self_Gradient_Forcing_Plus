"""Incremental adaptation of SGF+'s released causal_inference.py (Apache-2.0).

The spatial sampler, expert routing and streaming attention call upstream code.
The temporal loop is split into steps and the upstream causal VAE retains its
decode cache between steps. Source: Zihan-Su/Self_Gradient_Forcing_Plus.
"""

from __future__ import annotations

import sys
from contextlib import chdir
from dataclasses import dataclass

import numpy as np

from sgf_plus_model import RolloutComplete, Settings


@dataclass
class Run:
    """Worker-local tensors and native temporal position for a single rollout."""

    conditioning: dict
    prompt: str
    noise: object
    position: int
    input_frames: int
    prefix: np.ndarray | None
    index: int = 0


class SGFBackend:
    """Keep both SGF+ experts, T5, and VAE resident on one GPU."""

    def __init__(self, settings: Settings, device: str) -> None:
        import torch
        from omegaconf import OmegaConf

        # Upstream imports flat packages and uses relative asset paths. This
        # backend runs only in the isolated model worker, never the HTTP process.
        sys.path.insert(0, str(settings.source))
        from pipeline.causal_inference import CausalInferencePipeline

        config = OmegaConf.merge(
            OmegaConf.load(settings.source / "configs/default_config.yaml"),
            OmegaConf.load(settings.source / "configs/sgf_plus_chunkwise.yaml"),
        )
        self.device = device
        self.settings = settings
        self.block = int(config.num_frame_per_block)
        self.window, self.sink = 12, 3
        if (
            settings.output_latents < 2 * self.block
            or settings.output_latents % self.block
        ):
            raise ValueError(
                "output_latents must contain at least two complete native blocks"
            )
        with chdir(settings.weights):
            self.pipeline = CausalInferencePipeline(config, device=device)
        checkpoint = torch.load(
            settings.weights / "hf_weights/chunkwise/model.pt",
            map_location="cpu",
            weights_only=True,
        )["generator_ema"]
        checkpoint = {
            key.replace("model._fsdp_wrapped_module.", "model.", 1): value
            for key, value in checkpoint.items()
        }
        self.pipeline.generator.load_state_dict(checkpoint, strict=True)
        self.pipeline.to(dtype=torch.bfloat16, device=device).eval().requires_grad_(
            False
        )
        self.run: Run | None = None

    def reset(self) -> None:
        """Forget all per-world tensors, including both experts' text caches."""
        self.run = None
        self.pipeline.kv_cache1 = None
        self.pipeline.crossattn_cache = None
        self.pipeline.vae.model.clear_cache()

    def start(self, prompt: str, seed: int, image: np.ndarray | None) -> None:
        """Prepare the native prompt, conditioning chunk and complete noise draw."""
        import torch

        with torch.no_grad():
            self.reset()
            pipe = self.pipeline
            initial = None
            if image is not None:
                pixels = torch.from_numpy(image.copy()).to(self.device).permute(2, 0, 1)
                # Match torchvision ToTensor/Normalize before the BF16 cast.
                pixels = (pixels.float() / 255 - 0.5) / 0.5
                pixels = pixels[None, :, None].to(torch.bfloat16)
                pixels = pixels.repeat(1, 1, 1 + 4 * (self.block - 1), 1, 1)
                initial = pipe.vae.encode_to_latent(pixels).to(torch.bfloat16)
            input_frames = self.block if initial is not None else 0
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            # Native inference samples the ENTIRE initial noise before any
            # denoising noise. Per-chunk randn would change the same-seed video.
            noise = torch.randn(
                [1, self.settings.output_latents - input_frames, 16, 60, 104],
                device=self.device,
                dtype=torch.bfloat16,
            )
            pipe._configure_streaming(
                True, self.window, self.sink, self.window, "top_aligned"
            )
            conditioning = pipe.text_encoder(text_prompts=[prompt])
            pipe._initialize_kv_cache(1, noise.dtype, noise.device)
            pipe._initialize_crossattn_cache(1, noise.dtype, noise.device)
            prefix = None
            if initial is not None:
                pipe.generator(
                    noisy_image_or_video=initial,
                    conditional_dict=conditioning,
                    timestep=torch.zeros([1, 1], device=self.device, dtype=torch.int64),
                    kv_cache=pipe.kv_cache1,
                    crossattn_cache=pipe.crossattn_cache,
                    current_start=0,
                    use_memory_expert=True,
                )
                prefix = self._decode(initial)
            self.run = Run(
                conditioning=conditioning,
                prompt=prompt,
                noise=noise,
                position=input_frames,
                input_frames=input_frames,
                prefix=prefix,
            )

    def update_prompt(self, prompt: str) -> None:
        """Refresh both experts' text K/V without rewriting video history.

        This live control extends the released single-prompt inference loop.
        Past self-attention K/V, VAE context, noise and temporal position stay
        intact. The next denoising and memory-write calls refill their own text
        caches through upstream's native ``is_init`` protocol.
        """
        import torch

        run, pipe = self.run, self.pipeline
        assert run is not None, "Start a video before updating its prompt"
        if prompt == run.prompt:
            return
        with torch.no_grad():
            conditioning = pipe.text_encoder(text_prompts=[prompt])
        for block in pipe.crossattn_cache:
            block["is_init"] = False
            block["generation_expert"]["is_init"] = False
            block["memory_expert"]["is_init"] = False
        run.conditioning = conditioning
        run.prompt = prompt

    def step(self) -> tuple[np.ndarray, int, bool]:
        """Four native denoising steps, memory-expert commit, then causal decode."""
        import torch

        run, pipe = self.run, self.pipeline
        if run.position >= self.settings.output_latents:
            raise RolloutComplete("Start a new rollout to continue")
        with torch.no_grad():
            offset = run.position - run.input_frames
            noisy = run.noise[:, offset : offset + self.block]
            schedule = (
                pipe.denoising_step_list_first_chunk
                if run.index == 0 and pipe.denoising_step_list_first_chunk is not None
                else pipe.denoising_step_list
            )
            for index, value in enumerate(schedule):
                timestep = (
                    torch.ones([1, self.block], device=self.device, dtype=torch.int64)
                    * value
                )
                _, prediction = pipe.generator(
                    noisy_image_or_video=noisy,
                    conditional_dict=run.conditioning,
                    timestep=timestep,
                    kv_cache=pipe.kv_cache1,
                    crossattn_cache=pipe.crossattn_cache,
                    current_start=run.position * pipe.frame_seq_length,
                    use_memory_expert=False,
                )
                if index + 1 < len(schedule):
                    noisy = pipe.scheduler.add_noise(
                        prediction.flatten(0, 1),
                        torch.randn_like(prediction.flatten(0, 1)),
                        schedule[index + 1]
                        * torch.ones(
                            [self.block], device=self.device, dtype=torch.long
                        ),
                    ).unflatten(0, prediction.shape[:2])
            pipe.generator(
                noisy_image_or_video=prediction,
                conditional_dict=run.conditioning,
                timestep=torch.ones_like(timestep) * pipe.args.context_noise,
                kv_cache=pipe.kv_cache1,
                crossattn_cache=pipe.crossattn_cache,
                current_start=run.position * pipe.frame_seq_length,
                use_memory_expert=True,
            )
            frames = self._decode(prediction)
            if run.prefix is not None:
                frames = np.concatenate((run.prefix, frames))
                run.prefix = None
            run.position += self.block
            run.index += 1
            return frames, run.index, run.position >= self.settings.output_latents

    def _decode(self, latents) -> np.ndarray:
        """Emit only new RGB frames using the upstream VAE's retained context."""
        import torch

        video = self.pipeline.vae.decode_to_pixel(latents, use_cache=True)
        pixels = (video[0] * 0.5 + 0.5).clamp(0, 1)
        return (pixels.permute(0, 2, 3, 1) * 255).to(torch.uint8).cpu().numpy().copy()
