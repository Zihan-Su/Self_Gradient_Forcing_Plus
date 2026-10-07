import torch.nn.functional as F
from typing import Tuple
import torch
import random
from model.base import BaseModel
from utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper
from utils.scheduler import FlowMatchScheduler
from pipeline import CausalDiffusionInferencePipeline
class NaiveConsistency(BaseModel):
    def __init__(self, args, device):
        super().__init__(args, device)
        print(args)
        # Step 1: Initialize all models
        self.generator = WanDiffusionWrapper(**getattr(args, "model_kwargs", {}), is_causal=args.is_causal)
        self.generator.model.requires_grad_(True)
        
        
        self.generator_ema = WanDiffusionWrapper(**getattr(args, "model_kwargs", {}), is_causal=args.is_causal)
        self.generator_ema.model.requires_grad_(False)
        
        self.teacher = WanDiffusionWrapper(**getattr(args, "model_kwargs", {}), is_causal=True)
        self.teacher.model.requires_grad_(False)
        
        self.num_frame_per_block = getattr(args, "num_frame_per_block", 1)
        if self.num_frame_per_block > 1:
            self.generator.model.num_frame_per_block = self.num_frame_per_block
            self.generator_ema.model.num_frame_per_block = self.num_frame_per_block
            self.teacher.model.num_frame_per_block = self.num_frame_per_block
            
            
        if getattr(args, "generator_ckpt", False):
            print(f"Loading pretrained generator from {args.generator_ckpt}")
            state_dict = torch.load(args.generator_ckpt, map_location="cpu")[
                'generator']
            self.generator.load_state_dict(
                state_dict, strict=True
            )
            
            self.teacher.load_state_dict(
                state_dict, strict=True
            )
            
            self.generator_ema.load_state_dict(
                state_dict, strict=True
            )
                         
        self.independent_first_frame = getattr(args, "independent_first_frame", False)
        if self.independent_first_frame:
            self.generator.model.independent_first_frame = True
        if args.gradient_checkpointing:
            self.generator.enable_gradient_checkpointing()

        # Step 2: Initialize all hyperparameters
        self.timestep_shift = getattr(args, "timestep_shift", 1.0)
        self.guidance_scale = args.guidance_scale
        
        self.discrete_cd_N = getattr(args, "discrete_cd_N", 48)
        self.scheduler = FlowMatchScheduler(shift=5.0, sigma_min=0.0, extra_one_step=True)
        self.scheduler.set_timesteps(num_inference_steps=self.discrete_cd_N, denoising_strength=1.0)
        self.scheduler.sigmas = self.scheduler.sigmas.to(device)
        
        self.pipeline = CausalDiffusionInferencePipeline(args, device=device, need_vae=False)
        self.pipeline.generator = self.teacher
        self.pipeline.text_encoder = self.text_encoder
        
    def _initialize_models(self, args, device):
        self.generator = WanDiffusionWrapper(**getattr(args, "model_kwargs", {}), is_causal=True)
        self.generator.model.requires_grad_(True)

        self.teacher = WanDiffusionWrapper(**getattr(args, "model_kwargs", {}), is_causal=True)
        self.teacher.model.requires_grad_(False)

        self.generator_ema = WanDiffusionWrapper(**getattr(args, "model_kwargs", {}), is_causal=args.is_causal)
        self.generator_ema.model.requires_grad_(False)
        
        self.text_encoder = WanTextEncoder()
        self.text_encoder.requires_grad_(False)

        
        self.scheduler = self.generator.get_scheduler()
        self.scheduler.timesteps = self.scheduler.timesteps.to(device)

        
        
    def generator_loss(
            self, 
            conditional_dict,
            unconditional_dict,
            clean_latent,
            ema_model
        ) -> Tuple[torch.Tensor, dict]:
        
        clean_latent = clean_latent.to(self.device).to(torch.bfloat16)
        B, num_frames = clean_latent.shape[:2]
        timestep_idx = random.randrange(self.discrete_cd_N - 1)

        t = self.scheduler.timesteps[timestep_idx]
        timestep = t * torch.ones([B, num_frames], device=self.device, dtype=torch.bfloat16)
        t_next = self.scheduler.timesteps[timestep_idx + 1]
        timestep_next = t_next * torch.ones([B, num_frames], device=self.device, dtype=torch.bfloat16)
        
        noise = torch.randn_like(clean_latent)
        latent_t = self.scheduler.add_noise(
            clean_latent, noise=noise,
            timestep=t * torch.ones([1], device=self.device)
        ).to(torch.bfloat16)

        # Full-frame teacher forward (replaces per-frame loop)
        with torch.no_grad():
            v_cond, _ = self.teacher(
                latent_t, conditional_dict, timestep, clean_x=clean_latent)
            v_uncond, _ = self.teacher(
                latent_t, unconditional_dict, timestep, clean_x=clean_latent)
            v_pred = v_uncond + self.guidance_scale * (
                v_cond - v_uncond)
            dt = (timestep - timestep_next).reshape(B, num_frames, 1, 1, 1)
            dt /= 1000
            latent_t_next = latent_t - dt * v_pred

        # Share block_mask to avoid redundant allocation
        if self.generator.model.block_mask is None and self.teacher.model.block_mask is not None:
            self.generator.model.block_mask = self.teacher.model.block_mask
            self.generator_ema.model.block_mask = self.teacher.model.block_mask

        
        print(f't:{t}; t_next: {t_next}')
        
        _, cm_pred_t = self.generator(
            latent_t, conditional_dict, timestep, clean_x = clean_latent
        )

        with torch.no_grad():
            ema_model.copy_to(self.generator_ema)
            _, cm_pred_t_next = self.generator_ema(
                latent_t_next, conditional_dict, timestep_next, clean_x = clean_latent
            )
            
        with torch.enable_grad():
            loss = F.mse_loss(cm_pred_t, cm_pred_t_next, reduction="mean")

        log_dict = {
            "unnormalized_loss": F.mse_loss(cm_pred_t, cm_pred_t_next, reduction='none').mean(dim=[1, 2, 3, 4]).detach(),
        }

        return loss, log_dict
