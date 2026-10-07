import gc
import logging
from utils.dataset import cycle
from utils.dataset import TextDataset
from utils.distributed import EMA_FSDP, fsdp_wrap, fsdp_state_dict, launch_distributed_job
from utils.misc import set_seed
from utils.checkpoint import normalize_state_dict, initialize_memory_expert
import torch.distributed as dist
from omegaconf import OmegaConf
from model import DMD
import torch
import wandb
import time
import os


class Trainer:
    def __init__(self, config):
        self.config = config
        self.step = 0

        # Step 1: Initialize the distributed training environment (rank, seed, dtype, logging etc.)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        launch_distributed_job()
        global_rank = dist.get_rank()
        self.world_size = dist.get_world_size()

        self.dtype = torch.bfloat16 if config.mixed_precision else torch.float32
        self.device = torch.cuda.current_device()
        self.is_main_process = global_rank == 0
        self.causal = config.causal
        self.disable_wandb = config.disable_wandb

        # use a random seed for the training
        if config.seed == 0:
            random_seed = torch.randint(0, 10000000, (1,), device=self.device)
            dist.broadcast(random_seed, src=0)
            config.seed = random_seed.item()

        set_seed(config.seed + global_rank)

        if self.is_main_process and not self.disable_wandb:
            wandb.login(host=config.wandb_host, key=config.wandb_key)
            wandb.init(
                config=OmegaConf.to_container(config, resolve=True),
                name=config.config_name,
                mode="online",
                entity=config.wandb_entity,
                project=config.wandb_project,
                dir=config.wandb_save_dir
            )

        self.output_path = config.logdir

        # Step 2: Initialize the model and optimizer
        if config.distribution_loss == "dmd":
            self.model = DMD(config, device=self.device)
        else:
            raise ValueError("Invalid distribution matching loss")

        # Save pretrained model state_dicts to CPU
        self.fake_score_state_dict_cpu = self.model.fake_score.state_dict()

        self.model.generator = fsdp_wrap(
            self.model.generator,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.generator_fsdp_wrap_strategy,
            cpu_offload=False,
            sync_module_states=True,
        )

        self.model.real_score = fsdp_wrap(
            self.model.real_score,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.real_score_fsdp_wrap_strategy,
            cpu_offload=False
        )

        self.model.fake_score = fsdp_wrap(
            self.model.fake_score,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.fake_score_fsdp_wrap_strategy,
            cpu_offload=False
        )

        self.model.text_encoder = fsdp_wrap(
            self.model.text_encoder,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.text_encoder_fsdp_wrap_strategy,
            cpu_offload=getattr(config, "text_encoder_cpu_offload", False)
        )

        if not config.no_visualize or config.load_raw_video:
            self.model.vae = self.model.vae.to(
                device=self.device, dtype=torch.bfloat16 if config.mixed_precision else torch.float32)

        self.generator_optimizer = torch.optim.AdamW(
            [param for param in self.model.generator.parameters()
             if param.requires_grad],
            lr=config.lr,
            betas=(config.beta1, config.beta2),
            weight_decay=config.weight_decay
        )

        self.critic_optimizer = torch.optim.AdamW(
            [param for param in self.model.fake_score.parameters()
             if param.requires_grad],
            lr=config.lr_critic if hasattr(config, "lr_critic") else config.lr,
            betas=(config.beta1_critic, config.beta2_critic),
            weight_decay=config.weight_decay
        )

        # Step 3: Initialize the dataloader
        dataset = TextDataset(config.data_path)
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, shuffle=True, drop_last=True)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=config.batch_size,
            sampler=sampler,
            num_workers=8)

        if dist.get_rank() == 0:
            print("DATASET SIZE %d" % len(dataset))
        self.dataloader = cycle(dataloader)

        ##############################################################################################################
        # 6. Set up EMA parameter containers
        rename_param = (
            lambda name: name.replace("_fsdp_wrapped_module.", "")
            .replace("_checkpoint_wrapped_module.", "")
            .replace("_orig_mod.", "")
        )
        self.name_to_trainable_params = {}
        for n, p in self.model.generator.named_parameters():
            if not p.requires_grad:
                continue

            renamed_n = rename_param(n)
            self.name_to_trainable_params[renamed_n] = p
        ema_weight = config.ema_weight
        self.generator_ema = None

        ##############################################################################################################
        # 7. (If resuming) Load the model and optimizer, lr_scheduler, ema's statedicts
        if getattr(config, "generator_ckpt", False):
            print(f"Loading pretrained generator from {config.generator_ckpt}")
            state_dict = torch.load(config.generator_ckpt, map_location="cpu")
            for key in ("generator", "model", "generator_ema"):
                if key in state_dict:
                    state_dict = state_dict[key]
                    break
            state_dict = normalize_state_dict(state_dict)
            state_dict = initialize_memory_expert(state_dict, self.name_to_trainable_params)
            self.model.generator.load_state_dict(state_dict, strict=True)
            del state_dict

        ##############################################################################################################

        # Initialize EMA from the restored weights.
        if self.step >= config.ema_start_step and ema_weight and ema_weight > 0:
            self.generator_ema = EMA_FSDP(self.model.generator, decay=ema_weight)

        self.max_grad_norm_generator = getattr(config, "max_grad_norm_generator", 10.0)
        self.max_grad_norm_critic = getattr(config, "max_grad_norm_critic", 10.0)
        self.previous_time = None
        self.step_time_log_interval = max(
            0, int(getattr(config, "step_time_log_interval", 0) or 0))

    def save(self):
        print("Start gathering distributed model states...")
        generator_state_dict = fsdp_state_dict(
            self.model.generator)
        state_dict = {"generator": generator_state_dict, "step": self.step}
        if self.generator_ema is not None:
            state_dict["generator_ema"] = self.generator_ema.full_state_dict(self.model.generator)
        if getattr(self.config, "save_critic", False):
            state_dict["critic"] = fsdp_state_dict(self.model.fake_score)

        if self.is_main_process:
            os.makedirs(os.path.join(self.output_path,
                        f"checkpoint_model_{self.step:06d}"), exist_ok=True)
            torch.save(state_dict, os.path.join(self.output_path,
                       f"checkpoint_model_{self.step:06d}", "model.pt"))
            print("Model saved to", os.path.join(self.output_path,
                  f"checkpoint_model_{self.step:06d}", "model.pt"))

    def fwdbwd_one_step(self, batch, train_generator, clean_latent=None):
        self.model.eval()  # prevent any randomness (e.g. dropout)

        if self.step % 20 == 0:
            torch.cuda.empty_cache()

        # Step 1: Get the next batch of text prompts
        text_prompts = batch["prompts"]
        if self.config.i2v:
            # clean_latent = None #original code here
            image_latent = batch["ode_latent"][:, -1][:, 0:1, ].to(
                device=self.device, dtype=self.dtype)
        else:
            # clean_latent = None #original code here
            image_latent = None

        batch_size = len(text_prompts)
        image_or_video_shape = list(self.config.image_or_video_shape)
        image_or_video_shape[0] = batch_size

        # Step 2: Extract the conditional infos
        with torch.no_grad():
            conditional_dict = self.model.text_encoder(
                text_prompts=text_prompts)

            if not getattr(self, "unconditional_dict", None):
                unconditional_dict = self.model.text_encoder(
                    text_prompts=[self.config.negative_prompt] * batch_size)
                unconditional_dict = {k: v.detach()
                                      for k, v in unconditional_dict.items()}
                self.unconditional_dict = unconditional_dict  # cache the unconditional_dict
            else:
                unconditional_dict = self.unconditional_dict

        # Step 3: Store gradients for the generator (if training the generator)
        if train_generator:
            generator_loss, generator_log_dict = self.model.generator_loss(
                image_or_video_shape=image_or_video_shape,
                conditional_dict=conditional_dict,
                unconditional_dict=unconditional_dict,
                clean_latent=clean_latent,
                initial_latent=image_latent if self.config.i2v else None
            )
           
            generator_loss.backward()
            generator_grad_norm = self.model.generator.clip_grad_norm_(
                self.max_grad_norm_generator)

            generator_log_dict.update({"generator_loss": generator_loss,
                                       "generator_grad_norm": generator_grad_norm})

            return generator_log_dict
        else:
            generator_log_dict = {}

        # Step 4: Store gradients for the critic (if training the critic)
        critic_loss, critic_log_dict = self.model.critic_loss(
            image_or_video_shape=image_or_video_shape,
            conditional_dict=conditional_dict,
            unconditional_dict=unconditional_dict,
            clean_latent=clean_latent,
            initial_latent=image_latent if self.config.i2v else None
        )

        critic_loss.backward()
        critic_grad_norm = self.model.fake_score.clip_grad_norm_(
            self.max_grad_norm_critic)

        critic_log_dict.update({"critic_loss": critic_loss,
                                "critic_grad_norm": critic_grad_norm})

        return critic_log_dict


    def train(self):
        start_step = self.step
        # Total training steps from config (max_steps); -1 or absent = unlimited.
        max_steps = getattr(self.config, "max_steps", -1)
        step_time_interval = self.step_time_log_interval
        step_time_start_step = self.step
        if step_time_interval > 0:
            if torch.cuda.is_available():
                torch.cuda.synchronize(self.device)
            step_time_start = time.time()
        else:
            step_time_start = None

        while True:
            # Skip the generator update on the very first step so the critic
            # (fake_score) gets warmed up before the generator first moves.
            TRAIN_GENERATOR = self.step > 0 and self.step % self.config.dfake_gen_update_ratio == 0

            # Train the generator
            if TRAIN_GENERATOR:
                self.generator_optimizer.zero_grad(set_to_none=True)
                
                batch = next(self.dataloader)
                generator_log_dict = self.fwdbwd_one_step(batch, True)

                self.generator_optimizer.step()
                if self.generator_ema is not None:
                    self.generator_ema.update(self.model.generator)
                
                
                

            # Train the critic
            self.critic_optimizer.zero_grad(set_to_none=True)
            batch = next(self.dataloader)
            critic_log_dict = self.fwdbwd_one_step(batch, False)
                
            self.critic_optimizer.step()

            # Increment the step since we finished gradient update
            self.step += 1
            step_time_report = None
            if step_time_interval > 0 and (self.step - step_time_start_step) >= step_time_interval:
                if torch.cuda.is_available():
                    torch.cuda.synchronize(self.device)
                current_time = time.time()
                timed_steps = self.step - step_time_start_step
                elapsed_time = current_time - step_time_start
                seconds_per_step = elapsed_time / timed_steps
                step_time_report = (timed_steps, elapsed_time, seconds_per_step)

            # Create EMA params (if not already created)
            if (self.step >= self.config.ema_start_step) and \
                    (self.generator_ema is None) and (self.config.ema_weight > 0):
                self.generator_ema = EMA_FSDP(self.model.generator, decay=self.config.ema_weight)

            # Save the model
            if (not self.config.no_save) and (self.step - start_step) > 0 and self.step % self.config.log_iters == 0:
                torch.cuda.empty_cache()
                self.save()
                torch.cuda.empty_cache()

            # Logging
            if self.is_main_process:
                wandb_loss_dict = {}
                if TRAIN_GENERATOR:
                    wandb_loss_dict.update(
                        {
                            "generator_loss": generator_log_dict["generator_loss"].mean().item(),
                            "generator_grad_norm": generator_log_dict["generator_grad_norm"].mean().item(),
                            "dmdtrain_gradient_norm": generator_log_dict["dmdtrain_gradient_norm"].mean().item()
                        }
                    )

                wandb_loss_dict.update(
                    {
                        "critic_loss": critic_log_dict["critic_loss"].mean().item(),
                        "critic_grad_norm": critic_log_dict["critic_grad_norm"].mean().item()
                    }
                )

                log_str = "  ".join(f"{k}: {v:.4f}" for k, v in wandb_loss_dict.items())
                print(f"[step {self.step}] {log_str}", flush=True)

                if not self.disable_wandb:
                    wandb.log(wandb_loss_dict, step=self.step)

            if self.step % self.config.gc_interval == 0:
                if dist.get_rank() == 0:
                    logging.info("DistGarbageCollector: Running GC.")
                gc.collect()
                torch.cuda.empty_cache()

            if self.is_main_process:
                current_time = time.time()
                if self.previous_time is None:
                    self.previous_time = current_time
                else:
                    if not self.disable_wandb:
                        wandb.log({"per iteration time": current_time - self.previous_time}, step=self.step)
                    self.previous_time = current_time

            if step_time_report is not None:
                timed_steps, elapsed_time, seconds_per_step = step_time_report
                if self.is_main_process:
                    print(
                        f"[timing] step {self.step}: last {timed_steps} steps "
                        f"{elapsed_time:.2f}s total, {seconds_per_step:.2f}s/step",
                        flush=True
                    )
                    if not self.disable_wandb:
                        wandb.log(
                            {
                                "step_time/last_steps_sec": elapsed_time,
                                "step_time/sec_per_step": seconds_per_step,
                                "step_time/interval_steps": timed_steps,
                            },
                            step=self.step
                        )
                if torch.cuda.is_available():
                    torch.cuda.synchronize(self.device)
                step_time_start = time.time()
                step_time_start_step = self.step

            # Stop after reaching max_steps; checkpoint the final step if the
            # periodic save above did not already land on it.
            if max_steps > 0 and self.step >= max_steps:
                if (not self.config.no_save) and (self.step % self.config.log_iters != 0):
                    torch.cuda.empty_cache()
                    self.save()
                    torch.cuda.empty_cache()
                if self.is_main_process:
                    print(f"[step {self.step}] reached max_steps={max_steps}, stopping.", flush=True)
                break
