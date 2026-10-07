import argparse
import os
from omegaconf import OmegaConf
import wandb

from trainer import DiffusionTrainer, ODETrainer, ScoreDistillationTrainer, ConsistencyDistillationTrainer


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--no_save", action="store_true")
    parser.add_argument("--no_visualize", action="store_true")
    parser.add_argument("--logdir", type=str, default="", help="Path to the directory to save logs")
    parser.add_argument("--wandb-save-dir", type=str, default="", help="Path to the directory to save wandb logs")
    parser.add_argument("--disable-wandb", action="store_true")
    parser.add_argument("--tf", action="store_true")
    parser.add_argument(
        "--step-time-log-interval",
        type=int,
        default=None,
        help="Print wall-clock timing every N training steps; <=0 disables."
    )

    args = parser.parse_args()

    config = OmegaConf.load(args.config_path)
    default_config = OmegaConf.load("configs/default_config.yaml")
    config = OmegaConf.merge(default_config, config)
    config.no_save = args.no_save
    config.no_visualize = args.no_visualize
    config.tf = args.tf 
    # get the filename of config_path
    config_name = os.path.basename(args.config_path).split(".")[0]
    config.config_name = config_name
    config.logdir = args.logdir
    config.wandb_save_dir = args.wandb_save_dir
    config.disable_wandb = args.disable_wandb
    if args.step_time_log_interval is not None:
        config.step_time_log_interval = args.step_time_log_interval

    if config.trainer == "diffusion":
        trainer = DiffusionTrainer(config)
    elif config.trainer == "ode":
        trainer = ODETrainer(config)
    elif config.trainer == "score_distillation":
        trainer = ScoreDistillationTrainer(config)
    elif config.trainer == "consistency_distillation":
        trainer = ConsistencyDistillationTrainer(config)
    trainer.train()

    wandb.finish()


if __name__ == "__main__":
    main()
