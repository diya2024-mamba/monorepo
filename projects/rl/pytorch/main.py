import pprint
import warnings
from pathlib import Path

import hydra
import torch
import wandb
from modules.ppo_trainer import PPOTrainer
from omegaconf import OmegaConf

warnings.filterwarnings("ignore", category=DeprecationWarning)


@hydra.main(version_base=None, config_path="./configs", config_name="trainer")
def main(cfg: OmegaConf):
    pprint.pprint(cfg)
    print(f"CUDA is available: {torch.cuda.is_available()}")
    print(f"Number of devices: {torch.cuda.device_count()}")
    architecture = (
        "specific" if cfg.nn.env_specific_enc_dec else "agnostic"
    )  # agnostic or specific
    wandb_name = (
        "PPO"
        + "_"
        + architecture
        + "_"
        + str(cfg.experiment.seed)
        + "_"
        + str(cfg.experiment.finetuning_type)
        + "_"
        + str(cfg.nn.actor_critic.d_model)
    )

    cfg.wandb.group = architecture + str(cfg.experiment.finetuning_type)
    cfg.wandb.name = wandb_name
    output_dir = str(Path(cfg.paths.dir))
    wandb_logger = wandb.init(
        dir=output_dir,
        config=OmegaConf.to_container(cfg, resolve=True),
        # reinit=True,
        resume=True,
        **cfg.wandb,
    )
    trainer = PPOTrainer(cfg)
    trainer.run(wandb_logger)


if __name__ == "__main__":
    main()
