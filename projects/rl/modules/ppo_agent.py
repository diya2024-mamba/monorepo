from pathlib import Path
from typing import Any, Dict, List, TypeVar

import torch
import torch.nn as nn
import torch.optim as optim
from gymnasium.core import Env as GymnasiumEnv
from gymnasium.spaces import Box as GymnasiumBox
from gymnasium.spaces import Discrete as GymnasiumDiscrete
from modules.networks.actor_critic_agnostic import (
    AgnosticStochasticActor,
    AgnosticVNetwork,
)
from modules.networks.actor_critic_specific import SpecificActor, SpecificVNetwork
from omegaconf import OmegaConf

GymSpace = TypeVar('GymSpace', GymnasiumBox, GymnasiumDiscrete)
Env = GymnasiumEnv


class PPOAgent(nn.Module):
    def __init__(self, cfg: OmegaConf, env_ids: List[str], env_list, mode='pretraining'):
        super().__init__()
        self.cfg = cfg
        self.mode = mode
        if cfg.nn.env_specific_enc_dec:
            self.actor = SpecificActor(cfg, env_ids, env_list)
            self.critic = SpecificVNetwork(cfg, env_ids, env_list)
        else:
            self.actor = AgnosticStochasticActor(cfg, env_ids, env_list)
            self.critic = AgnosticVNetwork(cfg, env_ids, env_list)
        self.cfg = cfg
        self.use_grad_clip = cfg.ppo.use_grad_clip
        self.use_compile = cfg.nn.actor_critic.use_compile

        actor_lr = cfg.ppo.actor_lr
        critic_lr = cfg.ppo.critic_lr

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        # self.actor_lr_schedule = CosineAnnealingLR(self.actor_optimizer, cfg.experiment.total_timesteps)
        # self.critic_lr_schedule = CosineAnnealingLR(self.critic_optimizer, cfg.experiment.total_timesteps)

    def optim_zero_grad(self):
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()

    def optim_step(self):
        if self.use_grad_clip:
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.cfg.ppo.max_grad_norm)
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.cfg.ppo.max_grad_norm)
        self.actor_optimizer.step()
        self.critic_optimizer.step()
        # self.actor_lr_schedule.step()
        # self.critic_lr_schedule.step()

    def get_value(self, env: Env, x: torch.Tensor):
        value = self.critic(env, x)
        return value

    def get_action_and_value(self, env: Env, x: torch.Tensor, action: torch.Tensor = None):
        action_space = env.single_action_space
        value = self.critic(env, x)
        if isinstance(action_space, GymnasiumBox):
            # action_scale = torch.tensor((action_space.high - action_space.low) / 2.0).to(device=x.device)
            # action_bias = torch.tensor((action_space.high + action_space.low) / 2.0).to(device=x.device)
            dist, mean = self.actor(env, x)
            if action is None:
                action = dist.sample()
            log_prob = dist.log_prob(action).sum(-1)
            entropy = dist.entropy().sum(-1)
            return action, log_prob, entropy, value
        elif isinstance(action_space, GymnasiumDiscrete):
            dist, logits = self.actor(env, x)
            if action is None:
                action = dist.sample()
            return action, dist.log_prob(action), dist.entropy(), value

    def get_state_dict(self) -> Dict[str, Any]:
        return {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            # "actor_lr_schedule": self.actor_lr_schedule.state_dict(),
            # "critic_lr_schedule": self.critic_lr_schedule.state_dict(),
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.actor.load_state_dict(state_dict["actor"])
        self.critic.load_state_dict(state_dict["critic"])
        self.actor_optimizer.load_state_dict(state_dict["actor_optimizer"])
        self.critic_optimizer.load_state_dict(state_dict["critic_optimizer"])
        # self.actor_lr_schedule.load_state_dict(state_dict["actor_lr_schedule"])
        # self.critic_lr_schedule.load_state_dict(state_dict["critic_lr_schedule"])

    def load_state_only_weight(self, state_dict: Dict[str, Any]):
        self.actor.load_state_dict(state_dict["actor"])
        self.critic.load_state_dict(state_dict["critic"])

    def save_checkpoint(self, ckpt_dir, update_idx, mode='pretraining') -> None:
        ckpt_dir.mkdir(exist_ok=True, parents=False)
        state_dict = self.get_state_dict()
        state_dict["update_idx"] = update_idx
        torch.save(state_dict, ckpt_dir / f'{mode}_agent.pt')

    def load_checkpoint(self, ckpt_dir, device, mode='pretraining') -> None:
        ckpt_dir = Path(ckpt_dir)
        state_dict = torch.load(ckpt_dir / 'pretraining_agent.pt', map_location=device)
        if mode == 'pretraining':
            self.load_state_dict(state_dict)
        elif mode == 'finetuning':
            self.load_state_only_weight(state_dict)
