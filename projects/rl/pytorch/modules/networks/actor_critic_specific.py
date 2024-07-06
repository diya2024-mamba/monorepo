import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical
from omegaconf import OmegaConf
from modules.networks.mlp import ResidualMLP
from modules.utils import get_activation
from gymnasium.spaces import Box as GymnasiumBox
from gymnasium.spaces import Discrete as GymnasiumDiscrete
from gymnasium.core import Env as GymnasiumEnv
import math
from typing import TypeVar, List, Tuple

GymSpace = TypeVar('GymSpace', GymnasiumBox, GymnasiumDiscrete)
Env = GymnasiumEnv



class SpecificBase(nn.Module):
    def __init__(self, 
                 cfg: OmegaConf,
                env_ids: List[str], 
                env_list: List[Env]
                 ):
        super().__init__()
        self.cfg = cfg
        self.env_ids = env_ids[:]
        activation_name: str = cfg.nn.actor_critic.activation
        self.act_func: torch.nn.Module = get_activation(activation_name)()
        self.d_model = cfg.nn.actor_critic.d_model
        self.encoders_dict = nn.ModuleDict()
        encoders_dict = dict()
        for env_id, env in zip(env_ids, env_list):
            obs_dim = env.single_observation_space.shape
            # for 1-D input: MLP
            obs_dim = np.prod(obs_dim)
            obs_encoder = ResidualMLP(cfg, obs_dim, self.d_model)
            encoders_dict[env_id] = obs_encoder
        self.encoders_dict = nn.ModuleDict(encoders_dict)
        self.use_mlp = cfg.nn.actor_critic.use_mlp
        if self.use_mlp:
            self.shared_mlp: torch.nn.Module = ResidualMLP(cfg, self.d_model, self.d_model)
        
    def add_obs_encoders(self,
                    new_envs: List[Env]):
        # add new_encoders
        for env in new_envs:
            obs_dim = env.single_observation_space.shape
            # for 1-D input: MLP
            if len(obs_dim) < 2:
                obs_dim = np.prod(obs_dim)
                obs_encoder = ResidualMLP(self.cfg, obs_dim, self.d_model)
            self.encoders_dict[env.env_id] = obs_encoder
    
    def add_decoders(self,
                    new_envs: List[Env]):
        pass

    def encoding(self,
                 env: Env,
                 x: torch.Tensor):
        # x: [batch_size, feature_dim] or [batch_size, num_frames, H, W]
        env_id = env.env_id
        h = self.act_func(self.encoders_dict[env_id](x))
        if self.use_mlp:
            h = self.act_func(self.shared_mlp(h))
        else:
            h = self.act_func(h)
        return h
    
    def decoding(self):
        pass
    
    def forward(self):
        pass
        
class SpecificActor(SpecificBase):
    def __init__(self, 
                 cfg: OmegaConf,
                env_ids: List[str], 
                env_list: List[Env]
                 ):
        super().__init__(cfg, env_ids, env_list)
        mean_decoders_dict = dict()
        logstd_decoders_dict = dict()
        prob_decoders_dict = dict()        
        for env_id, env in zip(env_ids, env_list):
            if isinstance(env.single_action_space, GymnasiumBox):
                act_dim = np.prod(env.single_action_space.shape)
                mean_decoder = ResidualMLP(cfg, self.d_model, act_dim) # ResidualMLP(cfg, obs_dim, 256)
                logstd_decoder = ResidualMLP(cfg, self.d_model, act_dim) 
                mean_decoders_dict[env_id] = mean_decoder
                logstd_decoders_dict[env_id] = logstd_decoder
                
            elif isinstance(env.single_action_space, GymnasiumDiscrete):
                act_dim = env.single_action_space.n
                prob_decoder = ResidualMLP(cfg, self.d_model, act_dim) 
                prob_decoders_dict[env_id] = prob_decoder
            
        self.mean_decoders_dict = nn.ModuleDict(mean_decoders_dict)
        self.logstd_decoders_dict = nn.ModuleDict(logstd_decoders_dict)
        self.prob_decoders_dict = nn.ModuleDict(prob_decoders_dict)
    
    def add_policy_decoders(self,
                    new_envs: List[Env]):
        for env in new_envs:
            env_id = env.env_id
            if isinstance(env.single_action_space, GymnasiumBox):
                act_dim = np.prod(env.single_action_space.shape)
                mean_decoder = ResidualMLP(self.cfg, self.d_model, act_dim) # ResidualMLP(cfg, obs_dim, 256)
                logstd_decoder = ResidualMLP(self.cfg, self.d_model, act_dim) 
                self.mean_decoders_dict[env_id] = mean_decoder
                self.logstd_decoders_dict[env_id] = logstd_decoder
                
            elif isinstance(env.single_action_space, GymnasiumDiscrete):
                act_dim = env.single_action_space.n
                prob_decoder = ResidualMLP(self.cfg, self.d_model, act_dim) 
                self.prob_decoders_dict[env_id] = prob_decoder

    def decoding(self,
                 env: Env,
                 h: torch.Tensor):
        # ? Decoding continuous action
        env_id = env.env_id
        single_action_space = env.single_action_space
        if isinstance(single_action_space, GymnasiumBox):
            a_mean = self.mean_decoders_dict[env_id](h)
            a_logstd = self.logstd_decoders_dict[env_id](h)
            self.a_mean_weights = a_mean
            self.a_logstd_weights = a_logstd
            actor_std = F.softplus(a_logstd)
            dist = Normal(a_mean, actor_std)
            return dist, a_mean

        # ? Decoding discrete action
        elif isinstance(single_action_space, GymnasiumDiscrete):
            # get num_discretes
            a_probs = self.prob_decoders_dict[env_id](h)
            self.a_prob_weights = a_probs
            logits = F.softmax(a_probs, dim=-1)
            # get categorical distribution
            dist = Categorical(logits=logits)
            return dist, logits
    
    def forward(self, 
                env: GymSpace,
                x: torch.Tensor):
        # ? Encoding
        h = self.encoding(env, x)
        
        # ? Decoding
        dist, _ = self.decoding(env, h)
        return dist, _

        
class SpecificVNetwork(SpecificBase):
    def __init__(self, 
                 cfg: OmegaConf,
                env_ids: List[str], 
                 env_list: List[Env],):
        super().__init__(cfg, env_ids, env_list)
        decoders_dict = dict()
        for env_id, env in zip(env_ids, env_list):
            val_decoder = ResidualMLP(cfg, self.d_model, 1)
            decoders_dict[env_id] = val_decoder
        self.value_decoder_dict = nn.ModuleDict(decoders_dict)

    def add_v_decoders(self,
                    new_envs: List[Env]):
        for env in new_envs:
            env_id = env.env_id
            val_decoder = ResidualMLP(self.cfg, self.d_model, 1)
            self.value_decoder_dict[env_id] = val_decoder
    
    def decoding(self, 
                 env: Env,
                 h:torch.Tensor):
        env_id = env.env_id
        value = self.value_decoder_dict[env_id](h)
        self.value_weights = value 
        return value
        
    def forward(self, 
                env,
                x:torch.Tensor):
        # ? Encoding
        h = self.encoding(env, x)
        
        # ? Decoding
        value = self.decoding(env, h)
        return value


class SpecificDiscreteQNetwork(SpecificBase):
    def __init__(self, 
                 cfg: OmegaConf,
                env_ids: List[str], 
                 env_list: List[Env],):
        super().__init__(cfg, env_ids, env_list)
        decoders_dict = dict()
        for env_id, env in zip(env_ids, env_list):
            if isinstance(env.single_action_space, GymnasiumDiscrete):
                q_decoder = ResidualMLP(cfg, self.d_model, env.single_action_space.n)  
                decoders_dict[env_id] = q_decoder
            else:
                continue
        self.value_decoder_dict = nn.ModuleDict(decoders_dict)

    def add_q_decoders(self,
                    new_envs: List[Env]):
        for env in new_envs:
            env_id = env.env_id
            if isinstance(env.single_action_space, GymnasiumDiscrete):
                q_decoder = ResidualMLP(self.cfg, self.d_model, env.single_action_space.n)  
                self.value_decoder_dict[env_id] = q_decoder
            else:
                continue
            
    def decoding(self, 
                 env: Env,
                 h:torch.Tensor):
        env_id = env.env_id
        value = self.value_decoder_dict[env_id](h)
        self.value_weights = value 
        return value
        
    def forward(self, 
                env,
                x:torch.Tensor):
        # ? Encoding
        h = self.encoding(env, x)
        
        # ? Decoding
        value = self.decoding(env, h)
        return value


class SpecificContinuousQNetwork(SpecificBase):
    def __init__(self, 
                 cfg: OmegaConf,
                env_ids: List[str], 
                 env_list: List[Env],):
        super().__init__(cfg, env_ids, env_list)
        act_encoders_dict = dict()
        decoders_dict = dict()
        for env_id, env in zip(env_ids, env_list):
            if isinstance(env.single_action_space, GymnasiumBox): 
                act_encoder = ResidualMLP(cfg, np.prod(env.single_action_space.shape), self.d_model)
                act_encoders_dict[env_id] = act_encoder
                q_decoder = ResidualMLP(cfg, self.d_model, 1)
                decoders_dict[env_id] = q_decoder
            else:
                continue
        self.act_encoder_dict = nn.ModuleDict(act_encoders_dict)
        self.value_decoder_dict = nn.ModuleDict(decoders_dict)
    
    def add_act_encoders_q_decoders(self,
                            new_envs: List[Env]):
        for env in new_envs:
            env_id = env.env_id
            if isinstance(env.single_action_space, GymnasiumBox): 
                act_encoder = ResidualMLP(self.cfg, np.prod(env.single_action_space.shape), self.d_model)
                self.act_encoders_dict[env_id] = act_encoder
                q_decoder = ResidualMLP(self.cfg, self.d_model, 1)
                self.value_decoder_dict[env_id] = q_decoder
            else:
                continue
        
    def encoding(self,
                 env: Env,
                 x: torch.Tensor,
                 a: torch.Tensor):
        # x: [batch_size, feature_dim] or [batch_size, num_frames, H, W]
        env_id = env.env_id
        h = self.act_func(self.encoders_dict[env_id](x))
        a_h = self.act_func(self.act_encoder_dict[env_id](a))
        task_number = self.task_id_int_dict[env_id]
        task_number = torch.tensor(task_number, dtype=torch.int).to(x.device)
        task_embed = self.task_embedding(task_number)
        h = h + task_embed + a_h
        if self.use_mlp:
            h = self.act_func(self.shared_mlp(h))
        else:
            h = self.act_func(h)
        return h
    
    def decoding(self, 
                 env: Env,
                 h:torch.Tensor):
        env_id = env.env_id
        value = self.value_decoder_dict[env_id](h)
        self.value_weights = value 
        return value
        
    def forward(self, 
                env,
                x:torch.Tensor,
                a: torch.Tensor):
        # ? Encoding
        h = self.encoding(env, x, a)
        
        # ? Decoding
        value = self.decoding(env, h)
        return value


class SpecificContinuousTwinQ(nn.Module):
    def __init__(self, 
                 cfg: OmegaConf,
                env_ids: List[str], 
                 env_list: List[Env],
                ):
        super().__init__()
        self.cq1 = SpecificContinuousQNetwork(cfg, env_ids, env_list)
        self.cq2 = SpecificContinuousQNetwork(cfg, env_ids, env_list)

    def both(self, 
             env: Env,
             s:torch.Tensor,
             a:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.cq1(env, s, a), self.cq2(env, s, a)

    def forward(self, 
             env: Env,
             s:torch.Tensor,
             a:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return torch.min(*self.both(env, s, a))
        
        
class SpecificDiscreteTwinQ(nn.Module):
    def __init__(self, 
                 cfg: OmegaConf,
                env_ids: List[str], 
                 env_list: List[Env],
                ):
        super().__init__()
        self.dq1 = SpecificDiscreteQNetwork(cfg, env_ids, env_list)
        self.dq2 = SpecificDiscreteQNetwork(cfg, env_ids, env_list)

    def both(self, 
             env: Env,
             s:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.dq1(env, s), self.dq2(env, s)

    def forward(self, 
             env: Env,
             s:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return torch.min(*self.both(env, s))
