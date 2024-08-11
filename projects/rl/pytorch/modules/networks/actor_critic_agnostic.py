import math
from typing import List, Tuple, TypeVar

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium.core import Env as GymnasiumEnv
from gymnasium.spaces import Box as GymnasiumBox
from gymnasium.spaces import Discrete as GymnasiumDiscrete
from mamba_ssm import Mamba
from modules.networks.mlp import ResidualMLP
from modules.utils import get_activation
from omegaconf import OmegaConf
from torch.distributions import Categorical, Normal

GymSpace = TypeVar("GymSpace", GymnasiumBox, GymnasiumDiscrete)
Env = GymnasiumEnv


def positional_encoding(x: torch.Tensor, d_model: int, seq_len: int):
    position = torch.arange(seq_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
    pe = torch.zeros(1, seq_len, d_model).to(x.device)
    pe[0, :, 0::2] = torch.sin(position * div_term).to(x.device)
    pe[0:, :, 1::2] = torch.cos(position * div_term).to(x.device)
    return x + pe


class MambaEncoder(nn.Module):
    def __init__(
        self,
        cfg: OmegaConf,
    ):
        super().__init__()
        activation_name: str = cfg.nn.actor_critic.activation
        self.d_model = cfg.nn.actor_critic.d_model
        self.act_func: torch.nn.Module = get_activation(activation_name)()
        d_state = cfg.nn.mamba.d_state
        d_conv = cfg.nn.mamba.d_conv
        expand = cfg.nn.mamba.expand
        num_blocks = cfg.nn.mamba.num_blocks
        self.blocks = nn.Sequential()
        self.embedding = nn.Linear(1, self.d_model)
        for _ in range(num_blocks):
            mamba_block = Mamba(
                d_model=self.d_model,  # Model dimension d_model
                d_state=d_state,  # SSM state expansion factor
                d_conv=d_conv,  # Local convolution width
                expand=expand,  # Block expansion factor
            ).cuda()
            activation = self.act_func
            self.blocks.append(mamba_block)
            self.blocks.append(activation)

    def forward(self, x):
        # x: [batch, feature_dim]
        ux = x.unsqueeze(-1)
        ux = self.embedding(ux)
        hidden = self.blocks(ux)
        return hidden


class MambaDecoder(nn.Module):
    def __init__(
        self,
        cfg: OmegaConf,
    ):
        super().__init__()
        activation_name: str = cfg.nn.actor_critic.activation
        self.d_model = cfg.nn.actor_critic.d_model
        self.act_func: torch.nn.Module = get_activation(activation_name)()
        d_state = cfg.nn.mamba.d_state
        d_conv = cfg.nn.mamba.d_conv
        expand = cfg.nn.mamba.expand
        num_blocks = cfg.nn.mamba.num_blocks
        self.blocks = nn.Sequential()
        for _ in range(num_blocks):
            mamba_block = Mamba(
                d_model=self.d_model,  # Model dimension d_model
                d_state=d_state,  # SSM state expansion factor
                d_conv=d_conv,  # Local convolution width
                expand=expand,  # Block expansion factor
            ).cuda()
            activation = self.act_func
            self.blocks.append(mamba_block)
            self.blocks.append(activation)
        self.blocks.pop(-1)

    def forward(self, out_dim: int, hidden: torch.Tensor):
        # x: [batch, out_dim, hidden_dim]
        batch_size = hidden.size(0)
        h = hidden.unsqueeze(1)
        h = h.expand(batch_size, out_dim, self.d_model)
        # if self.pos_encoding:
        # h = positional_encoding(h, self.d_model, out_dim)
        output = self.blocks(h)
        return output


class RNNEncoder(nn.Module):
    def __init__(
        self,
        cfg: OmegaConf,
    ):
        super().__init__()
        activation_name: str = cfg.nn.actor_critic.activation
        self.d_model = cfg.nn.actor_critic.d_model
        self.act_func: torch.nn.Module = get_activation(activation_name)()
        self.blocks = nn.Sequential()
        self.embedding = nn.Linear(1, self.d_model)
        self.num_layers = cfg.nn.rnn.num_layers
        self.rnn = nn.GRU(
            input_size=self.d_model,
            hidden_size=self.d_model,
            num_layers=self.num_layers,
            bias=True,
            batch_first=True,
            bidirectional=True,
        )

    def forward(self, x):
        # x: [batch, feature_dim]
        batch_size = x.size(0)
        feature_dim = x.size(1)
        ux = x.unsqueeze(-1)
        ux = self.embedding(ux)
        hidden, h_n = self.rnn(
            ux
        )  # weights shape: [batch_size, feature_dim, 2*d_model]
        hidden = hidden.reshape(batch_size, feature_dim, 2, self.d_model)
        hidden = hidden.mean(dim=2, keepdim=False)
        return hidden


class RNNDecoder(nn.Module):
    def __init__(
        self,
        cfg: OmegaConf,
    ):
        super().__init__()
        activation_name: str = cfg.nn.actor_critic.activation
        self.d_model = cfg.nn.actor_critic.d_model
        self.pos_encoding = cfg.nn.actor_critic.pos_encoding
        self.act_func: torch.nn.Module = get_activation(activation_name)()
        self.num_layers = cfg.nn.rnn.num_layers
        self.rnn = nn.GRU(
            input_size=self.d_model,
            hidden_size=self.d_model,
            num_layers=self.num_layers,
            bias=True,
            batch_first=True,
            bidirectional=True,
        )

    def forward(self, out_dim: int, hidden: torch.Tensor):
        # x: [batch, out_dim, hidden_dim]
        batch_size = hidden.size(0)
        h = hidden.unsqueeze(1)
        h = h.expand(batch_size, out_dim, self.d_model)
        if self.pos_encoding:
            _x = positional_encoding(h, self.d_model, out_dim)
        weights, _ = self.rnn(h)
        # weights shape: [batch_size, out_dim, 2*d_model]
        weights = weights.reshape(batch_size, out_dim, 2, self.d_model)
        weights = weights.mean(dim=2, keepdim=False)
        # weights shape: [batch_size, out_dim, d_model]
        return weights


class AgnosticBase(nn.Module):
    def __init__(
        self,
        cfg: OmegaConf,
        env_ids: List[str],
        env_list,
    ):
        super().__init__()
        self.d_model = cfg.nn.actor_critic.d_model
        activation_name: str = cfg.nn.actor_critic.activation
        self.act_func: torch.nn.Module = get_activation(activation_name)()
        if cfg.nn.actor_critic.decoder_type == "mamba":
            Encoder = MambaEncoder
        elif cfg.nn.actor_critic.decoder_type == "rnn":
            Encoder = RNNEncoder
        self.obs_encoder_1d: torch.nn.Module = Encoder(cfg)
        self.input_to_hidden: bool = cfg.nn.actor_critic.input_to_hidden
        self.hidden_to_output: bool = cfg.nn.actor_critic.hidden_to_output
        self.use_mlp: bool = cfg.nn.actor_critic.use_mlp
        if self.input_to_hidden == "mean":
            self.in_hidden_op = torch.mean
        elif self.input_to_hidden == "sum":
            self.in_hidden_op = torch.sum
        if self.hidden_to_output == "mean":
            self.out_hidden_op = torch.mean
        elif self.hidden_to_output == "sum":
            self.out_hidden_op = torch.sum
        if self.use_mlp:
            self.res_mlp: torch.nn.Module = ResidualMLP(cfg, self.d_model, self.d_model)
        self.env_ids = env_ids[:]

    def encoding(self, env: Env, x: torch.Tensor):
        # env_id = env.env_id
        # ? Encoding from 1D
        # x: [batch_size, feature_dim]
        h = self.obs_encoder_1d(x)
        h = self.in_hidden_op(h, dim=1, keepdim=False)
        h = self.act_func(h)
        if self.use_mlp:
            h = self.act_func(self.res_mlp(h))
        else:
            h = self.act_func(h)
        return h

    def decoding(self):
        pass

    def forward(self):
        pass


class AgnosticStochasticActor(AgnosticBase):
    def __init__(
        self,
        cfg: OmegaConf,
        env_ids: List[str],
        env_list: List[Env],
    ):
        super().__init__(cfg, env_ids, env_list)
        if cfg.nn.actor_critic.decoder_type == "mamba":
            Decoder = MambaDecoder
        elif cfg.nn.actor_critic.decoder_type == "rnn":
            Decoder = RNNDecoder
        self.policy_mean_decoder: torch.nn.Module = Decoder(cfg)
        self.policy_logstd_decoder: torch.nn.Module = Decoder(cfg)
        self.policy_prob_decoder: torch.nn.Module = Decoder(cfg)

    def decoding(self, action_space: GymSpace, h: torch.Tensor):
        # ? Decoding continuous action
        if isinstance(action_space, GymnasiumBox):
            act_dim = np.prod(action_space.shape)
            a_mean = self.policy_mean_decoder(act_dim, h)
            a_logstd = self.policy_logstd_decoder(act_dim, h)
            self.a_mean_weights = a_mean
            self.a_logstd_weights = a_logstd
            # a_mean, a_logstd: [batch_size, act_dim, d_model]
            a_mu = self.out_hidden_op(
                a_mean, dim=2, keepdim=False
            )  # out: [batch_size, act_dim]
            a_logstd = self.out_hidden_op(
                a_logstd, dim=2, keepdim=False
            )  # out: [batch_size, act_dim]
            actor_std = F.softplus(a_logstd)
            dist = Normal(a_mu, actor_std)
            return dist, a_mu

        # ? Decoding discrete action
        elif isinstance(action_space, GymnasiumDiscrete):
            # get num_discretes
            num_discretes = action_space.n
            # generate policy weights
            a_probs = self.policy_prob_decoder(num_discretes, h)
            self.a_prob_weights = a_probs
            logits = self.out_hidden_op(a_probs, dim=2, keepdim=False)
            # get categorical distribution
            dist = Categorical(logits=logits)
            return dist, logits

    def forward(self, env: Env, x: torch.Tensor):
        action_space = env.single_action_space
        # ? Encoding
        # h: [batch_size, d_model]
        h = self.encoding(env, x)

        # ? Decoding
        dist, _ = self.decoding(action_space, h)
        return dist, _


class AgnosticVNetwork(AgnosticBase):
    def __init__(
        self,
        cfg: OmegaConf,
        env_ids: List[str],
        env_list: List[Env],
    ):
        super().__init__(cfg, env_ids, env_list)
        if cfg.nn.actor_critic.decoder_type == "mamba":
            Decoder = MambaDecoder
        elif cfg.nn.actor_critic.decoder_type == "rnn":
            Decoder = RNNDecoder
        self.value_decoder: torch.nn.Module = Decoder(cfg)

    def decoding(self, h: torch.Tensor):
        value = self.value_decoder(1, h)
        self.value_weights = value
        value = self.out_hidden_op(value, dim=2, keepdim=False)
        return value

    def forward(self, env: Env, x: torch.Tensor):
        # x: [batch_size, feature_dim]
        # ? Encoding
        h = self.encoding(env, x)
        # h: [batch_size, d_model]

        # ? Decoding
        value = self.decoding(h)
        return value


class AgnosticDiscreteQNetwork(AgnosticBase):
    def __init__(
        self,
        cfg: OmegaConf,
        env_ids: List[str],
        env_list: List[Env],
    ):
        super().__init__(cfg, env_ids, env_list)
        self.value_decoder: torch.nn.Module = MambaDecoder(cfg)

    def decoding(self, num_discretes, h: torch.Tensor):
        value = self.value_decoder(num_discretes, h)
        self.value_weights = value
        value = self.out_hidden_op(value, dim=2, keepdim=False)
        return value

    def forward(self, env: Env, x: torch.Tensor):
        # x: [batch_size, feature_dim]
        # ? Encoding
        h = self.encoding(env, x)
        # h: [batch_size, d_model]
        num_discrete = env.action_space.n
        # ? Decoding
        value = self.decoding(num_discrete, h)
        return value


class AgnosticContinuousQNetwork(AgnosticBase):
    def __init__(
        self,
        cfg: OmegaConf,
        env_ids: List[str],
        env_list: List[Env],
    ):
        super().__init__(cfg, env_ids, env_list)
        self.act_encoder_1d: torch.nn.Module = MambaDecoder(cfg)
        self.value_decoder: torch.nn.Module = MambaDecoder(cfg)

    # method overriding
    def encoding(self, env: Env, x: torch.Tensor, a: torch.Tensor):
        # ? Encoding from 1D
        # x: [batch_size, feature_dim]
        h = self.obs_encoder_1d(x)
        a_h = self.act_encoder_1d(a)
        h = self.in_hidden_op(h, dim=1, keepdim=False)
        a_h = self.in_hidden_op(a_h, dim=1, keepdim=False)
        h = h + a_h
        if self.use_mlp:
            h = self.act_func(self.res_mlp(h))
        else:
            h = self.act_func(h)
        return h

    def decoding(self, h: torch.Tensor):
        value = self.value_decoder(1, h)
        if self.use_transformer:
            value, value_attn_maps = self.value_transformer(value)
        self.value_weights = value
        value = self.out_hidden_op(value, dim=2, keepdim=False)
        return value

    def forward(self, env: Env, x: torch.Tensor, a: torch.Tensor):
        # x: [batch_size, feature_dim]
        # ? Encoding
        h = self.encoding(env, x, a)
        # h: [batch_size, d_model]

        # ? Decoding
        value = self.decoding(h)
        return value


class AgnosticContinuousTwinQ(nn.Module):
    def __init__(
        self,
        cfg: OmegaConf,
        env_ids: List[str],
        env_list: List[Env],
    ):
        super().__init__()
        self.cq1 = AgnosticContinuousQNetwork(cfg, env_ids, env_list)
        self.cq2 = AgnosticContinuousQNetwork(cfg, env_ids, env_list)

    def both(
        self, env: Env, s: torch.Tensor, a: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.cq1(env, s, a), self.cq2(env, s, a)

    def forward(
        self, env: Env, s: torch.Tensor, a: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return torch.min(*self.both(env, s, a))


class AgnosticDiscreteTwinQ(nn.Module):
    def __init__(
        self,
        cfg: OmegaConf,
        env_ids: List[str],
        env_list: List[Env],
    ):
        super().__init__()
        self.dq1 = AgnosticDiscreteQNetwork(cfg, env_ids, env_list)
        self.dq2 = AgnosticDiscreteQNetwork(cfg, env_ids, env_list)

    def both(self, env: Env, s: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.dq1(env, s), self.dq2(env, s)

    def forward(self, env: Env, s: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return torch.min(*self.both(env, s))
