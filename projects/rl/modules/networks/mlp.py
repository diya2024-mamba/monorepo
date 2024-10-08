import torch
import torch.nn as nn
from modules.utils import get_activation
from omegaconf import OmegaConf


def layer_init(layer, std=1.4142, bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Block(nn.Module):
    def __init__(self, nn_cfg: OmegaConf, dim: int, expansion_dim=4, dropout=0.0):
        super().__init__()
        activation_name = nn_cfg.mlp.activation
        activation_func = get_activation(activation_name)
        self.fn = nn.Sequential(
            layer_init(
                nn.Linear(dim, expansion_dim), std=1 / expansion_dim
            ),  # 1/(dim*expansion_dim)
            activation_func(),
            nn.Dropout(dropout),
            layer_init(
                nn.Linear(expansion_dim, dim), std=1 / dim
            ),  # std=np.sqrt(expansion_dim*dim)), # 1/(dim*expansion_dim)
            nn.Dropout(dropout),
        )
        self.ln = nn.LayerNorm(dim)

    def forward(self, x):
        return x + self.fn(self.ln(x))


class ResidualMLP(nn.Module):
    def __init__(self, cfg: OmegaConf, input_dim, output_dim):
        super().__init__()
        nn_cfg = cfg.nn
        depth = nn_cfg.mlp.depth
        hidden_dim = nn_cfg.mlp.hidden_dim
        expansion_dim = nn_cfg.mlp.expansion_dim
        dropout = nn_cfg.mlp.dropout
        activation_name = nn_cfg.mlp.activation
        activation_func = get_activation(activation_name)
        self.mlp_input = nn.Sequential(
            layer_init(
                nn.Linear(input_dim, hidden_dim), std=1 / hidden_dim
            ),  # np.sqrt(input_dim*hidden_dim)
            activation_func(),
        )
        self.mlp_blocks = nn.Sequential(
            *[Block(nn_cfg, hidden_dim, expansion_dim, dropout) for _ in range(depth)]
        )
        self.mlp_head = nn.Sequential(
            layer_init(nn.Linear(hidden_dim, output_dim), std=1 / output_dim),
        )

    def forward(self, x):
        x = self.mlp_input(x)
        for _, mlp_block in enumerate(self.mlp_blocks):
            x = mlp_block(x)
        return self.mlp_head(x)
