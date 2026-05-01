from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class DenseGINLayer(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.eps = nn.Parameter(torch.zeros(1))
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, h: torch.Tensor, adj: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # h: [b,n,d], adj: [b,n,n], mask: [b,n]
        neigh = torch.bmm(adj, h)
        out = self.mlp((1.0 + self.eps) * h + neigh)
        out = self.norm(out)
        out = F.relu(out)
        return out * mask.unsqueeze(-1).float()


class GINEncoder(nn.Module):
    """Dense-batch GIN encoder used as frozen GNN environment f_theta."""

    def __init__(self, in_dim: int, hidden_dim: int, num_layers: int):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.layers = nn.ModuleList(DenseGINLayer(hidden_dim) for _ in range(num_layers))
        self.out_dim = hidden_dim

    def forward(self, x: torch.Tensor, adj: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # add self-loops within valid nodes
        b, n, _ = x.shape
        eye = torch.eye(n, device=x.device).unsqueeze(0).expand(b, n, n)
        adj_loop = torch.where(mask.unsqueeze(1) & mask.unsqueeze(2), adj + eye, torch.zeros_like(adj))
        h = F.relu(self.input_proj(x)) * mask.unsqueeze(-1).float()
        for layer in self.layers:
            h = layer(h, adj_loop, mask)
        g = masked_mean(h, mask)
        return h, g


def masked_mean(h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    denom = mask.sum(dim=1, keepdim=True).clamp_min(1).float()
    return (h * mask.unsqueeze(-1).float()).sum(dim=1) / denom


class PredictionHead(nn.Module):
    """Downstream graph-level prediction head."""

    def __init__(self, in_dim: int, num_classes: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(in_dim, num_classes),
        )

    def forward(self, graph_repr: torch.Tensor) -> torch.Tensor:
        return self.net(graph_repr)


class HyperPromptGenerator(nn.Module):
    """Parameterized prompt generator H_psi(x_i)."""

    def __init__(self, feat_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feat_dim),
        )
        # keep initial prompts small for stable frozen-GNN adaptation
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        p = self.net(x)
        return p * mask.unsqueeze(-1).float()
