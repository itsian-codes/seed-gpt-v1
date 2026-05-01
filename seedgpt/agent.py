from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import SEEDGPTConfig
from .models import HyperPromptGenerator, masked_mean


@dataclass
class RealTransition:
    state: torch.Tensor
    action_node: torch.Tensor
    edit_vector: torch.Tensor
    log_prob: torch.Tensor
    reward: torch.Tensor
    next_state: torch.Tensor


@dataclass
class SyntheticExperience:
    state: torch.Tensor
    action_node: torch.Tensor
    edit_vector: torch.Tensor
    reward: torch.Tensor
    next_state: torch.Tensor
    score: torch.Tensor


class PrincipalPolicy(nn.Module):
    """pi_P: categorical node selector + continuous edit-vector generator."""

    def __init__(self, state_dim: int, node_dim: int, feat_dim: int, hidden_dim: int, edit_scale: float):
        super().__init__()
        self.selector = nn.Sequential(
            nn.Linear(state_dim + node_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.editor = nn.Sequential(
            nn.Linear(state_dim + node_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feat_dim),
        )
        self.edit_scale = edit_scale

    def distribution(self, state: torch.Tensor, node_repr: torch.Tensor, mask: torch.Tensor) -> torch.distributions.Categorical:
        b, n, _ = node_repr.shape
        s = state.unsqueeze(1).expand(b, n, state.shape[-1])
        logits = self.selector(torch.cat([s, node_repr], dim=-1)).squeeze(-1)
        logits = logits.masked_fill(~mask, -1e9)
        return torch.distributions.Categorical(logits=logits)

    def edit(self, state: torch.Tensor, selected_node_repr: torch.Tensor) -> torch.Tensor:
        raw = self.editor(torch.cat([state, selected_node_repr], dim=-1))
        return self.edit_scale * torch.tanh(raw)

    def act(self, state: torch.Tensor, node_repr: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dist = self.distribution(state, node_repr, mask)
        node = dist.sample()
        log_prob = dist.log_prob(node)
        chosen = gather_nodes(node_repr, node)
        edit = self.edit(state, chosen)
        return node, edit, log_prob


class AdversarialPolicy(nn.Module):
    """pi_A: proposes alternative node-selection paths."""

    def __init__(self, state_dim: int, node_dim: int, hidden_dim: int):
        super().__init__()
        self.selector = nn.Sequential(
            nn.Linear(state_dim + node_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def distribution(self, state: torch.Tensor, node_repr: torch.Tensor, mask: torch.Tensor) -> torch.distributions.Categorical:
        b, n, _ = node_repr.shape
        s = state.unsqueeze(1).expand(b, n, state.shape[-1])
        logits = self.selector(torch.cat([s, node_repr], dim=-1)).squeeze(-1)
        logits = logits.masked_fill(~mask, -1e9)
        return torch.distributions.Categorical(logits=logits)


class TransitionModel(nn.Module):
    """T_omega: predicts next state embedding in synthetic rollouts."""

    def __init__(self, state_dim: int, feat_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + feat_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim),
        )

    def forward(self, state: torch.Tensor, node_index: torch.Tensor, edit_vector: torch.Tensor, num_nodes: torch.Tensor) -> torch.Tensor:
        node_frac = node_index.float().unsqueeze(-1) / num_nodes.float().clamp_min(1).unsqueeze(-1)
        delta = self.net(torch.cat([state, edit_vector, node_frac], dim=-1))
        return state + delta


class TrajectoryEvaluator(nn.Module):
    """TE / R_xi: immediate reward predictor + trajectory scorer."""

    def __init__(self, state_dim: int, feat_dim: int, hidden_dim: int):
        super().__init__()
        self.reward = nn.Sequential(
            nn.Linear(state_dim + feat_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def immediate_reward(self, state: torch.Tensor, node_index: torch.Tensor, edit_vector: torch.Tensor, num_nodes: torch.Tensor) -> torch.Tensor:
        node_frac = node_index.float().unsqueeze(-1) / num_nodes.float().clamp_min(1).unsqueeze(-1)
        return self.reward(torch.cat([state, edit_vector, node_frac], dim=-1)).squeeze(-1)


class SyntheticExperienceBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.items: List[SyntheticExperience] = []

    def add_many(self, xs: List[SyntheticExperience]) -> None:
        self.items.extend(xs)
        if len(self.items) > self.capacity:
            self.items = self.items[-self.capacity:]

    def sample(self, batch_size: int) -> Optional[SyntheticExperience]:
        if not self.items:
            return None
        idx = torch.randint(0, len(self.items), (min(batch_size, len(self.items)),))
        selected = [self.items[int(i)] for i in idx]
        return stack_synthetic(selected)

    def __len__(self) -> int:
        return len(self.items)


class SEEDGPTAgent(nn.Module):
    def __init__(self, cfg: SEEDGPTConfig):
        super().__init__()
        self.cfg = cfg
        feat_dim = cfg.num_node_features
        node_dim = cfg.hidden_dim
        state_dim = cfg.hidden_dim
        self.prompt_generator = HyperPromptGenerator(feat_dim, cfg.prompt_hidden_dim)
        self.state_proj = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
        )
        self.principal = PrincipalPolicy(state_dim, node_dim, feat_dim, cfg.policy_hidden_dim, cfg.edit_scale)
        self.adversarial = AdversarialPolicy(state_dim, node_dim, cfg.policy_hidden_dim)
        self.transition_model = TransitionModel(state_dim, feat_dim, cfg.policy_hidden_dim)
        self.evaluator = TrajectoryEvaluator(state_dim, feat_dim, cfg.policy_hidden_dim)
        self.buffer = SyntheticExperienceBuffer(cfg.synthetic_buffer_size)

    def initial_prompt(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.prompt_generator(x, mask)

    def state_from_node_repr(self, node_repr: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.state_proj(masked_mean(node_repr, mask))


def gather_nodes(node_repr: torch.Tensor, node_index: torch.Tensor) -> torch.Tensor:
    b, _, d = node_repr.shape
    idx = node_index.view(b, 1, 1).expand(b, 1, d)
    return node_repr.gather(1, idx).squeeze(1)


def scatter_edit(prompt: torch.Tensor, node_index: torch.Tensor, edit: torch.Tensor) -> torch.Tensor:
    b, _, d = prompt.shape
    out = prompt.clone()
    idx = node_index.view(b, 1, 1).expand(b, 1, d)
    out.scatter_add_(1, idx, edit.unsqueeze(1))
    return out


def stack_synthetic(items: List[SyntheticExperience]) -> SyntheticExperience:
    return SyntheticExperience(
        state=torch.cat([x.state for x in items], dim=0),
        action_node=torch.cat([x.action_node for x in items], dim=0),
        edit_vector=torch.cat([x.edit_vector for x in items], dim=0),
        reward=torch.cat([x.reward for x in items], dim=0),
        next_state=torch.cat([x.next_state for x in items], dim=0),
        score=torch.cat([x.score for x in items], dim=0),
    )
