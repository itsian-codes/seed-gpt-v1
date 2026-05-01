from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .agent import (
    SEEDGPTAgent,
    SyntheticExperience,
    RealTransition,
    scatter_edit,
    gather_nodes,
)
from .config import SEEDGPTConfig
from .data import GraphBatch, make_batches
from .metrics import classification_metrics
from .models import GINEncoder, PredictionHead


@dataclass
class VariantFlags:
    use_adversarial: bool = True
    use_seg_te: bool = True
    use_ecr: bool = True


def flags_from_variant(name: str) -> VariantFlags:
    if name == "full":
        return VariantFlags(True, True, True)
    if name == "no_adv":
        return VariantFlags(False, True, True)
    if name == "no_seg_te":
        return VariantFlags(True, False, True)
    if name == "no_ecr":
        return VariantFlags(True, True, False)
    raise ValueError(f"Unknown variant: {name}")


def pretrain_encoder(
    encoder: GINEncoder,
    head: PredictionHead,
    pretrain_examples,
    cfg: SEEDGPTConfig,
    device: torch.device,
) -> None:
    encoder.train(); head.train()
    opt = torch.optim.Adam(list(encoder.parameters()) + list(head.parameters()), lr=cfg.lr_pretrain)
    for epoch in range(cfg.pretrain_epochs):
        for batch in make_batches(pretrain_examples, cfg.batch_size, shuffle=True, seed=cfg.seed + epoch):
            batch = batch.to(device)
            _, g = encoder(batch.x, batch.adj, batch.mask)
            logits = head(g)
            loss = F.cross_entropy(logits, batch.y_pretrain)
            opt.zero_grad()
            loss.backward()
            opt.step()


def evaluate(
    encoder: GINEncoder,
    head: PredictionHead,
    agent: Optional[SEEDGPTAgent],
    examples,
    cfg: SEEDGPTConfig,
    device: torch.device,
    use_prompt: bool = True,
) -> Dict[str, float]:
    encoder.eval(); head.eval()
    if agent is not None:
        agent.eval()
    ys: List[np.ndarray] = []
    logits_all: List[np.ndarray] = []
    with torch.no_grad():
        for batch in make_batches(examples, cfg.batch_size, shuffle=False, seed=cfg.seed):
            batch = batch.to(device)
            if agent is not None and use_prompt:
                prompt = agent.initial_prompt(batch.x, batch.mask)
                # Greedy short editing at evaluation time.
                for _ in range(cfg.rollout_length):
                    node_repr, _ = encoder(batch.x + prompt, batch.adj, batch.mask)
                    state = agent.state_from_node_repr(node_repr, batch.mask)
                    dist = agent.principal.distribution(state, node_repr, batch.mask)
                    node = dist.probs.argmax(dim=-1)
                    edit = agent.principal.edit(state, gather_nodes(node_repr, node))
                    prompt = scatter_edit(prompt, node, edit)
                _, g = encoder(batch.x + prompt, batch.adj, batch.mask)
            else:
                _, g = encoder(batch.x, batch.adj, batch.mask)
            logits = head(g)
            ys.append(batch.y.detach().cpu().numpy())
            logits_all.append(logits.detach().cpu().numpy())
    return classification_metrics(np.concatenate(ys), np.concatenate(logits_all))


def real_interaction_phase(
    encoder: GINEncoder,
    head: PredictionHead,
    agent: SEEDGPTAgent,
    batch: GraphBatch,
    flags: VariantFlags,
) -> Tuple[torch.Tensor, List[RealTransition], Dict[str, float]]:
    """Interact with frozen f_theta and collect real transitions.

    Reward is defined as loss reduction plus optional evaluator-style contextual signal.
    This keeps the implementation consistent with the manuscript while remaining runnable.
    """
    prompt = agent.initial_prompt(batch.x, batch.mask)
    real_transitions: List[RealTransition] = []

    with torch.no_grad():
        _, g0 = encoder(batch.x + prompt, batch.adj, batch.mask)
        prev_loss = F.cross_entropy(head(g0), batch.y, reduction="none")

    total_logprob = []
    for _ in range(agent.cfg.rollout_length):
        node_repr, _ = encoder(batch.x + prompt, batch.adj, batch.mask)
        state = agent.state_from_node_repr(node_repr, batch.mask)
        node, edit, log_prob = agent.principal.act(state, node_repr, batch.mask)
        prompt_next = scatter_edit(prompt, node, edit)

        with torch.no_grad():
            node_repr_next, g_next = encoder(batch.x + prompt_next, batch.adj, batch.mask)
            next_state = agent.state_from_node_repr(node_repr_next, batch.mask)
            new_loss = F.cross_entropy(head(g_next), batch.y, reduction="none")
            task_reward = prev_loss - new_loss
            if flags.use_ecr:
                # External contextual reward proxy: reward for lower loss and smaller excessive edits.
                contextual = 0.05 * torch.tanh(-new_loss) - 0.01 * edit.norm(dim=-1)
                reward = task_reward + contextual
            else:
                reward = task_reward

        real_transitions.append(
            RealTransition(
                state=state.detach(),
                action_node=node.detach(),
                edit_vector=edit.detach(),
                log_prob=log_prob,
                reward=reward.detach(),
                next_state=next_state.detach(),
            )
        )
        total_logprob.append(log_prob)
        prompt = prompt_next
        prev_loss = new_loss.detach()

    _, g_final = encoder(batch.x + prompt, batch.adj, batch.mask)
    logits_final = head(g_final)
    cls_loss = F.cross_entropy(logits_final, batch.y)
    logs = {
        "train_loss": float(cls_loss.detach().cpu()),
        "mean_real_reward": float(torch.cat([t.reward for t in real_transitions]).mean().detach().cpu()),
    }
    return prompt, real_transitions, logs


def synthetic_generation_phase(
    agent: SEEDGPTAgent,
    transitions: List[RealTransition],
    batch: GraphBatch,
    flags: VariantFlags,
) -> List[SyntheticExperience]:
    if not flags.use_seg_te:
        return []
    if len(transitions) == 0:
        return []

    seed_state = transitions[-1].next_state.detach()
    device = seed_state.device
    num_nodes = batch.mask.sum(dim=1).to(device)
    b = seed_state.shape[0]
    candidates: List[SyntheticExperience] = []

    # We use the most recent real state and imagine K trajectories.
    for _ in range(agent.cfg.imagined_trajectories):
        state = seed_state
        cumulative = torch.zeros(b, device=device)
        last_node = torch.zeros(b, dtype=torch.long, device=device)
        last_edit = torch.zeros(b, agent.cfg.num_node_features, device=device)
        next_state = state
        # In synthetic mode, we do not query f_theta. We use transition and reward models.
        for _step in range(agent.cfg.rollout_length):
            # For synthetic rollout, approximate node embeddings by repeating state as a compact context.
            n_max = batch.mask.shape[1]
            node_context = state.unsqueeze(1).expand(b, n_max, state.shape[-1])
            principal_dist = agent.principal.distribution(state, node_context, batch.mask)
            if flags.use_adversarial:
                adv_dist = agent.adversarial.distribution(state, node_context, batch.mask)
                mix_probs = agent.cfg.beta * principal_dist.probs + (1.0 - agent.cfg.beta) * adv_dist.probs
                mix_dist = torch.distributions.Categorical(probs=mix_probs)
                node = mix_dist.sample()
            else:
                node = principal_dist.sample()
            edit = agent.principal.edit(state, gather_nodes(node_context, node))
            pred_reward = agent.evaluator.immediate_reward(state, node, edit, num_nodes)
            next_state = agent.transition_model(state, node, edit, num_nodes)
            cumulative = cumulative + pred_reward
            last_node, last_edit = node, edit
            state = next_state
        # Diversity proxy: prefer non-collapsed selection across the batch.
        diversity = unique_ratio(last_node).to(device)
        score = cumulative + 0.05 * diversity
        candidates.append(SyntheticExperience(
            state=seed_state.detach(),
            action_node=last_node.detach(),
            edit_vector=last_edit.detach(),
            reward=cumulative.detach(),
            next_state=next_state.detach(),
            score=score.detach(),
        ))

    # Select top-M per generated candidate averaged over batch.
    candidates = sorted(candidates, key=lambda c: float(c.score.mean().detach().cpu()), reverse=True)
    accepted = candidates[: max(1, min(agent.cfg.synthetic_batch_size, len(candidates)))]
    agent.buffer.add_many(accepted)
    return accepted


def joint_update_phase(
    encoder: GINEncoder,
    head: PredictionHead,
    agent: SEEDGPTAgent,
    batch: GraphBatch,
    real_transitions: List[RealTransition],
    optimizer: torch.optim.Optimizer,
    flags: VariantFlags,
) -> Dict[str, float]:
    cfg = agent.cfg
    prompt, _, real_logs = real_interaction_phase(encoder, head, agent, batch, flags)
    _, g_final = encoder(batch.x + prompt, batch.adj, batch.mask)
    logits = head(g_final)
    cls_loss = F.cross_entropy(logits, batch.y)

    # PPO-style clipped objective on real transitions.
    ppo_losses = []
    transition_losses = []
    reward_losses = []
    for t in real_transitions:
        # Need fresh compact node context for log prob approximation.
        b = t.state.shape[0]
        n_max = batch.mask.shape[1]
        node_context = t.state.unsqueeze(1).expand(b, n_max, t.state.shape[-1])
        dist = agent.principal.distribution(t.state, node_context, batch.mask)
        new_log_prob = dist.log_prob(t.action_node)
        ratio = torch.exp(new_log_prob - t.log_prob.detach())
        adv = normalize_advantage(t.reward)
        clipped = torch.clamp(ratio, 1.0 - cfg.ppo_clip, 1.0 + cfg.ppo_clip) * adv
        ppo_losses.append(-torch.min(ratio * adv, clipped).mean())

        num_nodes = batch.mask.sum(dim=1)
        pred_next = agent.transition_model(t.state, t.action_node, t.edit_vector, num_nodes)
        pred_reward = agent.evaluator.immediate_reward(t.state, t.action_node, t.edit_vector, num_nodes)
        transition_losses.append(F.mse_loss(pred_next, t.next_state))
        reward_losses.append(F.mse_loss(pred_reward, t.reward))

    ppo_loss = torch.stack(ppo_losses).mean() if ppo_losses else torch.tensor(0.0, device=batch.x.device)
    transition_loss = torch.stack(transition_losses).mean() if transition_losses else torch.tensor(0.0, device=batch.x.device)
    reward_loss = torch.stack(reward_losses).mean() if reward_losses else torch.tensor(0.0, device=batch.x.device)

    syn_loss = torch.tensor(0.0, device=batch.x.device)
    if flags.use_seg_te:
        syn = agent.buffer.sample(cfg.synthetic_batch_size)
        if syn is not None:
            syn = move_synthetic(syn, batch.x.device)
            b = syn.state.shape[0]
            n_max = batch.mask.shape[1]
            # Repeat mask if synthetic sample size differs from current batch.
            syn_mask = repeat_mask(batch.mask, b)
            node_context = syn.state.unsqueeze(1).expand(b, n_max, syn.state.shape[-1])
            dist = agent.principal.distribution(syn.state, node_context, syn_mask)
            adv = normalize_advantage(syn.reward)
            syn_loss = -(dist.log_prob(syn.action_node) * adv).mean()

    adv_loss = torch.tensor(0.0, device=batch.x.device)
    if flags.use_adversarial:
        # Encourage adversarial distribution to differ from principal distribution.
        last_state = real_transitions[-1].next_state if real_transitions else agent.state_from_node_repr(encoder(batch.x, batch.adj, batch.mask)[0], batch.mask)
        n_max = batch.mask.shape[1]
        ctx = last_state.unsqueeze(1).expand(last_state.shape[0], n_max, last_state.shape[-1])
        p = agent.principal.distribution(last_state.detach(), ctx.detach(), batch.mask).probs.clamp_min(1e-8)
        q = agent.adversarial.distribution(last_state.detach(), ctx.detach(), batch.mask).probs.clamp_min(1e-8)
        kl = (p * (p.log() - q.log())).sum(dim=-1).mean()
        adv_loss = -kl

    loss = (
        cfg.classification_loss_weight * cls_loss
        + ppo_loss
        + cfg.transition_loss_weight * transition_loss
        + cfg.reward_loss_weight * reward_loss
        + (cfg.synthetic_policy_weight * syn_loss if flags.use_seg_te else 0.0)
        + (cfg.adversarial_loss_weight * adv_loss if flags.use_adversarial else 0.0)
    )
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(list(agent.parameters()) + list(head.parameters()), 5.0)
    optimizer.step()

    return {
        "loss": float(loss.detach().cpu()),
        "cls_loss": float(cls_loss.detach().cpu()),
        "ppo_loss": float(ppo_loss.detach().cpu()),
        "transition_loss": float(transition_loss.detach().cpu()),
        "reward_loss": float(reward_loss.detach().cpu()),
        "syn_loss": float(syn_loss.detach().cpu()),
        "adv_loss": float(adv_loss.detach().cpu()),
        "buffer_size": float(len(agent.buffer)),
        **real_logs,
    }


def train_seedgpt_epoch(
    encoder: GINEncoder,
    head: PredictionHead,
    agent: SEEDGPTAgent,
    train_examples,
    cfg: SEEDGPTConfig,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    flags: VariantFlags,
    epoch: int,
) -> Dict[str, float]:
    encoder.eval(); head.train(); agent.train()
    logs: List[Dict[str, float]] = []
    for batch in make_batches(train_examples, cfg.batch_size, shuffle=True, seed=cfg.seed + epoch):
        batch = batch.to(device)
        _, real_transitions, _ = real_interaction_phase(encoder, head, agent, batch, flags)
        synthetic_generation_phase(agent, real_transitions, batch, flags)
        logs.append(joint_update_phase(encoder, head, agent, batch, real_transitions, optimizer, flags))
    return average_logs(logs)


def normalize_advantage(x: torch.Tensor) -> torch.Tensor:
    return (x - x.mean()) / (x.std(unbiased=False) + 1e-6)


def unique_ratio(nodes: torch.Tensor) -> torch.Tensor:
    # return [b] scalar broadcast based on batch-level diversity
    ratio = len(torch.unique(nodes)) / max(1, nodes.numel())
    return torch.full_like(nodes.float(), float(ratio))


def repeat_mask(mask: torch.Tensor, b: int) -> torch.Tensor:
    if mask.shape[0] == b:
        return mask
    idx = torch.arange(b, device=mask.device) % mask.shape[0]
    return mask[idx]


def move_synthetic(s: SyntheticExperience, device: torch.device) -> SyntheticExperience:
    return SyntheticExperience(
        state=s.state.to(device),
        action_node=s.action_node.to(device),
        edit_vector=s.edit_vector.to(device),
        reward=s.reward.to(device),
        next_state=s.next_state.to(device),
        score=s.score.to(device),
    )


def average_logs(logs: List[Dict[str, float]]) -> Dict[str, float]:
    if not logs:
        return {}
    keys = logs[0].keys()
    return {k: float(np.mean([d[k] for d in logs])) for k in keys}
