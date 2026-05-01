from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict


@dataclass
class SEEDGPTConfig:
    seed: int = 42
    run_dir: str = "runs/toy_seedgpt"
    device: str = "cpu"

    # data
    num_graphs: int = 600
    num_node_features: int = 16
    min_nodes: int = 10
    max_nodes: int = 28
    few_shot_per_class: int = 25

    # training
    pretrain_epochs: int = 20
    epochs: int = 40
    batch_size: int = 32
    early_stop_patience: int = 10

    # frozen GNN and prompt modules
    hidden_dim: int = 64
    num_gnn_layers: int = 5
    prompt_hidden_dim: int = 64
    policy_hidden_dim: int = 96

    # SEED-GPT-specific terms from the manuscript
    rollout_length: int = 2          # L-step synthetic editing trajectories
    imagined_trajectories: int = 12  # K imagined trajectories per batch
    synthetic_batch_size: int = 16   # top-M accepted synthetic experiences
    synthetic_buffer_size: int = 1024
    beta: float = 0.5                # mixture ratio beta*pi_P+(1-beta)*pi_A
    edit_scale: float = 0.15         # vartheta in tanh-bounded edit vector

    # optimization
    lr_pretrain: float = 3e-3
    lr_prompt: float = 1e-3
    ppo_clip: float = 0.2
    reward_loss_weight: float = 0.5
    transition_loss_weight: float = 0.1
    synthetic_policy_weight: float = 0.5
    adversarial_loss_weight: float = 0.1
    classification_loss_weight: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def load_config(path: str | Path) -> SEEDGPTConfig:
    path = Path(path)
    data = json.loads(path.read_text())
    valid = {field.name for field in SEEDGPTConfig.__dataclass_fields__.values()}
    unknown = sorted(set(data) - valid)
    if unknown:
        raise ValueError(f"Unknown config keys in {path}: {unknown}")
    return SEEDGPTConfig(**data)
