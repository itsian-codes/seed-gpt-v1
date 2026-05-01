from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import torch


@dataclass
class GraphExample:
    x: torch.Tensor          # [n, feat_dim]
    adj: torch.Tensor        # [n, n], no self loops required
    y_pretrain: int
    y_downstream: int


@dataclass
class GraphBatch:
    x: torch.Tensor          # [b, n_max, feat_dim]
    adj: torch.Tensor        # [b, n_max, n_max]
    mask: torch.Tensor       # [b, n_max], bool
    y: torch.Tensor          # [b]
    y_pretrain: torch.Tensor # [b]

    def to(self, device: torch.device | str) -> "GraphBatch":
        return GraphBatch(
            x=self.x.to(device),
            adj=self.adj.to(device),
            mask=self.mask.to(device),
            y=self.y.to(device),
            y_pretrain=self.y_pretrain.to(device),
        )


def generate_synthetic_graphs(
    num_graphs: int,
    feat_dim: int,
    min_nodes: int,
    max_nodes: int,
    seed: int,
) -> List[GraphExample]:
    """Create a graph classification benchmark.

    Pretraining task: classify graph density / feature mass.
    Downstream task: classify a motif-like signal mixed with node-feature signal.
    This gives a useful frozen-GNN environment without external datasets.
    """
    rng = np.random.default_rng(seed)
    examples: List[GraphExample] = []
    for _ in range(num_graphs):
        n = int(rng.integers(min_nodes, max_nodes + 1))
        p = float(rng.uniform(0.12, 0.38))
        adj = rng.random((n, n)) < p
        adj = np.triu(adj, 1)
        adj = adj + adj.T
        # Add motif for about half the graphs.
        motif = bool(rng.random() < 0.5)
        if motif and n >= 4:
            nodes = rng.choice(n, 3, replace=False)
            for i in range(3):
                for j in range(i + 1, 3):
                    adj[nodes[i], nodes[j]] = 1
                    adj[nodes[j], nodes[i]] = 1
        deg = adj.sum(axis=1).astype(np.float32)
        x = rng.normal(0, 1, size=(n, feat_dim)).astype(np.float32)
        x[:, 0] += deg / max(1.0, deg.max())
        if motif:
            x[:, 1] += 0.45

        density = adj.sum() / max(1, n * (n - 1))
        y_pretrain = int(density + 0.05 * x[:, 0].mean() > 0.24)
        # Downstream task differs but shares structural / feature information.
        triangle_score = count_triangles(adj) / max(1.0, n)
        y_downstream = int((triangle_score + 0.25 * x[:, 1].mean() + 0.1 * rng.normal()) > 0.08)
        examples.append(
            GraphExample(
                x=torch.tensor(x, dtype=torch.float32),
                adj=torch.tensor(adj.astype(np.float32), dtype=torch.float32),
                y_pretrain=y_pretrain,
                y_downstream=y_downstream,
            )
        )
    return examples


def count_triangles(adj: np.ndarray) -> float:
    a = adj.astype(np.float32)
    return float(np.trace(a @ a @ a) / 6.0)


def stratified_split_few_shot(
    examples: Sequence[GraphExample],
    few_shot_per_class: int,
    seed: int,
) -> Tuple[List[GraphExample], List[GraphExample], List[GraphExample], List[GraphExample]]:
    """Return pretrain pool, downstream train, validation, test."""
    rng = np.random.default_rng(seed)
    idx_by_class = {0: [], 1: []}
    for i, ex in enumerate(examples):
        idx_by_class[ex.y_downstream].append(i)
    train_idx = []
    val_idx = []
    test_idx = []
    for c, idxs in idx_by_class.items():
        idxs = np.array(idxs)
        rng.shuffle(idxs)
        n_train = min(few_shot_per_class, max(1, len(idxs) // 4))
        n_val = max(5, min(len(idxs) - n_train, len(idxs) // 5))
        train_idx.extend(idxs[:n_train].tolist())
        val_idx.extend(idxs[n_train:n_train+n_val].tolist())
        test_idx.extend(idxs[n_train+n_val:].tolist())
    used = set(train_idx + val_idx + test_idx)
    pretrain_idx = [i for i in range(len(examples)) if i not in set(train_idx)]
    return (
        [examples[i] for i in pretrain_idx],
        [examples[i] for i in train_idx],
        [examples[i] for i in val_idx],
        [examples[i] for i in test_idx],
    )


def make_batches(examples: Sequence[GraphExample], batch_size: int, shuffle: bool, seed: int) -> Iterable[GraphBatch]:
    idxs = np.arange(len(examples))
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(idxs)
    for start in range(0, len(idxs), batch_size):
        batch = [examples[int(i)] for i in idxs[start:start+batch_size]]
        yield collate_graphs(batch)


def collate_graphs(batch: Sequence[GraphExample]) -> GraphBatch:
    b = len(batch)
    n_max = max(ex.x.shape[0] for ex in batch)
    feat_dim = batch[0].x.shape[1]
    x = torch.zeros(b, n_max, feat_dim)
    adj = torch.zeros(b, n_max, n_max)
    mask = torch.zeros(b, n_max, dtype=torch.bool)
    y = torch.zeros(b, dtype=torch.long)
    yp = torch.zeros(b, dtype=torch.long)
    for i, ex in enumerate(batch):
        n = ex.x.shape[0]
        x[i, :n] = ex.x
        adj[i, :n, :n] = ex.adj
        mask[i, :n] = True
        y[i] = ex.y_downstream
        yp[i] = ex.y_pretrain
    return GraphBatch(x=x, adj=adj, mask=mask, y=y, y_pretrain=yp)
