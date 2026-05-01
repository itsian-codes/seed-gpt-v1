from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path

import torch
from tqdm import trange

from seedgpt.agent import SEEDGPTAgent
from seedgpt.config import load_config
from seedgpt.data import generate_synthetic_graphs, stratified_split_few_shot
from seedgpt.models import GINEncoder, PredictionHead
from seedgpt.trainer import (
    evaluate,
    flags_from_variant,
    pretrain_encoder,
    train_seedgpt_epoch,
)
from seedgpt.utils import count_trainable_parameters, ensure_dir, freeze_module, save_json, set_seed


def parse_args():
    p = argparse.ArgumentParser(description="Runnable SEED-GPT graph-level prototype")
    p.add_argument("--config", type=str, default="configs/toy_graph.json")
    p.add_argument(
        "--variant",
        type=str,
        default="all",
        choices=["full", "no_adv", "no_seg_te", "no_ecr", "all"],
        help="Run a single SEED-GPT variant or all variants.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() or cfg.device == "cpu" else "cpu")
    run_dir = ensure_dir(cfg.run_dir)

    print(f"[SEED-GPT] device={device} config={args.config}")
    print("[Data] generating synthetic graph benchmark")
    graphs = generate_synthetic_graphs(
        num_graphs=cfg.num_graphs,
        feat_dim=cfg.num_node_features,
        min_nodes=cfg.min_nodes,
        max_nodes=cfg.max_nodes,
        seed=cfg.seed,
    )
    pretrain_set, train_set, val_set, test_set = stratified_split_few_shot(
        graphs, cfg.few_shot_per_class, cfg.seed
    )
    print(f"[Data] pretrain={len(pretrain_set)} train={len(train_set)} val={len(val_set)} test={len(test_set)}")

    print("[Pretrain] training frozen GNN environment f_theta on auxiliary task")
    base_encoder = GINEncoder(cfg.num_node_features, cfg.hidden_dim, cfg.num_gnn_layers).to(device)
    pretrain_head = PredictionHead(cfg.hidden_dim, 2).to(device)
    pretrain_encoder(base_encoder, pretrain_head, pretrain_set, cfg, device)
    freeze_module(base_encoder)
    base_state = copy.deepcopy(base_encoder.state_dict())

    variants = ["full", "no_adv", "no_seg_te", "no_ecr"] if args.variant == "all" else [args.variant]
    all_metrics = {
        "config": cfg.to_dict(),
        "data_sizes": {
            "pretrain": len(pretrain_set),
            "train": len(train_set),
            "val": len(val_set),
            "test": len(test_set),
        },
        "variants": {},
    }

    for variant in variants:
        print(f"\n[Train] variant={variant}")
        set_seed(cfg.seed)
        encoder = GINEncoder(cfg.num_node_features, cfg.hidden_dim, cfg.num_gnn_layers).to(device)
        encoder.load_state_dict(base_state)
        freeze_module(encoder)
        head = PredictionHead(cfg.hidden_dim, 2).to(device)
        agent = SEEDGPTAgent(cfg).to(device)
        flags = flags_from_variant(variant)
        optimizer = torch.optim.Adam(list(agent.parameters()) + list(head.parameters()), lr=cfg.lr_prompt)
        print(f"[Params] trainable agent+head={count_trainable_parameters(agent)+count_trainable_parameters(head)}")

        best_val = -1.0
        best_test = None
        best_epoch = -1
        patience = 0
        history = []
        start = time.time()
        for epoch in trange(cfg.epochs, desc=f"{variant}"):
            train_logs = train_seedgpt_epoch(
                encoder, head, agent, train_set, cfg, optimizer, device, flags, epoch
            )
            val_metrics = evaluate(encoder, head, agent, val_set, cfg, device)
            test_metrics = evaluate(encoder, head, agent, test_set, cfg, device)
            row = {"epoch": epoch, "train": train_logs, "val": val_metrics, "test": test_metrics}
            history.append(row)

            score = val_metrics["roc_auc"] if not torch.isnan(torch.tensor(val_metrics["roc_auc"])) else val_metrics["acc"]
            if score > best_val:
                best_val = score
                best_test = test_metrics
                best_epoch = epoch
                patience = 0
            else:
                patience += 1
                if patience >= cfg.early_stop_patience:
                    break

        elapsed = time.time() - start
        all_metrics["variants"][variant] = {
            "best_epoch": best_epoch,
            "best_val_score": best_val,
            "best_test": best_test,
            "elapsed_sec": elapsed,
            "history": history,
        }
        print(f"[Result] {variant} best_epoch={best_epoch} test={best_test}")

    save_json(all_metrics, run_dir / "metrics.json")
    print(f"\nSaved metrics to {run_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
