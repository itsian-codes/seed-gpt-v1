from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def classification_metrics(y_true: np.ndarray, logits: np.ndarray) -> dict:
    probs = softmax_np(logits)
    y_pred = probs.argmax(axis=-1)
    out = {
        "acc": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }
    if probs.shape[1] == 2 and len(np.unique(y_true)) == 2:
        out["roc_auc"] = float(roc_auc_score(y_true, probs[:, 1]))
    else:
        out["roc_auc"] = float("nan")
    return out


def softmax_np(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)
