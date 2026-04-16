"""
Метрики и визуализации для оценки моделей KWS.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

import config


def per_class_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    out: dict[str, float] = {}
    for idx, lbl in config.INDEX_TO_LABEL.items():
        mask = y_true == idx
        if mask.sum() == 0:
            out[lbl] = float("nan")
        else:
            out[lbl] = float((y_pred[mask] == idx).mean())
    return out


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    out_path: Path,
    title: str = "Confusion matrix",
    normalize: bool = True,
) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=list(config.INDEX_TO_LABEL.keys()))
    if normalize:
        row_sum = cm.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0] = 1
        cm = cm.astype(np.float32) / row_sum

    labels = [config.INDEX_TO_LABEL[i] for i in range(config.NUM_CLASSES)]
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        xticklabels=labels,
        yticklabels=labels,
        annot=True,
        fmt=".2f" if normalize else "d",
        cmap="Blues",
        cbar=True,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[metrics] confusion matrix -> {out_path}")


def print_classification_report(y_true: np.ndarray, y_pred: np.ndarray) -> str:
    labels = [config.INDEX_TO_LABEL[i] for i in range(config.NUM_CLASSES)]
    rep = classification_report(
        y_true,
        y_pred,
        labels=list(range(config.NUM_CLASSES)),
        target_names=labels,
        digits=4,
        zero_division=0,
    )
    print(rep)
    return rep


def accuracy_pct(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float((y_true == y_pred).mean() * 100.0)
