"""
Оценка FP32 .h5 модели на test set.
Запуск: python evaluate.py
"""

from __future__ import annotations

import sys

import numpy as np
import tensorflow as tf

import config
from data.dataset import build_dataset
from utils.metrics import (
    accuracy_pct,
    per_class_accuracy,
    plot_confusion_matrix,
    print_classification_report,
)


def main() -> None:
    if not config.FP32_H5.exists():
        print(
            f"[evaluate] ERROR: нет {config.FP32_H5}. Сначала запустите train.py",
            file=sys.stderr,
        )
        sys.exit(1)

    model = tf.keras.models.load_model(config.FP32_H5, compile=False)
    test_ds = build_dataset(
        config.MANIFEST_DIR / "test.csv", config.BATCH_SIZE, training=False
    )

    y_true: list[int] = []
    y_pred: list[int] = []
    for xb, yb in test_ds:
        logits = model(xb, training=False).numpy()
        y_true.extend(np.argmax(yb.numpy(), axis=-1).tolist())
        y_pred.extend(np.argmax(logits, axis=-1).tolist())
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    acc = accuracy_pct(y_true, y_pred)
    print(f"[evaluate] test accuracy: {acc:.2f} %")
    for k, v in per_class_accuracy(y_true, y_pred).items():
        print(f"  {k:>12s}: {v*100:.2f} %")
    print_classification_report(y_true, y_pred)
    plot_confusion_matrix(
        y_true,
        y_pred,
        out_path=config.PLOT_DIR / "cm_fp32_eval.png",
        title=f"FP32 — test acc {acc:.2f}%",
    )


if __name__ == "__main__":
    main()
