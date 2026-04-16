"""
Обучение FP32 DS-CNN на Google Speech Commands v2.

- Adam + cosine decay LR
- L2 regularization + label smoothing
- CSV + TensorBoard logging
- Сохранение в .keras (нативный Keras 3 формат)
- На финале печатает test accuracy, confusion matrix, размер модели
"""

from __future__ import annotations

import csv
import os
import random
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

import config
from data.dataset import build_dataset, build_dataset_cached, count_examples
from models.ds_cnn import build_ds_cnn
from utils.metrics import (
    accuracy_pct,
    per_class_accuracy,
    plot_confusion_matrix,
    print_classification_report,
)


def _set_seeds() -> None:
    random.seed(config.SEED)
    np.random.seed(config.SEED)
    tf.random.set_seed(config.SEED)
    os.environ["PYTHONHASHSEED"] = str(config.SEED)


def _require_manifests() -> None:
    for name in ["train.csv", "val.csv", "test.csv"]:
        p = config.MANIFEST_DIR / name
        if not p.exists():
            print(
                f"[train] ERROR: нет {p}. Запустите `python -m data.preprocess`",
                file=sys.stderr,
            )
            sys.exit(1)


def _build_lr_schedule(
    train_steps_per_epoch: int,
) -> tf.keras.optimizers.schedules.LearningRateSchedule:
    total_steps = train_steps_per_epoch * config.EPOCHS
    return tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=config.LEARNING_RATE_INIT,
        decay_steps=total_steps,
        alpha=config.LEARNING_RATE_MIN / config.LEARNING_RATE_INIT,
    )


def main() -> None:
    _set_seeds()
    _require_manifests()

    n_train = count_examples(config.MANIFEST_DIR / "train.csv")
    n_val = count_examples(config.MANIFEST_DIR / "val.csv")
    n_test = count_examples(config.MANIFEST_DIR / "test.csv")
    print(f"[train] examples: train={n_train} val={n_val} test={n_test}")

    train_ds = build_dataset_cached("train", config.BATCH_SIZE, training=True)
    val_ds = build_dataset_cached("val", config.BATCH_SIZE, training=False)
    test_ds = build_dataset_cached("test", config.BATCH_SIZE, training=False)

    steps_per_epoch = max(1, n_train // config.BATCH_SIZE)
    lr = _build_lr_schedule(steps_per_epoch)

    model = build_ds_cnn()
    model.summary()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss=tf.keras.losses.CategoricalCrossentropy(
            from_logits=True, label_smoothing=config.LABEL_SMOOTHING
        ),
        metrics=[tf.keras.metrics.CategoricalAccuracy(name="acc")],
    )

    best_ckpt_path = config.CHECKPOINT_DIR / "ds_cnn_best.keras"

    csv_logger = tf.keras.callbacks.CSVLogger(
        str(config.LOG_DIR / "train.csv"), append=False
    )
    tb_logger = tf.keras.callbacks.TensorBoard(
        log_dir=str(config.TENSORBOARD_DIR), histogram_freq=0
    )
    checkpt = tf.keras.callbacks.ModelCheckpoint(
        filepath=str(best_ckpt_path),  # FIX: используем ту же переменную
        monitor="val_acc",
        mode="max",
        save_best_only=True,
        save_weights_only=False,
    )

    print(f"[train] старт обучения: {config.EPOCHS} эпох, batch {config.BATCH_SIZE}")
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.EPOCHS,
        callbacks=[csv_logger, tb_logger, checkpt],
        verbose=2,
    )

    if best_ckpt_path.exists():
        model = tf.keras.models.load_model(str(best_ckpt_path), compile=False)
        model.compile(
            optimizer="adam",
            loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
            metrics=[tf.keras.metrics.CategoricalAccuracy(name="acc")],
        )
        print(f"[train] restored best model from {best_ckpt_path}")

    model.save(str(config.FP32_KERAS))
    size_kb = config.FP32_KERAS.stat().st_size / 1024.0
    print(f"[train] saved: {config.FP32_KERAS} ({size_kb:.1f} KB)")

    # Оценка на test
    print("[train] eval on test...")
    y_true: list[int] = []
    y_pred: list[int] = []
    for xb, yb in test_ds:
        logits = model(xb, training=False).numpy()
        y_true.extend(np.argmax(yb.numpy(), axis=-1).tolist())
        y_pred.extend(np.argmax(logits, axis=-1).tolist())
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    acc = accuracy_pct(y_true, y_pred)
    print(f"[train] FINAL TEST ACCURACY: {acc:.2f} %")
    print("[train] per-class:")
    for k, v in per_class_accuracy(y_true, y_pred).items():
        print(f"  {k:>12s}: {v*100:.2f} %")

    print_classification_report(y_true, y_pred)
    plot_confusion_matrix(
        y_true,
        y_pred,
        out_path=config.PLOT_DIR / "cm_fp32.png",
        title=f"FP32 — test acc {acc:.2f}%",
    )

    # мини-сводка
    with open(config.LOG_DIR / "final_fp32.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        w.writerow(["test_accuracy_pct", f"{acc:.4f}"])
        w.writerow(["size_kb", f"{size_kb:.2f}"])
        w.writerow(["params", f"{model.count_params()}"])


if __name__ == "__main__":
    main()
