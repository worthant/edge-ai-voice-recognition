"""
Обучение FP32 DS-CNN по заданному RunConfig.

Изменения относительно прежнего train.py:
- Принимает RunConfig (либо --slug из CLI), а не глобальный DS_CNN_CONFIG.
- Все артефакты пишутся в runs/<slug>/, а не в общую results/models/.
- Сохраняет результаты в meta.json (для последующей агрегации в _index.csv).

Запуск:
    python -m train --slug f176_b6_qat
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from datetime import datetime, timezone

import numpy as np
import tensorflow as tf

import config
from runs import RunConfig, find_run
from data.dataset import build_dataset_cached, count_examples
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
                f"[train] ERROR: no {p}. Run `python -m data.preprocess`",
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


def train_fp32(run: RunConfig) -> dict:
    """
    Обучает FP32 модель для данного RunConfig.
    Возвращает dict с метаданными (для записи в meta.json).
    """
    _set_seeds()
    _require_manifests()
    run.run_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"[train] === {run.slug} ===")
    print(
        f"[train] filters={run.filters} blocks={run.blocks} "
        f"simd_aligned={run.is_simd_aligned}"
    )
    print(f"[train] out: {run.run_dir}")
    print(f"{'='*70}\n")

    n_train = count_examples(config.MANIFEST_DIR / "train.csv")
    n_val = count_examples(config.MANIFEST_DIR / "val.csv")
    n_test = count_examples(config.MANIFEST_DIR / "test.csv")
    print(f"[train] examples: train={n_train} val={n_val} test={n_test}")

    train_ds = build_dataset_cached("train", config.BATCH_SIZE, training=True)
    val_ds = build_dataset_cached("val", config.BATCH_SIZE, training=False)
    test_ds = build_dataset_cached("test", config.BATCH_SIZE, training=False)

    steps_per_epoch = max(1, n_train // config.BATCH_SIZE)
    lr = _build_lr_schedule(steps_per_epoch)

    # ── Build model with run-specific architecture ─────────────────────
    model = build_ds_cnn(cfg=run.ds_cnn_config)
    model.summary()
    n_params = model.count_params()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss=tf.keras.losses.CategoricalCrossentropy(
            from_logits=True, label_smoothing=config.LABEL_SMOOTHING
        ),
        metrics=[tf.keras.metrics.CategoricalAccuracy(name="acc")],
    )

    best_ckpt = run.run_dir / "ds_cnn_best.keras"
    callbacks = [
        tf.keras.callbacks.CSVLogger(str(run.run_dir / "training.csv"), append=False),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(best_ckpt),
            monitor="val_acc",
            mode="max",
            save_best_only=True,
            save_weights_only=False,
        ),
    ]

    print(f"[train] starting: {config.EPOCHS} epochs, batch {config.BATCH_SIZE}")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.EPOCHS,
        callbacks=callbacks,
        verbose=2,
    )

    # Restore best checkpoint
    if best_ckpt.exists():
        model = tf.keras.models.load_model(str(best_ckpt), compile=False)
        model.compile(
            optimizer="adam",
            loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
            metrics=[tf.keras.metrics.CategoricalAccuracy(name="acc")],
        )

    model.save(str(run.fp32_keras_path))
    size_kb = run.fp32_keras_path.stat().st_size / 1024.0
    print(f"[train] saved: {run.fp32_keras_path} ({size_kb:.1f} KB)")

    # ── Evaluate on test set ───────────────────────────────────────────
    y_true, y_pred = [], []
    for xb, yb in test_ds:
        logits = model(xb, training=False).numpy()
        y_true.extend(np.argmax(yb.numpy(), axis=-1).tolist())
        y_pred.extend(np.argmax(logits, axis=-1).tolist())
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    fp32_acc = accuracy_pct(y_true, y_pred)
    print(f"[train] FP32 TEST ACCURACY: {fp32_acc:.2f} %")
    print_classification_report(y_true, y_pred)

    plot_confusion_matrix(
        y_true,
        y_pred,
        out_path=run.run_dir / "cm_fp32.png",
        title=f"{run.slug} FP32 — test acc {fp32_acc:.2f}%",
    )

    meta = {
        "slug": run.slug,
        "filters": run.filters,
        "blocks": run.blocks,
        "quant": run.quant,
        "simd_aligned": run.is_simd_aligned,
        "description": run.description,
        "params": int(n_params),
        "fp32_acc_pct": float(fp32_acc),
        "fp32_size_kb": float(size_kb),
        "train_date_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    return meta


def save_meta(run: RunConfig, meta: dict) -> None:
    """Записывает meta.json, merge-ит с существующим если есть."""
    if run.meta_path.exists():
        existing = json.loads(run.meta_path.read_text())
        existing.update(meta)
        meta = existing
    run.meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"[train] meta: {run.meta_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slug", required=True, help="Run identifier from runs.py")
    args = ap.parse_args()

    run = find_run(args.slug)
    meta = train_fp32(run)
    save_meta(run, meta)


if __name__ == "__main__":
    main()
