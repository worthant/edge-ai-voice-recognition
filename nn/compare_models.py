"""
Сравнительный анализ FP32 / PTQ INT8 / QAT INT8:
- test accuracy
- size (KB)
- средняя latency inference на CPU
- confusion matrix (каждая модель)
- график accuracy vs size
- markdown-таблица в results/comparison.md
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import config
from data.dataset import build_dataset
from utils.metrics import (
    accuracy_pct,
    plot_confusion_matrix,
)


def _time_ms(fn, warmup: int, runs: int) -> float:
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(runs):
        fn()
    elapsed = time.perf_counter() - t0
    return (elapsed / runs) * 1000.0


def _eval_fp32(model_path: Path) -> tuple[float, float, np.ndarray, np.ndarray]:
    model = tf.keras.models.load_model(model_path, compile=False)
    test_ds = build_dataset(
        config.MANIFEST_DIR / "test.csv", config.BATCH_SIZE, training=False
    )

    y_true: list[int] = []
    y_pred: list[int] = []
    sample_x = None
    for xb, yb in test_ds:
        if sample_x is None:
            sample_x = xb[:1].numpy()
        logits = model(xb, training=False).numpy()
        y_true.extend(np.argmax(yb.numpy(), axis=-1).tolist())
        y_pred.extend(np.argmax(logits, axis=-1).tolist())

    y_true_np, y_pred_np = np.asarray(y_true), np.asarray(y_pred)
    acc = accuracy_pct(y_true_np, y_pred_np)

    # latency
    @tf.function
    def _inf(x):
        return model(x, training=False)

    x = tf.constant(sample_x)
    latency = _time_ms(
        lambda: _inf(x).numpy(),
        warmup=config.EVAL_LATENCY_WARMUP,
        runs=config.EVAL_LATENCY_RUNS,
    )
    return acc, latency, y_true_np, y_pred_np


def _eval_tflite(model_path: Path) -> tuple[float, float, np.ndarray, np.ndarray]:
    interp = tf.lite.Interpreter(model_path=str(model_path))
    interp.allocate_tensors()
    in_det = interp.get_input_details()[0]
    out_det = interp.get_output_details()[0]
    in_scale, in_zp = in_det["quantization"]
    out_scale, out_zp = out_det["quantization"]

    test_ds = build_dataset(
        config.MANIFEST_DIR / "test.csv", batch_size=1, training=False
    )

    y_true: list[int] = []
    y_pred: list[int] = []
    sample_x = None
    for xb, yb in test_ds:
        if sample_x is None:
            sample_x = xb.numpy()
        x = xb.numpy()
        x_q = np.round(x / in_scale + in_zp).astype(np.int8)
        interp.set_tensor(in_det["index"], x_q)
        interp.invoke()
        out = interp.get_tensor(out_det["index"]).astype(np.float32)
        out = (out - out_zp) * out_scale
        y_true.append(int(np.argmax(yb.numpy(), axis=-1)[0]))
        y_pred.append(int(np.argmax(out, axis=-1)[0]))

    y_true_np, y_pred_np = np.asarray(y_true), np.asarray(y_pred)
    acc = accuracy_pct(y_true_np, y_pred_np)

    x_q_sample = np.round(sample_x / in_scale + in_zp).astype(np.int8)

    def _inf():
        interp.set_tensor(in_det["index"], x_q_sample)
        interp.invoke()
        _ = interp.get_tensor(out_det["index"])

    latency = _time_ms(
        _inf,
        warmup=config.EVAL_LATENCY_WARMUP,
        runs=config.EVAL_LATENCY_RUNS,
    )
    return acc, latency, y_true_np, y_pred_np


def main() -> None:
    rows: list[dict] = []

    # FP32
    if config.FP32_H5.exists():
        print("[compare] FP32...")
        acc, lat, yt, yp = _eval_fp32(config.FP32_H5)
        size_kb = config.FP32_H5.stat().st_size / 1024.0
        plot_confusion_matrix(
            yt, yp, config.PLOT_DIR / "cm_fp32_compare.png", f"FP32 ({acc:.2f}%)"
        )
        rows.append({"name": "FP32", "acc": acc, "size_kb": size_kb, "lat_ms": lat})
    else:
        print("[compare] FP32 модель отсутствует, пропускаю")

    # PTQ
    if config.PTQ_TFLITE.exists():
        print("[compare] PTQ INT8...")
        acc, lat, yt, yp = _eval_tflite(config.PTQ_TFLITE)
        size_kb = config.PTQ_TFLITE.stat().st_size / 1024.0
        plot_confusion_matrix(
            yt, yp, config.PLOT_DIR / "cm_ptq_compare.png", f"PTQ INT8 ({acc:.2f}%)"
        )
        rows.append({"name": "PTQ INT8", "acc": acc, "size_kb": size_kb, "lat_ms": lat})
    else:
        print("[compare] PTQ модель отсутствует, пропускаю")

    # QAT
    if config.QAT_TFLITE.exists():
        print("[compare] QAT INT8...")
        acc, lat, yt, yp = _eval_tflite(config.QAT_TFLITE)
        size_kb = config.QAT_TFLITE.stat().st_size / 1024.0
        plot_confusion_matrix(
            yt, yp, config.PLOT_DIR / "cm_qat_compare.png", f"QAT INT8 ({acc:.2f}%)"
        )
        rows.append({"name": "QAT INT8", "acc": acc, "size_kb": size_kb, "lat_ms": lat})
    else:
        print("[compare] QAT модель отсутствует, пропускаю")

    if not rows:
        print("[compare] Нет моделей для сравнения.")
        sys.exit(1)

    # Scatter accuracy vs size
    plt.figure(figsize=(7, 5))
    for r in rows:
        plt.scatter(r["size_kb"], r["acc"], s=120)
        plt.annotate(
            r["name"],
            (r["size_kb"], r["acc"]),
            textcoords="offset points",
            xytext=(8, 5),
        )
    plt.xlabel("Size, KB")
    plt.ylabel("Accuracy, %")
    plt.title("Accuracy vs Model Size")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(config.PLOT_DIR / "accuracy_vs_size.png", dpi=150)
    plt.close()

    # Markdown
    lines: list[str] = []
    lines.append("# Сравнение моделей\n")
    lines.append("| Модель | Test Accuracy, % | Size, KB | Inference (CPU, ms) |")
    lines.append("|---|---|---|---|")
    baseline_acc = rows[0]["acc"]
    for r in rows:
        lines.append(
            f"| {r['name']} | {r['acc']:.2f} | {r['size_kb']:.1f} | {r['lat_ms']:.2f} |"
        )

    lines.append("\n## Accuracy drop vs FP32\n")
    for r in rows[1:]:
        lines.append(f"- **{r['name']}**: {baseline_acc - r['acc']:+.2f} п.п.")
    lines.append("\n## Compression ratio (vs FP32)\n")
    for r in rows[1:]:
        ratio = rows[0]["size_kb"] / r["size_kb"]
        lines.append(f"- **{r['name']}**: {ratio:.2f}×")

    lines.append("\n## Графики\n")
    lines.append("- `plots/accuracy_vs_size.png`")
    lines.append(
        "- `plots/cm_fp32_compare.png`, `plots/cm_ptq_compare.png`, `plots/cm_qat_compare.png`"
    )

    out = config.RESULTS_DIR / "comparison.md"
    out.write_text("\n".join(lines))
    print(f"[compare] -> {out}")

    # Также в stdout
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
