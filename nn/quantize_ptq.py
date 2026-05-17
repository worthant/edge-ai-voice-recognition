"""
PTQ квантизация для архитектуры.

Принимает RunConfig (это теперь одна архитектура без quant-метки),
грузит FP32-веса из run.fp32_keras_path или legacy папки, делает
полную INT8 PTQ-квантизацию через TFLiteConverter,
оценивает на тестовой выборке, возвращает метрики.

Артефакты:
  runs/<slug>/model_ptq_int8.tflite  — финальная INT8 модель
  runs/<slug>/cm_ptq.png             — матрица ошибок INT8
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

import config
from runs import RunConfig, find_run
from data.dataset import build_dataset_cached
from utils.metrics import (
    accuracy_pct,
    plot_confusion_matrix,
    print_classification_report,
)


def _find_fp32_weights(run: RunConfig) -> Path:
    """Ищет FP32 веса: свой путь -> legacy _qat папка -> legacy _ptq."""
    candidates = [
        run.fp32_keras_path,
        run.legacy_qat_dir / "ds_cnn_fp32.keras",
        run.legacy_ptq_dir / "ds_cnn_fp32.keras",
    ]
    for c in candidates:
        if c.exists():
            return c
    print(f"[ptq] ERROR: no FP32 weights for {run.slug}", file=sys.stderr)
    print(f"[ptq] searched: {[str(c) for c in candidates]}", file=sys.stderr)
    sys.exit(1)


def _load_fp32_model(run: RunConfig) -> tf.keras.Model:
    """Пересобирает архитектуру + грузит веса (обход бага десериализации)."""
    from models.ds_cnn import build_ds_cnn

    model = build_ds_cnn(cfg=run.ds_cnn_config)
    weights_path = _find_fp32_weights(run)
    model.load_weights(str(weights_path))
    print(f"[ptq] loaded FP32 weights from {weights_path}")
    return model


def _representative_dataset_gen():
    ds = build_dataset_cached("train", batch_size=1, training=False)
    count = 0
    for xb, _ in ds:
        yield [tf.cast(xb, tf.float32)]
        count += 1
        if count >= config.PTQ_REPRESENTATIVE_SAMPLES:
            return


def _eval_tflite(tflite_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Прогон INT8-модели на тестовой выборке."""
    interp = tf.lite.Interpreter(model_path=str(tflite_path))
    interp.allocate_tensors()
    in_det = interp.get_input_details()[0]
    out_det = interp.get_output_details()[0]
    in_scale, in_zp = in_det["quantization"]
    out_scale, out_zp = out_det["quantization"]

    test_ds = build_dataset_cached("test", batch_size=1, training=False)
    y_true, y_pred = [], []
    for xb, yb in test_ds:
        x_q = np.round(xb.numpy() / in_scale + in_zp).astype(np.int8)
        interp.set_tensor(in_det["index"], x_q)
        interp.invoke()
        out = interp.get_tensor(out_det["index"]).astype(np.float32)
        out = (out - out_zp) * out_scale
        y_true.append(int(np.argmax(yb.numpy(), axis=-1)[0]))
        y_pred.append(int(np.argmax(out, axis=-1)[0]))
    return np.asarray(y_true), np.asarray(y_pred)


def quantize_ptq(run: RunConfig) -> dict:
    """Возвращает метрики PTQ для архитектуры: int8_acc, size, и т.д."""
    run.run_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'-' * 70}")
    print(f"[ptq] === {run.slug} ===")
    print(f"{'-' * 70}")

    model = _load_fp32_model(run)

    print("[ptq] converting FP32 → INT8 TFLite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = _representative_dataset_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    tflite_model = converter.convert()

    run.ptq_tflite_path.write_bytes(tflite_model)
    size_kb = len(tflite_model) / 1024.0
    print(f"[ptq] saved: {run.ptq_tflite_path} ({size_kb:.1f} KB)")

    y_true, y_pred = _eval_tflite(run.ptq_tflite_path)
    acc = accuracy_pct(y_true, y_pred)
    print(f"[ptq] PTQ INT8 accuracy: {acc:.2f} %")
    print_classification_report(y_true, y_pred)

    plot_confusion_matrix(
        y_true,
        y_pred,
        out_path=run.run_dir / "cm_ptq.png",
        title=f"{run.slug} PTQ INT8 — test acc {acc:.2f}%",
    )

    return {
        "ptq_acc_pct": float(acc),
        "ptq_size_kb": float(size_kb),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slug", required=True)
    args = ap.parse_args()
    run = find_run(args.slug)

    # Импорт save_meta из train (он же сольёт PTQ-метрики с уже существующими)
    from train import save_meta

    update = quantize_ptq(run)
    save_meta(run, update)


if __name__ == "__main__":
    main()
