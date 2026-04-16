"""
Post-Training Quantization (INT8) обученной FP32 модели.

- Полная целочисленная квантизация: веса + активации + I/O в int8
- Representative dataset: 500 примеров из train
- Совместимо с TFLite Micro (никаких float ops на входе/выходе)
"""

from __future__ import annotations

import sys
from pathlib import Path

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


def _representative_dataset_gen():
    ds = build_dataset(config.MANIFEST_DIR / "train.csv", batch_size=1, training=False)
    count = 0
    for xb, _ in ds:
        yield [tf.cast(xb, tf.float32)]
        count += 1
        if count >= config.PTQ_REPRESENTATIVE_SAMPLES:
            return


def _convert_ptq(saved_model_dir: Path, out_path: Path) -> int:
    converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = _representative_dataset_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()
    out_path.write_bytes(tflite_model)
    return len(tflite_model)


def _eval_tflite(tflite_path: Path) -> tuple[np.ndarray, np.ndarray]:
    interp = tf.lite.Interpreter(model_path=str(tflite_path))
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
    for xb, yb in test_ds:
        x = xb.numpy()
        # Квантуем вход в int8
        x_q = np.round(x / in_scale + in_zp).astype(np.int8)
        interp.set_tensor(in_det["index"], x_q)
        interp.invoke()
        out = interp.get_tensor(out_det["index"]).astype(np.float32)
        out = (out - out_zp) * out_scale
        y_true.append(int(np.argmax(yb.numpy(), axis=-1)[0]))
        y_pred.append(int(np.argmax(out, axis=-1)[0]))
    return np.asarray(y_true), np.asarray(y_pred)


def main() -> None:
    if not config.FP32_SAVEDMODEL.exists():
        print(
            f"[ptq] ERROR: нет {config.FP32_SAVEDMODEL}. Запустите train.py",
            file=sys.stderr,
        )
        sys.exit(1)

    print("[ptq] конвертация FP32 SavedModel -> INT8 TFLite...")
    size_bytes = _convert_ptq(config.FP32_SAVEDMODEL, config.PTQ_TFLITE)
    size_kb = size_bytes / 1024.0
    print(f"[ptq] сохранено: {config.PTQ_TFLITE} ({size_kb:.1f} KB)")

    print("[ptq] eval INT8 на test...")
    y_true, y_pred = _eval_tflite(config.PTQ_TFLITE)
    acc = accuracy_pct(y_true, y_pred)

    # FP32 accuracy для сравнения
    fp32 = tf.keras.models.load_model(config.FP32_H5, compile=False)
    t_ds = build_dataset(
        config.MANIFEST_DIR / "test.csv", config.BATCH_SIZE, training=False
    )
    yt, yp = [], []
    for xb, yb in t_ds:
        logits = fp32(xb, training=False).numpy()
        yt.extend(np.argmax(yb.numpy(), axis=-1).tolist())
        yp.extend(np.argmax(logits, axis=-1).tolist())
    fp32_acc = accuracy_pct(np.asarray(yt), np.asarray(yp))

    print(f"[ptq] FP32 accuracy:      {fp32_acc:.2f} %")
    print(f"[ptq] PTQ  accuracy:      {acc:.2f} %")
    print(f"[ptq] accuracy drop:      {fp32_acc - acc:+.2f} п.п.")
    print(f"[ptq] size:               {size_kb:.1f} KB")
    fp32_size_kb = config.FP32_H5.stat().st_size / 1024.0
    print(f"[ptq] compression:        {fp32_size_kb / size_kb:.2f}x (vs FP32 .h5)")

    for k, v in per_class_accuracy(y_true, y_pred).items():
        print(f"  {k:>12s}: {v*100:.2f} %")
    print_classification_report(y_true, y_pred)
    plot_confusion_matrix(
        y_true,
        y_pred,
        out_path=config.PLOT_DIR / "cm_ptq.png",
        title=f"PTQ INT8 — test acc {acc:.2f}%",
    )


if __name__ == "__main__":
    main()
