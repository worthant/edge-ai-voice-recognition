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


def _load_fp32_model() -> tf.keras.Model:
    """Загружает FP32 модель. Пробует .keras, если нет — пересобирает из чекпоинта."""
    if config.FP32_KERAS.exists():
        return tf.keras.models.load_model(str(config.FP32_KERAS), compile=False)

    # Fallback: пересобираем архитектуру + грузим веса из чекпоинта
    ckpt = config.CHECKPOINT_DIR / "ds_cnn_best.keras"
    if ckpt.exists():
        return tf.keras.models.load_model(str(ckpt), compile=False)

    # Последний fallback: старый .h5 через load_weights
    from models.ds_cnn import build_ds_cnn

    model = build_ds_cnn()
    h5 = config.MODEL_DIR / "ds_cnn_fp32.h5"
    if h5.exists():
        model.load_weights(str(h5))
        return model

    print("[ptq] ERROR: не найдена ни .keras, ни .h5, ни чекпоинт", file=sys.stderr)
    sys.exit(1)


def _representative_dataset_gen():
    """500 примеров из train для калибровки квантизации."""
    ds = build_dataset(config.MANIFEST_DIR / "train.csv", batch_size=1, training=False)
    count = 0
    for xb, _ in ds:
        yield [tf.cast(xb, tf.float32)]
        count += 1
        if count >= config.PTQ_REPRESENTATIVE_SAMPLES:
            return


def _eval_tflite(tflite_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Прогоняет TFLite модель на test set, возвращает (y_true, y_pred)."""
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
        x_q = np.round(xb.numpy() / in_scale + in_zp).astype(np.int8)
        interp.set_tensor(in_det["index"], x_q)
        interp.invoke()
        out = interp.get_tensor(out_det["index"]).astype(np.float32)
        out = (out - out_zp) * out_scale
        y_true.append(int(np.argmax(yb.numpy(), axis=-1)[0]))
        y_pred.append(int(np.argmax(out, axis=-1)[0]))
    return np.asarray(y_true), np.asarray(y_pred)


def main() -> None:
    print("[ptq] загрузка FP32 модели...")
    model = _load_fp32_model()

    print("[ptq] конвертация FP32 → INT8 TFLite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = _representative_dataset_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()
    config.PTQ_TFLITE.write_bytes(tflite_model)
    size_kb = len(tflite_model) / 1024.0
    print(f"[ptq] сохранено: {config.PTQ_TFLITE} ({size_kb:.1f} KB)")

    # Eval PTQ на test
    print("[ptq] eval INT8 на test...")
    y_true, y_pred = _eval_tflite(config.PTQ_TFLITE)
    ptq_acc = accuracy_pct(y_true, y_pred)

    # FP32 accuracy для сравнения
    test_ds = build_dataset(
        config.MANIFEST_DIR / "test.csv", config.BATCH_SIZE, training=False
    )
    yt_fp32, yp_fp32 = [], []
    for xb, yb in test_ds:
        logits = model(xb, training=False).numpy()
        yt_fp32.extend(np.argmax(yb.numpy(), axis=-1).tolist())
        yp_fp32.extend(np.argmax(logits, axis=-1).tolist())
    fp32_acc = accuracy_pct(np.asarray(yt_fp32), np.asarray(yp_fp32))
    fp32_size_kb = config.FP32_KERAS.stat().st_size / 1024.0

    print(f"\n{'='*50}")
    print(f"[ptq] FP32 accuracy:      {fp32_acc:.2f} %  ({fp32_size_kb:.1f} KB)")
    print(f"[ptq] PTQ INT8 accuracy:  {ptq_acc:.2f} %  ({size_kb:.1f} KB)")
    print(f"[ptq] accuracy drop:      {fp32_acc - ptq_acc:+.2f} п.п.")
    print(f"[ptq] compression:        {fp32_size_kb / size_kb:.2f}×")
    print(f"{'='*50}")

    print_classification_report(y_true, y_pred)
    plot_confusion_matrix(
        y_true,
        y_pred,
        out_path=config.PLOT_DIR / "cm_ptq.png",
        title=f"PTQ INT8 — test acc {ptq_acc:.2f}%",
    )


if __name__ == "__main__":
    main()
