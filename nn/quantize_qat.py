"""
Quantization-Aware Training (QAT) INT8.

ВАЖНО: TF_USE_LEGACY_KERAS=1 обязателен, потому что tfmot 0.8.0
написан под tf_keras (Keras 2) и не понимает модели Keras 3.
Эта переменная должна быть установлена ДО любого import tensorflow.
"""

from __future__ import annotations

import os

os.environ["TF_USE_LEGACY_KERAS"] = "1"  # ДО импорта TF!

import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot

import config
from data.dataset import build_dataset_cached
from models.ds_cnn import build_ds_cnn
from utils.metrics import (
    accuracy_pct,
    per_class_accuracy,
    plot_confusion_matrix,
    print_classification_report,
)


def _load_fp32_model() -> tf.keras.Model:
    """
    Пересобирает модель + грузит веса из .weights.h5.

    В режиме TF_USE_LEGACY_KERAS=1 tf.keras = tf_keras (Keras 2).
    build_ds_cnn() создаст tf_keras модель, которую tfmot понимает.
    .weights.h5 — универсальный формат весов, работает и в Keras 2 и 3.
    """
    model = build_ds_cnn()

    weight_sources = [
        config.MODEL_DIR / "ds_cnn_fp32.weights.h5",  # портабельные веса
        config.CHECKPOINT_DIR / "ds_cnn_best.keras",  # чекпоинт
        config.FP32_KERAS,  # полная модель
    ]

    for src in weight_sources:
        if src.exists():
            try:
                model.load_weights(str(src))
                print(f"[qat] загружены веса из {src}")
                return model
            except Exception as e:
                print(f"[qat] не удалось загрузить {src}: {e}")
                continue

    print("[qat] ERROR: не найдены веса модели", file=sys.stderr)
    sys.exit(1)


def _representative_dataset_gen():
    ds = build_dataset_cached("train", batch_size=1, training=False)
    count = 0
    for xb, _ in ds:
        yield [tf.cast(xb, tf.float32)]
        count += 1
        if count >= config.PTQ_REPRESENTATIVE_SAMPLES:
            return


def _eval_tflite(tflite_path: Path) -> tuple[np.ndarray, np.ndarray]:
    interp = tf.lite.Interpreter(model_path=str(tflite_path))
    interp.allocate_tensors()
    in_det = interp.get_input_details()[0]
    out_det = interp.get_output_details()[0]
    in_scale, in_zp = in_det["quantization"]
    out_scale, out_zp = out_det["quantization"]

    test_ds = build_dataset_cached("test", batch_size=1, training=False)

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
    print("[qat] загрузка FP32 модели...")
    fp32_model = _load_fp32_model()

    print("[qat] применяю quantize_model (fake-quant)...")
    qat_model = tfmot.quantization.keras.quantize_model(fp32_model)

    qat_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.QAT_LEARNING_RATE),
        loss=tf.keras.losses.CategoricalCrossentropy(
            from_logits=True, label_smoothing=config.LABEL_SMOOTHING
        ),
        metrics=[tf.keras.metrics.CategoricalAccuracy(name="acc")],
    )

    train_ds = build_dataset_cached("train", config.QAT_BATCH_SIZE, training=True)
    val_ds = build_dataset_cached("val", config.QAT_BATCH_SIZE, training=False)

    csv_logger = tf.keras.callbacks.CSVLogger(
        str(config.LOG_DIR / "qat.csv"), append=False
    )

    print(f"[qat] fine-tune {config.QAT_EPOCHS} эпох...")
    qat_model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.QAT_EPOCHS,
        callbacks=[csv_logger],
        verbose=2,
    )

    print("[qat] конвертация QAT → TFLite INT8...")
    converter = tf.lite.TFLiteConverter.from_keras_model(qat_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = _representative_dataset_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    tflite = converter.convert()
    config.QAT_TFLITE.write_bytes(tflite)
    qat_size_kb = len(tflite) / 1024.0
    print(f"[qat] сохранено: {config.QAT_TFLITE} ({qat_size_kb:.1f} KB)")

    y_true, y_pred = _eval_tflite(config.QAT_TFLITE)
    qat_acc = accuracy_pct(y_true, y_pred)

    ptq_acc = None
    if config.PTQ_TFLITE.exists():
        yt, yp = _eval_tflite(config.PTQ_TFLITE)
        ptq_acc = accuracy_pct(yt, yp)

    test_ds = build_dataset_cached("test", config.BATCH_SIZE, training=False)

    yt_fp32, yp_fp32 = [], []
    for xb, yb in test_ds:
        logits = fp32_model(xb, training=False).numpy()
        yt_fp32.extend(np.argmax(yb.numpy(), axis=-1).tolist())
        yp_fp32.extend(np.argmax(logits, axis=-1).tolist())
    fp32_acc = accuracy_pct(np.asarray(yt_fp32), np.asarray(yp_fp32))

    print(f"\n{'='*50}")
    print(f"FP32 baseline:   {fp32_acc:.2f} %")
    if ptq_acc is not None:
        print(f"PTQ INT8:        {ptq_acc:.2f} %  (drop {fp32_acc - ptq_acc:+.2f})")
    print(f"QAT INT8:        {qat_acc:.2f} %  (drop {fp32_acc - qat_acc:+.2f})")
    print(f"QAT size:        {qat_size_kb:.1f} KB")
    print(f"{'='*50}")

    print_classification_report(y_true, y_pred)
    plot_confusion_matrix(
        y_true,
        y_pred,
        out_path=config.PLOT_DIR / "cm_qat.png",
        title=f"QAT INT8 — test acc {qat_acc:.2f}%",
    )


if __name__ == "__main__":
    main()
