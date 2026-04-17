"""
Quantization-Aware Training (QAT) INT8.

- Оборачивает FP32 Keras-модель в fake-quant через tfmot
- Дообучает 10 эпох на том же датасете
- Конвертирует в TFLite INT8 (I/O тоже int8)
- Сравнивает с PTQ и FP32
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot

import config
from data.dataset import build_dataset
from utils.metrics import (
    accuracy_pct,
    per_class_accuracy,
    plot_confusion_matrix,
    print_classification_report,
)


def _load_fp32_model() -> tf.keras.Model:
    """Загружает FP32 модель. Пробует .keras, потом чекпоинт, потом .h5."""
    if config.FP32_KERAS.exists():
        return tf.keras.models.load_model(str(config.FP32_KERAS), compile=False)

    ckpt = config.CHECKPOINT_DIR / "ds_cnn_best.keras"
    if ckpt.exists():
        return tf.keras.models.load_model(str(ckpt), compile=False)

    from models.ds_cnn import build_ds_cnn

    model = build_ds_cnn()
    h5 = config.MODEL_DIR / "ds_cnn_fp32.h5"
    if h5.exists():
        model.load_weights(str(h5))
        return model

    print("[qat] ERROR: не найдена модель", file=sys.stderr)
    sys.exit(1)


def _representative_dataset_gen():
    ds = build_dataset(config.MANIFEST_DIR / "train.csv", batch_size=1, training=False)
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
    qat_model.summary()

    train_ds = build_dataset(
        config.MANIFEST_DIR / "train.csv", config.QAT_BATCH_SIZE, training=True
    )
    val_ds = build_dataset(
        config.MANIFEST_DIR / "val.csv", config.QAT_BATCH_SIZE, training=False
    )

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

    # Конвертация
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

    # Eval
    y_true, y_pred = _eval_tflite(config.QAT_TFLITE)
    qat_acc = accuracy_pct(y_true, y_pred)

    # PTQ для сравнения (если есть)
    ptq_acc = None
    if config.PTQ_TFLITE.exists():
        yt, yp = _eval_tflite(config.PTQ_TFLITE)
        ptq_acc = accuracy_pct(yt, yp)

    # FP32 для сравнения
    test_ds = build_dataset(
        config.MANIFEST_DIR / "test.csv", config.BATCH_SIZE, training=False
    )
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
