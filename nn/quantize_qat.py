"""
QAT квантизация для архитектуры.

ВАЖНО: фикс бага tfmot «to_quantize can only either be a Sequential
or Functional model» на TF 2.19 — установить TF_USE_LEGACY_KERAS=1
ДО импорта tf. Это работает только при запуске файла как процесса
(python -m quantize_qat). Если файл импортирован из уже запущенного
python, где TF уже загружен — фикс не сработает. Поэтому train_all
запускает квантизации через subprocess.

Артефакты:
  runs/<slug>/model_qat_int8.tflite
  runs/<slug>/cm_qat.png
  runs/<slug>/qat.csv
"""

from __future__ import annotations

import os

os.environ["TF_USE_LEGACY_KERAS"] = "1"  # ДО import tf

import argparse
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot

import config
from runs import RunConfig, find_run
from data.dataset import build_dataset_cached
from utils.metrics import (
    accuracy_pct,
    plot_confusion_matrix,
    print_classification_report,
)


def _find_fp32_weights(run: RunConfig) -> Path:
    """Свой путь -> legacy _qat -> legacy _ptq."""
    candidates = [
        run.fp32_keras_path,
        run.legacy_qat_dir / "ds_cnn_fp32.keras",
        run.legacy_ptq_dir / "ds_cnn_fp32.keras",
    ]
    for c in candidates:
        if c.exists():
            return c
    print(f"[qat] ERROR: no FP32 weights for {run.slug}", file=sys.stderr)
    print(f"[qat] searched: {[str(c) for c in candidates]}", file=sys.stderr)
    sys.exit(1)


def _build_folded_model(run: RunConfig) -> tf.keras.Model:
    """DS-CNN с BN-folded весами."""
    from models.ds_cnn import _regularizer, build_ds_cnn

    cfg = run.ds_cnn_config

    orig = build_ds_cnn(cfg=cfg)
    weights_path = _find_fp32_weights(run)
    orig.load_weights(str(weights_path))
    print(f"[qat] loaded FP32 weights from {weights_path}")

    x_test = np.random.randn(1, 49, 10, 1).astype(np.float32)
    orig_out = orig(x_test, training=False).numpy()
    assert not np.any(np.isnan(orig_out)), "Original model has NaN!"

    def fold_bn(conv_layer, bn_layer):
        gamma, beta, moving_mean, moving_var = bn_layer.get_weights()
        eps = bn_layer.epsilon
        scale = gamma / np.sqrt(moving_var + eps)
        kernel = conv_layer.get_weights()[0]
        if isinstance(conv_layer, tf.keras.layers.DepthwiseConv2D):
            new_kernel = kernel * scale[np.newaxis, np.newaxis, :, np.newaxis]
        else:
            new_kernel = kernel * scale[np.newaxis, np.newaxis, np.newaxis, :]
        new_bias = beta - moving_mean * scale
        return new_kernel.astype(np.float32), new_bias.astype(np.float32)

    reg = _regularizer()
    inputs = tf.keras.Input(shape=config.INPUT_SHAPE, name="mfcc_input")

    x = tf.keras.layers.Conv2D(
        filters=cfg["first_conv_filters"],
        kernel_size=cfg["first_conv_kernel"],
        strides=cfg["first_conv_stride"],
        padding="same",
        use_bias=True,
        kernel_regularizer=reg,
        name="stem_conv",
    )(inputs)
    x = tf.keras.layers.ReLU(name="stem_relu")(x)

    for i in range(cfg["num_ds_blocks"]):
        blk = i + 1
        x = tf.keras.layers.DepthwiseConv2D(
            kernel_size=cfg["ds_kernel"],
            padding="same",
            use_bias=True,
            depthwise_regularizer=reg,
            name=f"ds{blk}_dw",
        )(x)
        x = tf.keras.layers.ReLU(name=f"ds{blk}_dw_relu")(x)

        x = tf.keras.layers.Conv2D(
            filters=cfg["ds_filters"],
            kernel_size=(1, 1),
            padding="same",
            use_bias=True,
            kernel_regularizer=reg,
            name=f"ds{blk}_pw",
        )(x)
        x = tf.keras.layers.ReLU(name=f"ds{blk}_pw_relu")(x)

    x = tf.keras.layers.GlobalAveragePooling2D(name="gap")(x)
    x = tf.keras.layers.Dropout(0.2, name="dropout")(x)
    outputs = tf.keras.layers.Dense(
        config.NUM_CLASSES,
        activation=None,
        kernel_regularizer=reg,
        name="logits",
    )(x)
    folded = tf.keras.Model(inputs, outputs, name=f"ds_cnn_folded_{run.slug}")

    stem_k, stem_b = fold_bn(orig.get_layer("stem_conv"), orig.get_layer("stem_bn"))
    folded.get_layer("stem_conv").set_weights([stem_k, stem_b])
    for i in range(cfg["num_ds_blocks"]):
        blk = i + 1
        dw_k, dw_b = fold_bn(
            orig.get_layer(f"ds{blk}_dw"),
            orig.get_layer(f"ds{blk}_dw_bn"),
        )
        folded.get_layer(f"ds{blk}_dw").set_weights([dw_k, dw_b])

        pw_k, pw_b = fold_bn(
            orig.get_layer(f"ds{blk}_pw"),
            orig.get_layer(f"ds{blk}_pw_bn"),
        )
        folded.get_layer(f"ds{blk}_pw").set_weights([pw_k, pw_b])

    folded.get_layer("logits").set_weights(orig.get_layer("logits").get_weights())

    folded_out = folded(x_test, training=False).numpy()
    diff = float(np.max(np.abs(orig_out - folded_out)))
    print(f"[qat] folded max diff vs original: {diff:.6f}")
    if diff > 0.1:
        print(f"[qat] WARNING: folding error too large ({diff:.4f})")

    return folded


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


def quantize_qat(run: RunConfig) -> dict:
    run.run_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'-' * 70}")
    print(f"[qat] === {run.slug} ===")
    print(f"{'-' * 70}")

    folded_model = _build_folded_model(run)

    print("[qat] wrapping with tfmot.quantize_model...")
    qat_model = tfmot.quantization.keras.quantize_model(folded_model)
    qat_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.QAT_LEARNING_RATE),
        loss=tf.keras.losses.CategoricalCrossentropy(
            from_logits=True, label_smoothing=config.LABEL_SMOOTHING
        ),
        metrics=[tf.keras.metrics.CategoricalAccuracy(name="acc")],
    )

    train_ds = build_dataset_cached("train", config.QAT_BATCH_SIZE, training=True)
    val_ds = build_dataset_cached("val", config.QAT_BATCH_SIZE, training=False)

    print(f"[qat] fine-tune {config.QAT_EPOCHS} epochs...")
    qat_model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.QAT_EPOCHS,
        callbacks=[
            tf.keras.callbacks.CSVLogger(str(run.run_dir / "qat.csv"), append=False),
        ],
        verbose=2,
    )

    print("[qat] converting to INT8 TFLite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(qat_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = _representative_dataset_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    tflite = converter.convert()
    run.qat_tflite_path.write_bytes(tflite)
    size_kb = len(tflite) / 1024.0
    print(f"[qat] saved: {run.qat_tflite_path} ({size_kb:.1f} KB)")

    y_true, y_pred = _eval_tflite(run.qat_tflite_path)
    acc = accuracy_pct(y_true, y_pred)
    print(f"[qat] QAT INT8 accuracy: {acc:.2f} %")
    print_classification_report(y_true, y_pred)

    plot_confusion_matrix(
        y_true,
        y_pred,
        out_path=run.run_dir / "cm_qat.png",
        title=f"{run.slug} QAT INT8 — test acc {acc:.2f}%",
    )

    return {
        "qat_acc_pct": float(acc),
        "qat_size_kb": float(size_kb),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slug", required=True)
    args = ap.parse_args()
    run = find_run(args.slug)

    from train import save_meta

    update = quantize_qat(run)
    save_meta(run, update)


if __name__ == "__main__":
    main()
