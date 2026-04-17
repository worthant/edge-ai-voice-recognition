"""
QAT с предварительным BN folding.

Шаги:
1. Загружаем FP32 модель
2. Сливаем BatchNorm в предшествующие Conv/DepthwiseConv (BN folding)
3. Строим новую модель БЕЗ BatchNorm
4. tfmot.quantize_model — теперь нет BN → нет NaN
5. Fine-tune 10 эпох
6. Конвертация в TFLite INT8
"""

from __future__ import annotations

import os

os.environ["TF_USE_LEGACY_KERAS"] = "1"

import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot

import config
from data.dataset import build_dataset_cached
from utils.metrics import (
    accuracy_pct,
    plot_confusion_matrix,
    print_classification_report,
)


def _build_folded_model() -> tf.keras.Model:
    """
    Строит DS-CNN БЕЗ BatchNorm — BN слит в свёртки.

    BN folding:
      new_weight = weight * gamma / sqrt(var + eps)
      new_bias   = beta - gamma * mean / sqrt(var + eps)
    """
    from models.ds_cnn import _regularizer

    # Загружаем оригинальную модель с весами
    from models.ds_cnn import build_ds_cnn

    orig = build_ds_cnn()
    npz_path = config.MODEL_DIR / "ds_cnn_fp32_weights.npz"
    if npz_path.exists():
        data = np.load(str(npz_path))
        weights = [data[f"arr_{i}"] for i in range(len(data.files))]
        orig.set_weights(weights)
    else:
        orig.load_weights(str(config.FP32_KERAS))

    # Sanity check
    x_test = np.random.randn(1, 49, 10, 1).astype(np.float32)
    orig_out = orig(x_test, training=False).numpy()
    assert not np.any(np.isnan(orig_out)), "Original model has NaN!"
    print(f"[qat] original model output: {orig_out[0,:3]}")

    def fold_bn(conv_layer, bn_layer):
        """Сливает BN в свёрточный слой, возвращает (new_weights, new_bias)."""
        # BN параметры
        gamma, beta, moving_mean, moving_var = bn_layer.get_weights()
        eps = bn_layer.epsilon

        # Коэффициент масштабирования
        scale = gamma / np.sqrt(moving_var + eps)

        # Conv веса
        conv_weights = conv_layer.get_weights()  # [kernel] или [kernel, bias]
        kernel = conv_weights[0]

        # Для DepthwiseConv2D: kernel shape = (H, W, C, 1)
        # Для Conv2D: kernel shape = (H, W, Cin, Cout)
        if isinstance(conv_layer, tf.keras.layers.DepthwiseConv2D):
            # scale shape: (C,) → нужно (1, 1, C, 1)
            new_kernel = kernel * scale[np.newaxis, np.newaxis, :, np.newaxis]
        else:
            # Conv2D: scale по выходным каналам (последняя ось kernel)
            new_kernel = kernel * scale[np.newaxis, np.newaxis, np.newaxis, :]

        new_bias = beta - moving_mean * scale

        return new_kernel.astype(np.float32), new_bias.astype(np.float32)

    # Строим новую модель без BN
    reg = _regularizer()
    cfg = config.DS_CNN_CONFIG

    inputs = tf.keras.Input(shape=config.INPUT_SHAPE, name="mfcc_input")

    # Stem: Conv + BN → Conv с bias
    stem_k, stem_b = fold_bn(orig.get_layer("stem_conv"), orig.get_layer("stem_bn"))
    x = tf.keras.layers.Conv2D(
        filters=cfg["first_conv_filters"],
        kernel_size=cfg["first_conv_kernel"],
        strides=cfg["first_conv_stride"],
        padding="same",
        use_bias=True,  # теперь с bias!
        kernel_regularizer=reg,
        name="stem_conv",
    )(inputs)
    x = tf.keras.layers.ReLU(name="stem_relu")(x)

    for i in range(cfg["num_ds_blocks"]):
        blk = i + 1
        # DW Conv + BN → DW Conv с bias
        dw_k, dw_b = fold_bn(
            orig.get_layer(f"ds{blk}_dw"),
            orig.get_layer(f"ds{blk}_dw_bn"),
        )
        x = tf.keras.layers.DepthwiseConv2D(
            kernel_size=cfg["ds_kernel"],
            padding="same",
            use_bias=True,
            depthwise_regularizer=reg,
            name=f"ds{blk}_dw",
        )(x)
        x = tf.keras.layers.ReLU(name=f"ds{blk}_dw_relu")(x)

        # PW Conv + BN → PW Conv с bias
        pw_k, pw_b = fold_bn(
            orig.get_layer(f"ds{blk}_pw"),
            orig.get_layer(f"ds{blk}_pw_bn"),
        )
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

    folded = tf.keras.Model(inputs, outputs, name="ds_cnn_folded")

    # Загружаем folded веса
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

    # Dense (logits) — просто копируем веса
    folded.get_layer("logits").set_weights(orig.get_layer("logits").get_weights())

    # Проверяем что folded модель даёт тот же результат
    folded_out = folded(x_test, training=False).numpy()
    diff = np.max(np.abs(orig_out - folded_out))
    print(f"[qat] folded model output: {folded_out[0,:3]}")
    print(f"[qat] max diff vs original: {diff:.6f}")
    if diff > 0.1:
        print(f"[qat] WARNING: folding error too large ({diff:.4f}), check BN params")

    folded.summary()
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


def main() -> None:
    # 1. BN folding
    print("[qat] === BN FOLDING ===")
    folded_model = _build_folded_model()

    # 2. QAT
    print("\n[qat] === QUANTIZE MODEL ===")
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

    print(f"\n[qat] === FINE-TUNE {config.QAT_EPOCHS} эпох ===")
    qat_model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.QAT_EPOCHS,
        callbacks=[
            tf.keras.callbacks.CSVLogger(str(config.LOG_DIR / "qat.csv"), append=False),
        ],
        verbose=2,
    )

    # 3. Конвертация
    print("\n[qat] === CONVERT TO TFLITE INT8 ===")
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

    # 4. Eval
    y_true, y_pred = _eval_tflite(config.QAT_TFLITE)
    qat_acc = accuracy_pct(y_true, y_pred)

    ptq_acc = None
    if config.PTQ_TFLITE.exists():
        yt, yp = _eval_tflite(config.PTQ_TFLITE)
        ptq_acc = accuracy_pct(yt, yp)

    print(f"\n{'='*50}")
    print(f"FP32 baseline:   96.46 %  (reference)")
    if ptq_acc is not None:
        print(f"PTQ INT8:        {ptq_acc:.2f} %")
    print(f"QAT INT8:        {qat_acc:.2f} %")
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
