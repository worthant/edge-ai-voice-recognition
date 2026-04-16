"""
Depthwise Separable CNN (DS-CNN) вариант S из
"Hello Edge: Keyword Spotting on Microcontrollers" (Zhang et al., 2017).

Архитектура:
    Input [49, 10, 1]
    -> Conv2D(64, 10x4, stride=2x2) + BN + ReLU
    -> N × {DepthwiseConv2D(3x3) + BN + ReLU + Conv2D(64, 1x1) + BN + ReLU}
    -> GlobalAveragePooling2D
    -> Dense(num_classes)

Размер: ~23K параметров, ~38 KB в FP32, ~25-30 KB в INT8 + overhead.
"""

from typing import Optional

import tensorflow as tf
from tensorflow.keras import layers, regularizers

import config


def _regularizer() -> Optional[regularizers.Regularizer]:
    l2 = config.L2_REGULARIZATION
    return regularizers.l2(l2) if l2 and l2 > 0 else None


def _ds_block(x: tf.Tensor, filters: int, kernel, name: str) -> tf.Tensor:
    reg = _regularizer()
    x = layers.DepthwiseConv2D(
        kernel_size=kernel,
        padding="same",
        strides=(1, 1),
        depthwise_regularizer=reg,
        use_bias=False,
        name=f"{name}_dw",
    )(x)
    x = layers.BatchNormalization(name=f"{name}_dw_bn")(x)
    x = layers.ReLU(name=f"{name}_dw_relu")(x)
    x = layers.Conv2D(
        filters=filters,
        kernel_size=(1, 1),
        padding="same",
        strides=(1, 1),
        kernel_regularizer=reg,
        use_bias=False,
        name=f"{name}_pw",
    )(x)
    x = layers.BatchNormalization(name=f"{name}_pw_bn")(x)
    x = layers.ReLU(name=f"{name}_pw_relu")(x)
    return x


def build_ds_cnn(
    input_shape: tuple[int, int, int] = config.INPUT_SHAPE,
    num_classes: int = config.NUM_CLASSES,
    cfg: dict = config.DS_CNN_CONFIG,
) -> tf.keras.Model:
    reg = _regularizer()
    inputs = tf.keras.Input(shape=input_shape, name="mfcc_input")

    x = layers.Conv2D(
        filters=cfg["first_conv_filters"],
        kernel_size=cfg["first_conv_kernel"],
        strides=cfg["first_conv_stride"],
        padding="same",
        kernel_regularizer=reg,
        use_bias=False,
        name="stem_conv",
    )(inputs)
    x = layers.BatchNormalization(name="stem_bn")(x)
    x = layers.ReLU(name="stem_relu")(x)

    for i in range(cfg["num_ds_blocks"]):
        x = _ds_block(x, cfg["ds_filters"], cfg["ds_kernel"], name=f"ds{i+1}")

    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(0.2, name="dropout")(x)
    outputs = layers.Dense(
        num_classes,
        activation=None,  # логиты; softmax применяется в loss или постобработке
        kernel_regularizer=reg,
        name="logits",
    )(x)

    model = tf.keras.Model(inputs, outputs, name="ds_cnn_s")
    return model


if __name__ == "__main__":
    m = build_ds_cnn()
    m.summary()
