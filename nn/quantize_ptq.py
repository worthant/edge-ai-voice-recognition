"""
PTQ для заданного RunConfig.

Загружает FP32-чекпойнт из runs/<slug>/, выполняет полную INT8
квантизацию, сохраняет .tflite и обновляет meta.json.

Запуск:
    python -m quantize_ptq --slug f172_b6_ptq
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

import config
from runs import RunConfig, find_run
from train import save_meta
from data.dataset import build_dataset_cached
from utils.metrics import (
    accuracy_pct,
    plot_confusion_matrix,
    print_classification_report,
)


def _load_fp32_model(run: RunConfig) -> tf.keras.Model:
    """
    Загружает FP32 модель: пересобирает архитектуру из RunConfig + грузит веса.
    Обходит баг десериализации .keras на TF 2.19.

    ПРИМЕЧАНИЕ: PTQ модель может НЕ ИМЕТЬ собственного FP32-чекпойнта (slug
    оканчивается на _ptq). В этом случае используем FP32 от парной QAT
    модели с тем же filters/blocks.
    """
    from models.ds_cnn import build_ds_cnn

    model = build_ds_cnn(cfg=run.ds_cnn_config)

    # Источники весов по приоритету: свой → парная QAT → старый legacy
    candidates: list[Path] = [run.fp32_keras_path]
    if run.quant == "ptq":
        # Парная QAT модель того же размера
        qat_slug = f"f{run.filters}_b{run.blocks}_qat"
        try:
            qat_run = find_run(qat_slug)
            candidates.append(qat_run.fp32_keras_path)
        except ValueError:
            pass

    for src in candidates:
        if src.exists():
            model.load_weights(str(src))
            print(f"[ptq] loaded weights from {src}")
            return model

    print(f"[ptq] ERROR: no FP32 weights found for {run.slug}", file=sys.stderr)
    print(f"[ptq] searched: {[str(c) for c in candidates]}", file=sys.stderr)
    print(
        f"[ptq] hint: run `python -m train --slug f{run.filters}_b{run.blocks}_qat` first",
        file=sys.stderr,
    )
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
    assert run.quant == "ptq", f"quantize_ptq called on non-PTQ run {run.slug}"
    run.run_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"[ptq] === {run.slug} ===")
    print(f"{'='*70}\n")

    model = _load_fp32_model(run)

    print("[ptq] converting FP32 → INT8 TFLite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = _representative_dataset_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    tflite_model = converter.convert()

    run.tflite_path.write_bytes(tflite_model)
    size_kb = len(tflite_model) / 1024.0
    print(f"[ptq] saved: {run.tflite_path} ({size_kb:.1f} KB)")

    y_true, y_pred = _eval_tflite(run.tflite_path)
    ptq_acc = accuracy_pct(y_true, y_pred)
    print(f"[ptq] PTQ INT8 accuracy: {ptq_acc:.2f} %")
    print_classification_report(y_true, y_pred)

    plot_confusion_matrix(
        y_true,
        y_pred,
        out_path=run.run_dir / "cm_ptq.png",
        title=f"{run.slug} PTQ INT8 — test acc {ptq_acc:.2f}%",
    )

    return {
        "int8_acc_pct": float(ptq_acc),
        "int8_size_kb": float(size_kb),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slug", required=True)
    args = ap.parse_args()
    run = find_run(args.slug)
    meta_update = quantize_ptq(run)
    save_meta(run, meta_update)


if __name__ == "__main__":
    main()
