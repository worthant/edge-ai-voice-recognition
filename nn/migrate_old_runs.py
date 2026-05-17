"""
Миграция старых результатов в новую структуру.

Старая структура (один RunConfig = один метод квантизации):
  runs/f96_b6_qat/ds_cnn_fp32.keras
  runs/f96_b6_qat/model_qat_int8.tflite
  runs/f96_b6_qat/meta.json  (qat-specific)
  runs/f96_b6_ptq/model_ptq_int8.tflite
  runs/f96_b6_ptq/meta.json  (ptq-specific)

Новая структура (RunConfig = архитектура, все методы вместе):
  runs/f96_b6/ds_cnn_fp32.keras
  runs/f96_b6/model_qat_int8.tflite
  runs/f96_b6/model_ptq_int8.tflite
  runs/f96_b6/meta.json  (fp32_*, ptq_*, qat_*)

Запуск:
  python -m migrate_old_runs              # dry run, показать что будет сделано
  python -m migrate_old_runs --apply       # реально перенести
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from runs import ALL_RUNS, RUNS_ROOT, RunConfig


def _migrate_one(run: RunConfig, apply: bool) -> dict:
    """Переносит файлы из legacy папок в run.run_dir."""
    actions = []
    meta_merged: dict = {}

    old_qat_dir = run.legacy_qat_dir
    old_ptq_dir = run.legacy_ptq_dir

    # Создаём новую папку если нет
    if apply:
        run.run_dir.mkdir(parents=True, exist_ok=True)

    # FP32 веса
    if not run.fp32_keras_path.exists():
        for src in [
            old_qat_dir / "ds_cnn_fp32.keras",
            old_ptq_dir / "ds_cnn_fp32.keras",
        ]:
            if src.exists():
                actions.append(f"copy {src} -> {run.fp32_keras_path}")
                if apply:
                    shutil.copy2(src, run.fp32_keras_path)
                break

    # PTQ tflite
    if not run.ptq_tflite_path.exists():
        src = old_ptq_dir / "model_ptq_int8.tflite"
        if src.exists():
            actions.append(f"copy {src} -> {run.ptq_tflite_path}")
            if apply:
                shutil.copy2(src, run.ptq_tflite_path)

    # QAT tflite
    if not run.qat_tflite_path.exists():
        src = old_qat_dir / "model_qat_int8.tflite"
        if src.exists():
            actions.append(f"copy {src} -> {run.qat_tflite_path}")
            if apply:
                shutil.copy2(src, run.qat_tflite_path)

    # Метрики — мерджим оба meta.json
    for src_meta, prefix in [
        (old_qat_dir / "meta.json", "qat"),
        (old_ptq_dir / "meta.json", "ptq"),
    ]:
        if not src_meta.exists():
            continue
        try:
            data = json.loads(src_meta.read_text())
        except Exception as e:
            print(f"  ! skip malformed {src_meta}: {e}", file=sys.stderr)
            continue

        # Переименование ключей под новую схему
        # Старая схема: int8_acc_pct, int8_size_kb, fp32_acc_pct, fp32_size_kb
        if "int8_acc_pct" in data:
            meta_merged[f"{prefix}_acc_pct"] = data["int8_acc_pct"]
        if "int8_size_kb" in data:
            meta_merged[f"{prefix}_size_kb"] = data["int8_size_kb"]
        # FP32 поля одинаковые в обоих legacy meta — копируем из любого
        for key in ("fp32_acc_pct", "fp32_size_kb", "params"):
            if key in data and key not in meta_merged:
                meta_merged[key] = data[key]

    if meta_merged:
        meta_merged.update(
            {
                "slug": run.slug,
                "filters": run.filters,
                "blocks": run.blocks,
                "description": run.description,
                "simd_aligned": run.is_simd_aligned,
            }
        )
        actions.append(f"write meta {run.meta_path}: {list(meta_merged.keys())}")
        if apply:
            run.meta_path.write_text(json.dumps(meta_merged, indent=2))

    # Копируем матрицы ошибок (если есть)
    for fname in ("cm_fp32.png", "cm_ptq.png", "cm_qat.png", "training.csv", "qat.csv"):
        dest = run.run_dir / fname
        if dest.exists():
            continue
        for src_dir in (old_qat_dir, old_ptq_dir):
            src = src_dir / fname
            if src.exists():
                actions.append(f"copy {src} -> {dest}")
                if apply:
                    shutil.copy2(src, dest)
                break

    return {"actions": actions, "meta": meta_merged}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="Actually do it")
    args = ap.parse_args()

    if not args.apply:
        print("DRY RUN — use --apply to actually migrate\n")

    total_actions = 0
    for run in ALL_RUNS:
        # Есть ли вообще legacy данные для этой архитектуры?
        has_legacy = run.legacy_qat_dir.exists() or run.legacy_ptq_dir.exists()
        if not has_legacy:
            continue

        print(f"\n=== {run.slug} ===")
        result = _migrate_one(run, args.apply)
        for a in result["actions"]:
            print(f"  {a}")
        total_actions += len(result["actions"])

    print(f"\nTotal actions: {total_actions}")
    if not args.apply:
        print("Re-run with --apply to execute")


if __name__ == "__main__":
    main()
