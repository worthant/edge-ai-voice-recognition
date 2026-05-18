"""
Пакетный запуск всех архитектур из runs.py.

Для каждой архитектуры:
  1. Train FP32 (в основном процессе через импорт train)
  2. PTQ INT8 в subprocess (изоляция памяти TF между итерациями)
  3. QAT INT8 в subprocess с TF_USE_LEGACY_KERAS=1 (фикс tfmot для TF 2.19)

Идемпотентен: пропускает уже сделанные шаги.

Выбор группы:
    KWS_GROUP=A python -m train_all
    KWS_GROUP=B python -m train_all
    python -m train_all                 # обе

Опции:
    --only f96_b6,f128_b6
    --force
    --skip-fp32                          # не обучать FP32 если нет, использовать legacy
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

from runs import ALL_RUNS, INDEX_CSV, RunConfig, find_run, selected_runs

WORK_DIR = Path(__file__).parent.resolve()


def _is_arch_complete(run: RunConfig) -> bool:
    return (
        run.fp32_keras_path.exists()
        and run.ptq_tflite_path.exists()
        # and run.qat_tflite_path.exists()
        and run.meta_path.exists()
    )


def _run_subprocess(module: str, slug: str) -> bool:
    """Запускает quantize_ptq/quantize_qat в subprocess с TF_USE_LEGACY_KERAS=1."""
    env = os.environ.copy()
    env["TF_USE_LEGACY_KERAS"] = "1"  # обязательно ДО импорта TF в новом процессе
    env["PYTHONUNBUFFERED"] = "1"

    result = subprocess.run(
        [sys.executable, "-m", module, "--slug", slug],
        cwd=str(WORK_DIR),
        env=env,
    )
    return result.returncode == 0


def _train_fp32_step(run: RunConfig, force: bool, skip_fp32: bool) -> bool:
    """FP32 обучение в основном процессе. True если успешно/уже есть."""
    if not force and run.fp32_keras_path.exists():
        print(f"[batch] {run.slug}: FP32 already exists, skipping")
        return True

    if skip_fp32:
        legacy = run.legacy_qat_dir / "ds_cnn_fp32.keras"
        if legacy.exists():
            print(f"[batch] {run.slug}: --skip-fp32, using legacy at {legacy}")
            return True
        legacy2 = run.legacy_ptq_dir / "ds_cnn_fp32.keras"
        if legacy2.exists():
            print(f"[batch] {run.slug}: --skip-fp32, using legacy at {legacy2}")
            return True
        print(f"[batch] {run.slug}: no FP32 and --skip-fp32, abort")
        return False

    try:
        from train import save_meta, train_fp32

        print(f"[batch] {run.slug}: training FP32...")
        fp32_meta = train_fp32(run)
        save_meta(run, fp32_meta)
        return True
    except Exception as e:
        print(f"[batch] FAILED FP32 {run.slug}: {e}", file=sys.stderr)
        traceback.print_exc()
        return False


def _train_one(run: RunConfig, force: bool, skip_fp32: bool) -> dict:
    counters = {"fp32": 0, "ptq": 0, "qat": 0, "errors": 0}

    # FP32 в основном процессе
    if not _train_fp32_step(run, force, skip_fp32):
        counters["errors"] += 1
        return counters
    if not skip_fp32 and run.fp32_keras_path.exists():
        # Считаем как успех обучения только если файл реально появился сейчас.
        # Простой эвристики «если в этой итерации обучили» нет; помечаем 1
        # если файл существует — это удобно для подсчётов в логах.
        counters["fp32"] = 1

    # PTQ в subprocess
    if force or not run.ptq_tflite_path.exists():
        print(f"\n[batch] {run.slug}: launching PTQ subprocess...")
        if _run_subprocess("quantize_ptq", run.slug) and run.ptq_tflite_path.exists():
            counters["ptq"] = 1
        else:
            print(f"[batch] FAILED PTQ {run.slug}", file=sys.stderr)
            counters["errors"] += 1
    else:
        print(f"[batch] {run.slug}: PTQ already exists, skipping")

    # QAT в subprocess (фикс TF_USE_LEGACY_KERAS=1 работает только так)
    # if force or not run.qat_tflite_path.exists():
    #     print(f"\n[batch] {run.slug}: launching QAT subprocess...")
    #     if _run_subprocess("quantize_qat", run.slug) and run.qat_tflite_path.exists():
    #         counters["qat"] = 1
    #     else:
    #         print(f"[batch] FAILED QAT {run.slug}", file=sys.stderr)
    #         counters["errors"] += 1
    # else:
    #     print(f"[batch] {run.slug}: QAT already exists, skipping")

    return counters


def _rebuild_index() -> None:
    rows = []
    for run in ALL_RUNS:
        if not run.meta_path.exists():
            continue
        try:
            meta = json.loads(run.meta_path.read_text())
            rows.append(meta)
        except Exception as e:
            print(f"[batch] skip malformed: {run.meta_path}: {e}", file=sys.stderr)

    if not rows:
        print("[batch] no meta files, _index.csv not written")
        return

    keys: list[str] = []
    for r in rows:
        for k in r.keys():
            if k not in keys:
                keys.append(k)

    with open(INDEX_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[batch] wrote {INDEX_CSV} ({len(rows)} rows)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="Comma-separated slugs")
    ap.add_argument("--force", action="store_true", help="Re-do all steps")
    ap.add_argument("--skip-fp32", action="store_true", help="Reuse legacy FP32")
    args = ap.parse_args()

    runs_to_do: list[RunConfig] = selected_runs()
    if args.only:
        wanted = {s.strip() for s in args.only.split(",")}
        runs_to_do = [find_run(s) for s in wanted]

    if not args.force:
        complete = [r for r in runs_to_do if _is_arch_complete(r)]
        runs_to_do = [r for r in runs_to_do if not _is_arch_complete(r)]
        for r in complete:
            print(f"[batch] SKIP {r.slug} (complete)")

    print(f"\n[batch] will process {len(runs_to_do)} architecture(s):")
    for r in runs_to_do:
        print(f"  - {r.slug}")
    print()

    totals = {"fp32": 0, "ptq": 0, "qat": 0, "errors": 0}
    t0 = time.time()

    for i, run in enumerate(runs_to_do, 1):
        print(f"\n{'#' * 70}")
        print(f"# [{i}/{len(runs_to_do)}] {run.slug}  ({run.description})")
        print(f"{'#' * 70}")
        counters = _train_one(run, args.force, args.skip_fp32)
        for k, v in counters.items():
            totals[k] += v
        _rebuild_index()

    elapsed = (time.time() - t0) / 60
    print(f"\n{'=' * 70}")
    print(f"[batch] done in {elapsed:.1f} min")
    print(
        f"[batch] fp32_ok={totals['fp32']} ptq_done={totals['ptq']} "
        f"qat_done={totals['qat']} errors={totals['errors']}"
    )
    print(f"[batch] index: {INDEX_CSV}")


if __name__ == "__main__":
    main()
