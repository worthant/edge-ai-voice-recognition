"""
Пакетный запуск всех экспериментов из runs.py.

Идемпотентен: если для slug уже есть финальный артефакт
(tflite-файл) — пропускает. Это позволяет:
  - прервать на середине и продолжить позже;
  - удалить одну папку и переобучить только её;
  - добавить новые runs в runs.py — обучатся только они.

После завершения всех runs пересобирает _index.csv — сводную таблицу
всех результатов для построения сравнительных графиков.

Запуск:
    python -m train_all                 # все runs из ALL_RUNS
    python -m train_all --only f176_b6_qat,f176_b6_ptq
    python -m train_all --force          # переобучить всё, игнорировать существующие
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import traceback

from runs import ALL_RUNS, RUNS_ROOT, INDEX_CSV, RunConfig


def _is_done(run: RunConfig) -> bool:
    """Run считается завершённым если есть финальный tflite + meta.json."""
    return run.tflite_path.exists() and run.meta_path.exists()


def _train_one(run: RunConfig) -> bool:
    """
    Прогоняет полный пайплайн для одного RunConfig.
      - fp32:  train.train_fp32 → save_meta
      - ptq:   train_fp32 (если ещё нет) → quantize_ptq → save_meta
      - qat:   train_fp32 (если ещё нет) → quantize_qat → save_meta
    Возвращает True если успешно.
    """
    # Импорт внутри функции: train.py импортирует tensorflow, что медленно.
    # Не хотим грузить TF только чтобы посчитать сколько runs пропустить.
    from train import train_fp32, save_meta

    try:
        # FP32 нужен всем методам квантизации (PTQ грузит веса, QAT folding'ит)
        if not run.fp32_keras_path.exists():
            fp32_meta = train_fp32(run)
            save_meta(run, fp32_meta)
        else:
            print(f"[batch] FP32 already exists for {run.slug}, skipping training")

        if run.quant == "ptq":
            from quantize_ptq import quantize_ptq

            meta_update = quantize_ptq(run)
            save_meta(run, meta_update)
        elif run.quant == "qat":
            from quantize_qat import quantize_qat

            meta_update = quantize_qat(run)
            save_meta(run, meta_update)
        elif run.quant == "fp32":
            # FP32 уже обучен и сохранён в .keras, отдельная TFLite не нужна
            pass
        else:
            print(f"[batch] unknown quant method: {run.quant}", file=sys.stderr)
            return False

        return True
    except Exception as e:
        print(f"[batch] FAILED {run.slug}: {e}", file=sys.stderr)
        traceback.print_exc()
        return False


def _rebuild_index() -> None:
    """Сводная таблица всех meta.json → _index.csv."""
    rows = []
    for run in ALL_RUNS:
        if not run.meta_path.exists():
            continue
        try:
            meta = json.loads(run.meta_path.read_text())
            rows.append(meta)
        except Exception as e:
            print(f"[batch] skip malformed meta: {run.meta_path}: {e}", file=sys.stderr)

    if not rows:
        print("[batch] no meta files yet, _index.csv not written")
        return

    # Объединение всех ключей из всех meta-файлов
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
    ap.add_argument(
        "--only", default=None, help="Comma-separated slugs to run (default: all)"
    )
    ap.add_argument("--force", action="store_true", help="Re-run even if outputs exist")
    args = ap.parse_args()

    runs_to_do: list[RunConfig] = ALL_RUNS
    if args.only:
        wanted = {s.strip() for s in args.only.split(",")}
        runs_to_do = [r for r in ALL_RUNS if r.slug in wanted]
        missing = wanted - {r.slug for r in runs_to_do}
        if missing:
            print(f"[batch] unknown slugs: {missing}", file=sys.stderr)
            sys.exit(1)

    if not args.force:
        skipped = [r for r in runs_to_do if _is_done(r)]
        runs_to_do = [r for r in runs_to_do if not _is_done(r)]
        for r in skipped:
            print(f"[batch] SKIP {r.slug} (already done)")

    print(f"[batch] will run {len(runs_to_do)} experiment(s):")
    for r in runs_to_do:
        print(f"  - {r.slug}")
    print()

    successes, failures = 0, 0
    t0 = time.time()
    for i, run in enumerate(runs_to_do, 1):
        print(f"\n{'#' * 70}")
        print(f"# [{i}/{len(runs_to_do)}] {run.slug}")
        print(f"{'#' * 70}")
        if _train_one(run):
            successes += 1
        else:
            failures += 1
        _rebuild_index()  # после каждого, чтобы не потерять прогресс при прерывании

    elapsed = (time.time() - t0) / 60
    print(f"\n{'=' * 70}")
    print(
        f"[batch] done in {elapsed:.1f} min: "
        f"{successes} successes, {failures} failures"
    )
    print(f"[batch] index: {INDEX_CSV}")


if __name__ == "__main__":
    main()
