"""
Объединение результатов обучения с нескольких аккаунтов.

Использование:
  1. Скачать с каждого аккаунта папку results/runs/ как zip.
  2. Распаковать в отдельные папки, например:
       ~/Downloads/runs_acc_A/
       ~/Downloads/runs_acc_B/
       ~/Downloads/runs_acc_C/
       ~/Downloads/runs_acc_D/
  3. Запустить:
       python merge_results.py \
           ~/Downloads/runs_acc_A \
           ~/Downloads/runs_acc_B \
           ~/Downloads/runs_acc_C \
           ~/Downloads/runs_acc_D \
           --out ~/merged_runs

Скрипт:
  - Сливает все подпапки архитектур (f32_b4/, f48_b5/, ...) в --out.
  - При конфликте (одна архитектура на двух аккаунтах) оставляет более
    полную версию: где больше файлов и где meta.json содержит больше
    ключей (ptq+qat > только qat > только fp32).
  - Пересобирает _index.csv из всех meta.json в --out.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path


def _score_arch_dir(d: Path) -> int:
    """Чем выше — тем полнее набор файлов в папке архитектуры."""
    score = 0
    for fname in (
        "ds_cnn_fp32.keras",
        "model_ptq_int8.tflite",
        "model_qat_int8.tflite",
        "cm_fp32.png",
        "cm_ptq.png",
        "cm_qat.png",
        "training.csv",
        "qat.csv",
    ):
        if (d / fname).exists():
            score += 1
    meta = d / "meta.json"
    if meta.exists():
        try:
            data = json.loads(meta.read_text())
            score += len(data)  # больше ключей — лучше
        except Exception:
            pass
    return score


def _merge_dirs(sources: list[Path], out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)

    # Собираем кандидатов: slug -> [(score, src_dir), ...]
    candidates: dict[str, list[tuple[int, Path]]] = {}
    for src in sources:
        if not src.exists():
            print(f"[merge] WARN: {src} does not exist, skip")
            continue
        for arch_dir in src.iterdir():
            if not arch_dir.is_dir():
                continue
            if arch_dir.name.startswith("_") or arch_dir.name.startswith("."):
                continue  # _index.csv, .DS_Store, ...
            score = _score_arch_dir(arch_dir)
            candidates.setdefault(arch_dir.name, []).append((score, arch_dir))

    # Для каждой архитектуры выбираем лучшую версию
    for slug, options in sorted(candidates.items()):
        options.sort(key=lambda x: x[0], reverse=True)
        best_score, best_dir = options[0]
        dst = out / slug
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(best_dir, dst)

        note = ""
        if len(options) > 1:
            scores = [s for s, _ in options]
            note = f"  (chose score={best_score} from {scores})"
        print(f"[merge] {slug}{note}")


def _rebuild_index(out: Path) -> None:
    """Собирает _index.csv из всех meta.json в out с единым набором колонок."""
    rows = []
    for arch_dir in sorted(out.iterdir()):
        if not arch_dir.is_dir():
            continue
        meta = arch_dir / "meta.json"
        if not meta.exists():
            continue
        try:
            data = json.loads(meta.read_text())
            rows.append(data)
        except Exception as e:
            print(f"[merge] skip malformed {meta}: {e}", file=sys.stderr)

    if not rows:
        print("[merge] no metas found")
        return

    # Унифицированный список колонок: фиксированный порядок для основных,
    # затем всё остальное в алфавитном порядке для стабильности
    preferred_order = [
        "slug",
        "filters",
        "blocks",
        "description",
        "simd_aligned",
        "params",
        "fp32_acc_pct",
        "fp32_size_kb",
        "ptq_acc_pct",
        "ptq_size_kb",
        "qat_acc_pct",
        "qat_size_kb",
        "train_date_utc",
        "updated_utc",
    ]
    all_keys: set[str] = set()
    for r in rows:
        all_keys.update(r.keys())
    other = sorted(all_keys - set(preferred_order))
    columns = [c for c in preferred_order if c in all_keys] + other

    csv_path = out / "_index.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in columns})

    print(f"\n[merge] wrote {csv_path} ({len(rows)} rows, {len(columns)} columns)")
    print(f"[merge] columns: {columns}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("sources", nargs="+", help="Folders with runs/ from each account")
    ap.add_argument("--out", required=True, help="Output merged folder")
    args = ap.parse_args()

    sources = [Path(s).expanduser().resolve() for s in args.sources]
    out = Path(args.out).expanduser().resolve()

    print(f"[merge] sources: {sources}")
    print(f"[merge] output:  {out}\n")

    _merge_dirs(sources, out)
    _rebuild_index(out)


if __name__ == "__main__":
    main()
