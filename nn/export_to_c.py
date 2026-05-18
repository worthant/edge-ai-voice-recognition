"""
Экспорт .tflite в C-массив для TFLite Micro.

Slug формат: <base>_<qat|ptq>, например f172_b6_qat / f176_b2_ptq.

Источник .tflite (в порядке приоритета):
  1. results/merged_runs/<slug>/model_<variant>_int8.tflite
     (legacy: папка-с-суффиксом, она же содержит и model_data.c)
  2. results/merged_runs/<base>/model_<variant>_int8.tflite
     (new: одна папка на архитектуру, оба варианта рядом)

Выход:
  results/merged_runs/<slug>/model_data.{c,h}

CMake (src/CMakeLists.txt) читает model_data.c по тому же пути,
а platformio.ini задаёт slug через custom_model_slug.

Запуск:
    python -m export_to_c --slug f172_b6_qat
    python -m export_to_c --all                # экспортирует все доступные
    python -m export_to_c --input <p.tflite> --output <dir>   # legacy
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import config

MERGED_RUNS = config.RESULTS_DIR / "merged_runs"
VARIANTS = ("qat", "ptq")


HEADER_TEMPLATE = """\
// Автоматически сгенерировано export_to_c.py
// Slug: {slug}
// Источник: {src}
// Сгенерировано: {ts}
// Размер: {size} байт
// НЕ РЕДАКТИРОВАТЬ

#ifndef NN_MODEL_DATA_H_
#define NN_MODEL_DATA_H_

#define MODEL_SLUG "{slug}"

#ifdef __cplusplus
extern "C" {{
#endif

extern const unsigned int g_model_data_size;
extern const unsigned char g_model_data[];

#ifdef __cplusplus
}}
#endif

#endif  // NN_MODEL_DATA_H_
"""


SOURCE_HEADER = """\
// Автоматически сгенерировано export_to_c.py
// Slug: {slug}
// Источник: {src}
// Сгенерировано: {ts}
// Размер: {size} байт
// НЕ РЕДАКТИРОВАТЬ

#include "model_data.h"

__attribute__((aligned(16))) const unsigned char g_model_data[] = {{
"""

SOURCE_FOOTER = """\
}};
const unsigned int g_model_data_size = {size}u;
"""


def _bytes_to_c_array(data: bytes, per_line: int = 12) -> str:
    lines: list[str] = []
    for i in range(0, len(data), per_line):
        chunk = data[i : i + per_line]
        hex_bytes = ", ".join(f"0x{b:02x}" for b in chunk)
        lines.append(f"    {hex_bytes},")
    if lines:
        lines[-1] = lines[-1].rstrip(",")
    return "\n".join(lines)


def _split_slug(slug: str) -> tuple[str, str]:
    """Возвращает (base, variant). Бросает ValueError если суффикс не qat/ptq."""
    for v in VARIANTS:
        suffix = f"_{v}"
        if slug.endswith(suffix) and len(slug) > len(suffix):
            return slug[: -len(suffix)], v
    raise ValueError(
        f"Slug '{slug}' must end with _qat or _ptq "
        f"(e.g. f172_b6_qat). Variant chooses which tflite to embed."
    )


def resolve_tflite(slug: str) -> tuple[Path, str, str]:
    """Находит .tflite для slug.

    Возвращает (tflite_path, base, variant).
    """
    base, variant = _split_slug(slug)
    fname = f"model_{variant}_int8.tflite"

    legacy = MERGED_RUNS / slug / fname  # папка с суффиксом
    new = MERGED_RUNS / base / fname  # новая, общая папка

    for cand in (legacy, new):
        if cand.exists():
            return cand, base, variant

    raise FileNotFoundError(
        f"No tflite for slug='{slug}'. Tried:\n  {legacy}\n  {new}"
    )


def export(tflite_path: Path, out_dir: Path, slug: str) -> None:
    if not tflite_path.exists():
        raise FileNotFoundError(f"No file: {tflite_path}")
    out_dir.mkdir(parents=True, exist_ok=True)

    data = tflite_path.read_bytes()
    size = len(data)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    header = HEADER_TEMPLATE.format(slug=slug, src=tflite_path.name, ts=ts, size=size)
    source = (
        SOURCE_HEADER.format(slug=slug, src=tflite_path.name, ts=ts, size=size)
        + _bytes_to_c_array(data)
        + "\n"
        + SOURCE_FOOTER.format(size=size)
    )

    (out_dir / "model_data.h").write_text(header)
    (out_dir / "model_data.c").write_text(source)
    print(
        f"[export_to_c] {slug}: {tflite_path.relative_to(config.RESULTS_DIR)} "
        f"({size / 1024:.1f} KB) → {out_dir.relative_to(config.RESULTS_DIR)}/"
    )


def _export_slug(slug: str) -> None:
    tflite, _base, _variant = resolve_tflite(slug)
    out_dir = MERGED_RUNS / slug
    export(tflite, out_dir, slug)


def _discover_all_slugs() -> list[str]:
    """Все slug, для которых найдётся .tflite в merged_runs/.

    Для каждой папки в merged_runs/:
      - если её имя уже оканчивается на _qat/_ptq — добавляем сам slug;
      - иначе (новый формат) — добавляем <name>_qat и/или <name>_ptq,
        если соответствующий .tflite присутствует.
    """
    slugs: set[str] = set()
    if not MERGED_RUNS.exists():
        return []
    for d in sorted(MERGED_RUNS.iterdir()):
        if not d.is_dir() or d.name.startswith(("_", ".")):
            continue
        name = d.name
        # legacy: уже с суффиксом
        legacy_match = False
        for v in VARIANTS:
            if name.endswith(f"_{v}"):
                if (d / f"model_{v}_int8.tflite").exists():
                    slugs.add(name)
                legacy_match = True
                break
        if legacy_match:
            continue
        # new: пробуем оба варианта
        for v in VARIANTS:
            if (d / f"model_{v}_int8.tflite").exists():
                slugs.add(f"{name}_{v}")
    return sorted(slugs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slug", help="Slug формата <base>_<qat|ptq>")
    ap.add_argument(
        "--all", action="store_true", help="Экспортировать все slug из merged_runs/"
    )
    ap.add_argument("--input", type=Path, help="Legacy: путь к .tflite")
    ap.add_argument(
        "--output", type=Path, help="Legacy: каталог вывода (только с --input)"
    )
    args = ap.parse_args()

    if args.all:
        slugs = _discover_all_slugs()
        if not slugs:
            print(f"[export_to_c] no models found under {MERGED_RUNS}", file=sys.stderr)
            sys.exit(1)
        print(f"[export_to_c] exporting {len(slugs)} slugs")
        failures: list[tuple[str, str]] = []
        for s in slugs:
            try:
                _export_slug(s)
            except (FileNotFoundError, ValueError) as e:
                failures.append((s, str(e)))
                print(f"[export_to_c] FAIL {s}: {e}", file=sys.stderr)
        print(f"[export_to_c] done: {len(slugs) - len(failures)}/{len(slugs)} ok")
        if failures:
            sys.exit(1)
        return

    if args.slug:
        _export_slug(args.slug)
        return

    if args.input and args.output:
        export(args.input, args.output, slug=args.input.stem)
        return

    ap.error("Specify --slug, --all, OR (--input AND --output)")


if __name__ == "__main__":
    main()
