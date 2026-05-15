"""
Экспорт .tflite в C-массив для TFLite Micro.

Изменения:
- Принимает --slug, читает model.tflite из runs/<slug>/.
- Записывает model_data.{c,h} в ту же папку (runs/<slug>/).
- Прошивка подключает model_data из runs/<MODEL_SLUG>/ через
  CMake-параметр (см. src/CMakeLists.txt).

Старый --input / --output режим тоже поддерживается для legacy-вызовов.

Запуск:
    python -m export_to_c --slug f176_b6_qat
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

from runs import RunConfig, find_run

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


def export(tflite_path: Path, out_dir: Path, slug: str = "") -> None:
    if not tflite_path.exists():
        raise FileNotFoundError(f"No file: {tflite_path}")
    out_dir.mkdir(parents=True, exist_ok=True)

    data = tflite_path.read_bytes()
    size = len(data)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    header = HEADER_TEMPLATE.format(
        slug=slug or tflite_path.stem,
        src=tflite_path.name,
        ts=ts,
        size=size,
    )
    source = (
        SOURCE_HEADER.format(
            slug=slug or tflite_path.stem, src=tflite_path.name, ts=ts, size=size
        )
        + _bytes_to_c_array(data)
        + "\n"
        + SOURCE_FOOTER.format(size=size)
    )

    (out_dir / "model_data.h").write_text(header)
    (out_dir / "model_data.c").write_text(source)
    print(f"[export_to_c] {tflite_path.name} ({size/1024:.1f} KB) → {out_dir}/")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slug", help="RunConfig slug (preferred)")
    ap.add_argument("--input", type=Path, help="Legacy: path to .tflite")
    ap.add_argument(
        "--output", type=Path, help="Legacy: output directory (only with --input)"
    )
    args = ap.parse_args()

    if args.slug:
        run = find_run(args.slug)
        if not run.tflite_path.exists():
            print(
                f"[export_to_c] ERROR: no tflite for {args.slug} "
                f"at {run.tflite_path}",
                file=sys.stderr,
            )
            sys.exit(1)
        export(run.tflite_path, run.run_dir, slug=run.slug)
    elif args.input and args.output:
        export(args.input, args.output, slug=args.input.stem)
    else:
        ap.error("Specify --slug OR (--input AND --output)")


if __name__ == "__main__":
    main()
