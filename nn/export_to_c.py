"""
Конвертация .tflite -> model_data.cc / model_data.h для TFLite Micro.
Формат совместим с tflite-micro examples (alignas(16), const uint8_t).
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import config


HEADER_TEMPLATE = """\
// Автоматически сгенерировано export_to_c.py
// Источник: {src}
// Сгенерировано: {ts}
// Размер: {size} байт

#ifndef NN_MODEL_DATA_H_
#define NN_MODEL_DATA_H_

#include <cstdint>

extern const unsigned int g_model_data_size;
extern const unsigned char g_model_data[];

#endif  // NN_MODEL_DATA_H_
"""


SOURCE_HEADER = """\
// Автоматически сгенерировано export_to_c.py
// Источник: {src}
// Сгенерировано: {ts}
// Размер: {size} байт
// НЕ редактируйте руками — будет перезаписано.

#include "model_data.h"

alignas(16) const unsigned char g_model_data[] = {{
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
    # убрать последнюю запятую
    if lines:
        lines[-1] = lines[-1].rstrip(",")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input",
        type=Path,
        default=config.QAT_TFLITE,
        help="Путь к .tflite файлу (по умолчанию: QAT модель)",
    )
    ap.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Директория для model_data.cc и model_data.h",
    )
    args = ap.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Нет файла {args.input}")
    args.output.mkdir(parents=True, exist_ok=True)

    data = args.input.read_bytes()
    size = len(data)
    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

    header = HEADER_TEMPLATE.format(src=args.input.name, ts=ts, size=size)
    source = (
        SOURCE_HEADER.format(src=args.input.name, ts=ts, size=size)
        + _bytes_to_c_array(data)
        + "\n"
        + SOURCE_FOOTER.format(size=size)
    )

    (args.output / "model_data.h").write_text(header)
    (args.output / "model_data.cc").write_text(source)

    print(f"[export_to_c] {args.input} ({size / 1024:.1f} KB) -> {args.output}")
    print(f"[export_to_c] model_data.cc: {len(source)} символов")
    # Грубая оценка RAM footprint: tensor arena + model data в flash
    print(f"[export_to_c] Оценка:")
    print(f"  Flash (model_data):        ~{size / 1024:.1f} KB")
    print(
        f"  RAM (tensor arena):        ~60 KB (начать с этого, корректировать после ESP_LOGI)"
    )
    print(
        f"  RAM (MFCC + audio buf):    ~8 KB (1с×int16 = 32KB если хранить целиком; ≤8KB stream)"
    )


if __name__ == "__main__":
    main()
