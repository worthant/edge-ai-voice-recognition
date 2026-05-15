"""
Конфигурация всех экспериментов для ВКР.

Каждый эксперимент = RunConfig: гиперпараметры архитектуры + метод
квантизации. Slug формируется автоматически и используется как имя
папки в results/runs/<slug>/, идентификатор в _index.csv, имя файла
профиля на устройстве (profile_<slug>.csv).

Эксперименты сгруппированы по исследовательским вопросам:

  GROUP_A — SIMD-выравнивание + Pareto по filters (10 моделей):
            blocks=6, варьируем filters от 64 до 224. Накрывает SIMD-границу
            (172 не aligned, остальные aligned). Главное исследование.
  GROUP_B — PTQ vs QAT (4 модели): PTQ-параллели для 96/172/176/192.
            QAT-парные уже есть в GROUP_A. Задача 3 ВКР.
  GROUP_C — глубина сети (5 моделей, b6 уже в A): blocks 2/4/5/7/8
            на filters=176. Кривая accuracy/latency vs depth.

Итого: 10 + 4 + 5 = 19 уникальных runs.

Запуск всей серии:    python train_all.py
Запуск одного:         python train.py --slug f176_b6_qat
Сводка экспериментов:  python runs.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import config

QuantMethod = Literal["fp32", "ptq", "qat"]


@dataclass(frozen=True)
class RunConfig:
    """Параметры одного эксперимента."""

    filters: int
    blocks: int
    quant: QuantMethod
    description: str = ""

    @property
    def slug(self) -> str:
        return f"f{self.filters}_b{self.blocks}_{self.quant}"

    @property
    def is_simd_aligned(self) -> bool:
        """Кратно ли filters 8 (условие SIMD-кернела esp-nn для 1×1)."""
        return self.filters % 8 == 0

    @property
    def ds_cnn_config(self) -> dict:
        return {
            "first_conv_filters": self.filters,
            "first_conv_kernel": (10, 4),
            "first_conv_stride": (2, 2),
            "num_ds_blocks": self.blocks,
            "ds_filters": self.filters,
            "ds_kernel": (3, 3),
        }

    @property
    def run_dir(self) -> Path:
        return RUNS_ROOT / self.slug

    @property
    def fp32_keras_path(self) -> Path:
        return self.run_dir / "ds_cnn_fp32.keras"

    @property
    def tflite_path(self) -> Path:
        if self.quant == "fp32":
            return self.run_dir / "model_fp32.tflite"
        return self.run_dir / f"model_{self.quant}_int8.tflite"

    @property
    def model_data_c_path(self) -> Path:
        return self.run_dir / "model_data.c"

    @property
    def model_data_h_path(self) -> Path:
        return self.run_dir / "model_data.h"

    @property
    def meta_path(self) -> Path:
        return self.run_dir / "meta.json"


RUNS_ROOT = config.RESULTS_DIR / "runs"
RUNS_ROOT.mkdir(parents=True, exist_ok=True)
INDEX_CSV = RUNS_ROOT / "_index.csv"


# =============================================================================
# Группа A — SIMD-выравнивание + Pareto frontier по filters (10 моделей)
# =============================================================================
GROUP_A: list[RunConfig] = [
    RunConfig(filters=64, blocks=6, quant="qat", description="small, aligned"),
    RunConfig(filters=96, blocks=6, quant="qat", description="small-mid, aligned"),
    RunConfig(filters=128, blocks=6, quant="qat", description="mid, aligned"),
    RunConfig(
        filters=160, blocks=6, quant="qat", description="aligned, below baseline"
    ),
    RunConfig(
        filters=168,
        blocks=6,
        quant="qat",
        description="aligned, slightly below baseline",
    ),
    RunConfig(
        filters=172,
        blocks=6,
        quant="qat",
        description="BASELINE: Hello Edge DS-CNN-M (NOT aligned)",
    ),
    RunConfig(
        filters=176, blocks=6, quant="qat", description="aligned, just above baseline"
    ),
    RunConfig(
        filters=184, blocks=6, quant="qat", description="aligned, above baseline"
    ),
    RunConfig(
        filters=192, blocks=6, quant="qat", description="aligned, above baseline"
    ),
    RunConfig(
        filters=224, blocks=6, quant="qat", description="large, aligned (top of range)"
    ),
]

# =============================================================================
# Группа B — PTQ vs QAT (4 модели)
# =============================================================================
GROUP_B: list[RunConfig] = [
    RunConfig(filters=96, blocks=6, quant="ptq", description="PTQ on small model"),
    RunConfig(
        filters=172,
        blocks=6,
        quant="ptq",
        description="PTQ on baseline (172, not aligned)",
    ),
    RunConfig(
        filters=176, blocks=6, quant="ptq", description="PTQ on aligned baseline"
    ),
    RunConfig(
        filters=192, blocks=6, quant="ptq", description="PTQ on larger aligned model"
    ),
]

# =============================================================================
# Группа C — глубина сети (5 моделей)
# =============================================================================
GROUP_C: list[RunConfig] = [
    RunConfig(filters=176, blocks=2, quant="qat", description="very shallow"),
    RunConfig(filters=176, blocks=4, quant="qat", description="shallow"),
    RunConfig(filters=176, blocks=5, quant="qat", description="medium-shallow"),
    RunConfig(filters=176, blocks=7, quant="qat", description="deep"),
    RunConfig(filters=176, blocks=8, quant="qat", description="very deep"),
]

# Объединение с дедупликацией по slug.
_all_with_duplicates = GROUP_A + GROUP_B + GROUP_C
_seen: set[str] = set()
ALL_RUNS: list[RunConfig] = []
for r in _all_with_duplicates:
    if r.slug not in _seen:
        _seen.add(r.slug)
        ALL_RUNS.append(r)


def find_run(slug: str) -> RunConfig:
    for r in ALL_RUNS:
        if r.slug == slug:
            return r
    available = "\n  ".join(r.slug for r in ALL_RUNS)
    raise ValueError(f"Unknown slug '{slug}'. Available:\n  {available}")


def print_summary() -> None:
    print(f"Total runs: {len(ALL_RUNS)}\n")
    print(
        f"{'#':<3} {'SLUG':<22} {'FILT':<5} {'BLK':<4} "
        f"{'QUANT':<6} {'%8':<4} DESCRIPTION"
    )
    print("-" * 100)
    for i, r in enumerate(ALL_RUNS, 1):
        aligned = "yes" if r.is_simd_aligned else "NO "
        print(
            f"{i:<3} {r.slug:<22} {r.filters:<5} {r.blocks:<4} "
            f"{r.quant:<6} {aligned:<4} {r.description}"
        )


if __name__ == "__main__":
    print_summary()
