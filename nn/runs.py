"""
Конфигурация всех экспериментов для ВКР.

Каждый эксперимент = RunConfig: гиперпараметры архитектуры + метод
квантизации. Slug формируется автоматически из параметров и используется
как имя папки в results/runs/<slug>/, как идентификатор в _index.csv,
как имя файла профиля бенчмарка на устройстве (profile_<slug>.csv).

Эксперименты сгруппированы по исследовательским вопросам:

  GROUP_A — SIMD-выравнивание (главная находка): фиксируем blocks=6,
            варьируем filters так чтобы пересечь границу % 8 == 0.
            Все QAT.
  GROUP_B — PTQ vs QAT: на двух конфигурациях (baseline 172 и aligned 176)
            делаем по два метода квантизации.
  GROUP_C — глубина сети: 4, 6, 8 блоков на f176 + QAT.

Запуск всей серии:
    python -m train_all
Запуск одного:
    python -m train --slug f176_b6_qat
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import config

QuantMethod = Literal["fp32", "ptq", "qat"]


@dataclass(frozen=True)
class RunConfig:
    """Параметры одного эксперимента."""

    filters: int  # число фильтров и в stem-conv, и в DS-блоках
    blocks: int  # число DS-Conv блоков
    quant: QuantMethod  # метод квантизации
    description: str = ""  # короткое описание для отчётов

    @property
    def slug(self) -> str:
        """Машинно-читаемый идентификатор: f176_b6_qat."""
        return f"f{self.filters}_b{self.blocks}_{self.quant}"

    @property
    def is_simd_aligned(self) -> bool:
        """Кратно ли число фильтров 8 (условие SIMD-кернела esp-nn для 1×1)."""
        return self.filters % 8 == 0

    @property
    def ds_cnn_config(self) -> dict:
        """Конфиг для build_ds_cnn(), совместим с прежним DS_CNN_CONFIG."""
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
        """results/runs/<slug>/ — все артефакты эксперимента живут здесь."""
        return RUNS_ROOT / self.slug

    @property
    def fp32_keras_path(self) -> Path:
        return self.run_dir / "ds_cnn_fp32.keras"

    @property
    def tflite_path(self) -> Path:
        """Финальная квантованная (или fp32) модель для деплоя."""
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


# Корень для всех экспериментов
RUNS_ROOT = config.RESULTS_DIR / "runs"
RUNS_ROOT.mkdir(parents=True, exist_ok=True)

# Сводный индекс всех runs (sklearn-style: одна строка на эксперимент)
INDEX_CSV = RUNS_ROOT / "_index.csv"


# =============================================================================
# Группа A — SIMD-выравнивание
# =============================================================================
# Главное исследование. blocks=6 фиксирован (=Medium из Hello Edge).
# Варьируем filters вокруг 172 (baseline) с пересечением границы % 8.
# Из этих 6 точек выйдет основной график "латентность vs filters".
GROUP_A: list[RunConfig] = [
    RunConfig(
        filters=160,
        blocks=6,
        quant="qat",
        description="SIMD aligned, weaker than baseline",
    ),
    RunConfig(
        filters=168, blocks=6, quant="qat", description="SIMD aligned, slightly weaker"
    ),
    RunConfig(
        filters=172,
        blocks=6,
        quant="qat",
        description="BASELINE: Hello Edge DS-CNN-M (NOT aligned)",
    ),
    RunConfig(
        filters=176,
        blocks=6,
        quant="qat",
        description="SIMD aligned, slightly stronger",
    ),
    RunConfig(filters=184, blocks=6, quant="qat", description="SIMD aligned, stronger"),
    RunConfig(
        filters=192,
        blocks=6,
        quant="qat",
        description="SIMD aligned, strongest in series",
    ),
]

# =============================================================================
# Группа B — PTQ vs QAT
# =============================================================================
# Прямое сравнение методов квантизации по задаче 3 ВКР.
# Берём baseline (172) и aligned (176), для каждого делаем PTQ и QAT.
# Две из четырёх моделей (172_qat, 176_qat) уже есть в Группе A — не дублируем.
GROUP_B: list[RunConfig] = [
    RunConfig(
        filters=172,
        blocks=6,
        quant="ptq",
        description="PTQ on baseline, paired with f172_b6_qat",
    ),
    RunConfig(
        filters=176,
        blocks=6,
        quant="ptq",
        description="PTQ on aligned, paired with f176_b6_qat",
    ),
]

# =============================================================================
# Группа C — глубина сети
# =============================================================================
# Опциональное расширение: как влияет число DS-блоков? Фиксируем filters=176
# (aligned), варьируем blocks. Точка blocks=6 уже есть в Группе A.
GROUP_C: list[RunConfig] = [
    RunConfig(
        filters=176,
        blocks=4,
        quant="qat",
        description="Shallower variant of aligned baseline",
    ),
    RunConfig(
        filters=176,
        blocks=8,
        quant="qat",
        description="Deeper variant of aligned baseline",
    ),
]

# Объединённый порядок выполнения. Если хотим прервать на половине —
# чтобы хотя бы Группа A была закрыта, она идёт первой.
ALL_RUNS: list[RunConfig] = GROUP_A + GROUP_B + GROUP_C

# Дедупликация на случай если в группах одна и та же модель встретится дважды.
_seen: set[str] = set()
ALL_RUNS = [r for r in ALL_RUNS if not (r.slug in _seen or _seen.add(r.slug))]


def find_run(slug: str) -> RunConfig:
    """Найти RunConfig по slug; используется из train.py / quantize_*.py / export."""
    for r in ALL_RUNS:
        if r.slug == slug:
            return r
    available = "\n  ".join(r.slug for r in ALL_RUNS)
    raise ValueError(f"Unknown slug '{slug}'. Available:\n  {available}")


def print_summary() -> None:
    """Печатает сводку: что будет обучено, в каком порядке, итого моделей."""
    print(f"Total runs: {len(ALL_RUNS)}\n")
    print(
        f"{'#':<3} {'SLUG':<20} {'FILTERS':<8} {'BLOCKS':<7} "
        f"{'QUANT':<6} {'%8':<4} DESCRIPTION"
    )
    print("-" * 100)
    for i, r in enumerate(ALL_RUNS, 1):
        aligned = "yes" if r.is_simd_aligned else "NO "
        print(
            f"{i:<3} {r.slug:<20} {r.filters:<8} {r.blocks:<7} "
            f"{r.quant:<6} {aligned:<4} {r.description}"
        )


if __name__ == "__main__":
    print_summary()
