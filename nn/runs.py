"""
Конфигурация экспериментов для ВКР.

Каждый эксперимент = RunConfig: одна архитектура DS-CNN (filters + blocks).
Для каждой архитектуры пайплайн обучает FP32 один раз, затем делает PTQ
и QAT параллельно от того же FP32-чекпойнта. Это даёт честное сравнение
методов квантизации.

Слаг архитектуры: f<FILT>_b<BLK>  (без суффикса метода квантизации).
Метрики хранятся в одном meta.json со всеми ключами:
  fp32_acc_pct, ptq_acc_pct, qat_acc_pct,
  fp32_size_kb, ptq_size_kb, qat_size_kb,
  params, simd_aligned, ...

Группы:
  GROUP_A — 19 архитектур, унаследованных от старых runs (старые tflite
            подхватываем; нужно только досчитать PTQ/QAT там где их нет).
  GROUP_B — 21 новая архитектура: плотная сетка в «быстрой зоне»
            (filters 32-128) + контрольные точки.

Для двух колаб-аккаунтов используем переменную окружения KWS_GROUP:
  KWS_GROUP=A  python -m train_all   # обучаем GROUP_A
  KWS_GROUP=B  python -m train_all   # обучаем GROUP_B
  (без переменной — оба).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import config


@dataclass(frozen=True)
class RunConfig:
    """Одна архитектура DS-CNN."""

    filters: int
    blocks: int
    description: str = ""

    @property
    def slug(self) -> str:
        return f"f{self.filters}_b{self.blocks}"

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
    def ptq_tflite_path(self) -> Path:
        return self.run_dir / "model_ptq_int8.tflite"

    @property
    def qat_tflite_path(self) -> Path:
        return self.run_dir / "model_qat_int8.tflite"

    @property
    def meta_path(self) -> Path:
        return self.run_dir / "meta.json"

    @property
    def legacy_qat_dir(self) -> Path:
        """Путь к старой папке вида f<F>_b<B>_qat/ для подхвата FP32-весов."""
        return RUNS_ROOT / f"{self.slug}_qat"

    @property
    def legacy_ptq_dir(self) -> Path:
        return RUNS_ROOT / f"{self.slug}_ptq"


RUNS_ROOT = config.RESULTS_DIR / "runs"
RUNS_ROOT.mkdir(parents=True, exist_ok=True)
INDEX_CSV = RUNS_ROOT / "_index.csv"


# =============================================================================
# Группа A — 19 архитектур, что уже обучены в _old_19_runs.
# Стратегия: переиспользуем существующие FP32-чекпойнты,
# для каждой архитектуры досчитываем недостающие PTQ/QAT.
# =============================================================================
GROUP_A: list[RunConfig] = [
    RunConfig(filters=64, blocks=6, description="small, aligned"),
    RunConfig(filters=96, blocks=6, description="small-mid, aligned"),
    RunConfig(filters=128, blocks=6, description="mid, aligned"),
    RunConfig(filters=160, blocks=6, description="aligned, below baseline"),
    RunConfig(filters=168, blocks=6, description="aligned, slightly below baseline"),
    RunConfig(filters=172, blocks=6, description="Hello Edge baseline (not aligned)"),
    RunConfig(filters=176, blocks=6, description="aligned, just above baseline"),
    RunConfig(filters=184, blocks=6, description="aligned, above baseline"),
    RunConfig(filters=192, blocks=6, description="aligned, above baseline"),
    RunConfig(filters=224, blocks=6, description="large, aligned (top of range)"),
    # f176 — варианты глубины
    RunConfig(filters=176, blocks=2, description="very shallow"),
    RunConfig(filters=176, blocks=4, description="shallow"),
    RunConfig(filters=176, blocks=5, description="medium-shallow"),
    RunConfig(filters=176, blocks=7, description="deep"),
    RunConfig(filters=176, blocks=8, description="very deep"),
]

# =============================================================================
# Группа B — 21 новая архитектура, плотная сетка в «быстрой зоне».
# Гипотеза: Парето-оптимальные модели для ESP32-S3 лежат при filters ≤ 128.
# Для каждой архитектуры обучаем FP32 с нуля + PTQ + QAT.
# =============================================================================
GROUP_B: list[RunConfig] = [
    # filters=32, тiny
    RunConfig(filters=32, blocks=4, description="tiny depth=4"),
    RunConfig(filters=32, blocks=5, description="tiny depth=5"),
    RunConfig(filters=32, blocks=6, description="tiny depth=6"),
    # filters=48, xs
    RunConfig(filters=48, blocks=4, description="xs depth=4"),
    RunConfig(filters=48, blocks=5, description="xs depth=5"),
    RunConfig(filters=48, blocks=6, description="xs depth=6"),
    # filters=64, s (b6 в group A)
    RunConfig(filters=64, blocks=3, description="s depth=3"),
    RunConfig(filters=64, blocks=4, description="s depth=4"),
    RunConfig(filters=64, blocks=5, description="s depth=5"),
    # filters=80, s-mid
    RunConfig(filters=80, blocks=4, description="s-mid depth=4"),
    RunConfig(filters=80, blocks=5, description="s-mid depth=5"),
    RunConfig(filters=80, blocks=6, description="s-mid depth=6"),
    # filters=96, mid (b6 в group A)
    RunConfig(filters=96, blocks=3, description="mid depth=3"),
    RunConfig(filters=96, blocks=4, description="mid depth=4"),
    RunConfig(filters=96, blocks=5, description="mid depth=5"),
    # filters=112, mid-l
    RunConfig(filters=112, blocks=4, description="mid-l depth=4"),
    RunConfig(filters=112, blocks=5, description="mid-l depth=5"),
    RunConfig(filters=112, blocks=6, description="mid-l depth=6"),
    # filters=128, l (b6 в group A)
    RunConfig(filters=128, blocks=4, description="l depth=4"),
    RunConfig(filters=128, blocks=5, description="l depth=5"),
]

# Объединение с дедупликацией по slug.
_all_with_duplicates = GROUP_A + GROUP_B
_seen: set[str] = set()
ALL_RUNS: list[RunConfig] = []
for r in _all_with_duplicates:
    if r.slug not in _seen:
        _seen.add(r.slug)
        ALL_RUNS.append(r)


def selected_runs() -> list[RunConfig]:
    """Runs выбранные по KWS_GROUP (A, B, или оба если переменная не задана)."""
    group = os.environ.get("KWS_GROUP", "").upper().strip()
    if group == "A":
        return GROUP_A
    if group == "B":
        return GROUP_B
    return ALL_RUNS


def find_run(slug: str) -> RunConfig:
    for r in ALL_RUNS:
        if r.slug == slug:
            return r
    available = "\n  ".join(r.slug for r in ALL_RUNS)
    raise ValueError(f"Unknown slug '{slug}'. Available:\n  {available}")


def print_summary() -> None:
    sel = selected_runs()
    group = os.environ.get("KWS_GROUP", "ALL")
    print(f"Group: {group}  ({len(sel)} runs)\n")
    print(f"{'#':<3} {'SLUG':<14} {'FILT':<5} {'BLK':<4} {'%8':<4} DESCRIPTION")
    print("-" * 80)
    for i, r in enumerate(sel, 1):
        aligned = "yes" if r.is_simd_aligned else "NO "
        print(
            f"{i:<3} {r.slug:<14} {r.filters:<5} {r.blocks:<4} "
            f"{aligned:<4} {r.description}"
        )


if __name__ == "__main__":
    print_summary()
