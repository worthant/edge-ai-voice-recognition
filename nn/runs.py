"""
Конфигурация экспериментов для ВКР.

Каждый эксперимент = RunConfig: одна архитектура DS-CNN (filters + blocks).
Для каждой архитектуры пайплайн обучает FP32 один раз, затем делает PTQ
и QAT параллельно от того же FP32-чекпойнта.

Группы:
  GROUP_A — 15 архитектур, унаследованных от старых runs.
  GROUP_B — 20 новых, плотная сетка в «быстрой зоне» (filters 32-128).
  GROUP_C — 13 «дыр-закрывающих»: ультра-мелкие (f16/f24),
            доп. глубина для f64/f128, крупные filters с разной глубиной.

Выбор группы:
  KWS_GROUP=A python -m train_all
  KWS_GROUP=B python -m train_all
  KWS_GROUP=C python -m train_all
  python -m train_all                  # все три
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
        return RUNS_ROOT / f"{self.slug}_qat"

    @property
    def legacy_ptq_dir(self) -> Path:
        return RUNS_ROOT / f"{self.slug}_ptq"


RUNS_ROOT = config.RESULTS_DIR / "runs"
RUNS_ROOT.mkdir(parents=True, exist_ok=True)
INDEX_CSV = RUNS_ROOT / "_index.csv"


# =============================================================================
# Группа A — старые архитектуры (15)
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
    RunConfig(filters=176, blocks=2, description="very shallow"),
    RunConfig(filters=176, blocks=4, description="shallow"),
    RunConfig(filters=176, blocks=5, description="medium-shallow"),
    RunConfig(filters=176, blocks=7, description="deep"),
    RunConfig(filters=176, blocks=8, description="very deep"),
]

# =============================================================================
# Группа B — плотная сетка в «быстрой зоне» (20)
# =============================================================================
GROUP_B: list[RunConfig] = [
    RunConfig(filters=32, blocks=4, description="tiny depth=4"),
    RunConfig(filters=32, blocks=5, description="tiny depth=5"),
    RunConfig(filters=32, blocks=6, description="tiny depth=6"),
    RunConfig(filters=48, blocks=4, description="xs depth=4"),
    RunConfig(filters=48, blocks=5, description="xs depth=5"),
    RunConfig(filters=48, blocks=6, description="xs depth=6"),
    RunConfig(filters=64, blocks=3, description="s depth=3"),
    RunConfig(filters=64, blocks=4, description="s depth=4"),
    RunConfig(filters=64, blocks=5, description="s depth=5"),
    RunConfig(filters=80, blocks=4, description="s-mid depth=4"),
    RunConfig(filters=80, blocks=5, description="s-mid depth=5"),
    RunConfig(filters=80, blocks=6, description="s-mid depth=6"),
    RunConfig(filters=96, blocks=3, description="mid depth=3"),
    RunConfig(filters=96, blocks=4, description="mid depth=4"),
    RunConfig(filters=96, blocks=5, description="mid depth=5"),
    RunConfig(filters=112, blocks=4, description="mid-l depth=4"),
    RunConfig(filters=112, blocks=5, description="mid-l depth=5"),
    RunConfig(filters=112, blocks=6, description="mid-l depth=6"),
    RunConfig(filters=128, blocks=4, description="l depth=4"),
    RunConfig(filters=128, blocks=5, description="l depth=5"),
]

# =============================================================================
# Группа C — закрытие дыр (13)
#
# Зачем каждая:
#   f16/f24       — «нижняя граница» точности (модель ломается ниже 90%).
#   f64 b7/b8     — расширение глубины для маленькой ширины.
#   f128 b3/b7/b8 — вторая серия по глубине параллельно f176.
#   f160/192/224 b4/b8 — крупные filters с разной глубиной, заполняют
#                  верхний правый угол Парето.
# =============================================================================
GROUP_C: list[RunConfig] = [
    # «Нижняя граница»
    RunConfig(filters=16, blocks=4, description="ultra-tiny depth=4"),
    RunConfig(filters=16, blocks=6, description="ultra-tiny depth=6"),
    RunConfig(filters=24, blocks=5, description="ultra-xs depth=5"),
    # f64 — расширение глубины
    RunConfig(filters=64, blocks=7, description="s deep"),
    RunConfig(filters=64, blocks=8, description="s very deep"),
    # f128 — вторая серия по глубине
    RunConfig(filters=128, blocks=3, description="l depth=3"),
    RunConfig(filters=128, blocks=7, description="l deep"),
    RunConfig(filters=128, blocks=8, description="l very deep"),
    # Крупные filters с разной глубиной
    RunConfig(filters=160, blocks=4, description="xl-mid shallow"),
    RunConfig(filters=192, blocks=4, description="xl shallow"),
    RunConfig(filters=192, blocks=8, description="xl very deep"),
    RunConfig(filters=224, blocks=4, description="xxl shallow"),
    RunConfig(filters=224, blocks=8, description="xxl very deep"),
]

GROUP_D: list[RunConfig] = [
    # filters=40 — между B (32) и B (48)
    RunConfig(filters=40, blocks=4, description="xxs depth=4"),
    RunConfig(filters=40, blocks=5, description="xxs depth=5"),
    RunConfig(filters=40, blocks=6, description="xxs depth=6"),
    # filters=56 — между B (48) и B (64)
    RunConfig(filters=56, blocks=4, description="ss depth=4"),
    RunConfig(filters=56, blocks=5, description="ss depth=5"),
    RunConfig(filters=56, blocks=6, description="ss depth=6"),
    # filters=72 — между B (64) и B (80)
    RunConfig(filters=72, blocks=4, description="ss-mid depth=4"),
    RunConfig(filters=72, blocks=5, description="ss-mid depth=5"),
    RunConfig(filters=72, blocks=6, description="ss-mid depth=6"),
    # filters=88 — между B (80) и B (96)
    RunConfig(filters=88, blocks=4, description="mmid depth=4"),
    RunConfig(filters=88, blocks=5, description="mmid depth=5"),
    RunConfig(filters=88, blocks=6, description="mmid depth=6"),
    # filters=104 — между B (96) и B (112)
    RunConfig(filters=104, blocks=4, description="mmid-l depth=4"),
    RunConfig(filters=104, blocks=5, description="mmid-l depth=5"),
    RunConfig(filters=104, blocks=6, description="mmid-l depth=6"),
]

GROUP_LAST1: list[RunConfig] = [
    RunConfig(filters=224, blocks=4, description="xxl shallow"),
    RunConfig(filters=224, blocks=8, description="xxl very deep"),
    RunConfig(filters=96, blocks=4, description="mid depth=4"),
    RunConfig(filters=96, blocks=5, description="mid depth=5"),
]

GROUP_LAST2: list[RunConfig] = [
    RunConfig(filters=112, blocks=4, description="mid-l depth=4"),
    RunConfig(filters=112, blocks=5, description="mid-l depth=5"),
    RunConfig(filters=112, blocks=6, description="mid-l depth=6"),
    RunConfig(filters=128, blocks=4, description="l depth=4"),
    RunConfig(filters=128, blocks=5, description="l depth=5"),
]

GROUP_PATCH_1: list[RunConfig] = [
    RunConfig(filters=16, blocks=3, description="patch f16 b3"),
    RunConfig(filters=24, blocks=3, description="patch f24 b3"),
    RunConfig(filters=24, blocks=7, description="patch f24 b7"),
    RunConfig(filters=32, blocks=3, description="patch f32 b3"),
    RunConfig(filters=32, blocks=8, description="patch f32 b8"),
    RunConfig(filters=48, blocks=7, description="patch f48 b7"),
    RunConfig(filters=56, blocks=2, description="patch f56 b2"),
    RunConfig(filters=56, blocks=8, description="patch f56 b8"),
    RunConfig(filters=80, blocks=2, description="patch f80 b2"),
    RunConfig(filters=104, blocks=2, description="patch f104 b2"),
    RunConfig(filters=104, blocks=8, description="patch f104 b8"),
    RunConfig(filters=112, blocks=3, description="patch f112 b3"),
    RunConfig(filters=160, blocks=2, description="patch f160 b2"),
    RunConfig(filters=160, blocks=7, description="patch f160 b7"),
    RunConfig(filters=172, blocks=7, description="patch f172 b7"),
    RunConfig(filters=184, blocks=4, description="patch f184 b4"),
    RunConfig(filters=192, blocks=2, description="patch f192 b2"),
    RunConfig(filters=224, blocks=2, description="patch f224 b2"),
    RunConfig(filters=224, blocks=3, description="patch f224 b3"),
    RunConfig(filters=224, blocks=7, description="patch f224 b7"),
]

GROUP_PATCH_2: list[RunConfig] = [
    RunConfig(filters=16, blocks=5, description="patch f16 b5"),
    RunConfig(filters=16, blocks=8, description="patch f16 b8"),
    RunConfig(filters=24, blocks=6, description="patch f24 b6"),
    RunConfig(filters=40, blocks=3, description="patch f40 b3"),
    RunConfig(filters=40, blocks=7, description="patch f40 b7"),
    RunConfig(filters=64, blocks=2, description="patch f64 b2"),
    RunConfig(filters=72, blocks=3, description="patch f72 b3"),
    RunConfig(filters=72, blocks=7, description="patch f72 b7"),
    RunConfig(filters=80, blocks=3, description="patch f80 b3"),
    RunConfig(filters=88, blocks=8, description="patch f88 b8"),
    RunConfig(filters=96, blocks=7, description="patch f96 b7"),
    RunConfig(filters=112, blocks=2, description="patch f112 b2"),
    RunConfig(filters=168, blocks=4, description="patch f168 b4"),
    RunConfig(filters=168, blocks=8, description="patch f168 b8"),
    RunConfig(filters=176, blocks=3, description="patch f176 b3"),
    RunConfig(filters=184, blocks=3, description="patch f184 b3"),
    RunConfig(filters=184, blocks=5, description="patch f184 b5"),
    RunConfig(filters=184, blocks=8, description="patch f184 b8"),
    RunConfig(filters=192, blocks=5, description="patch f192 b5"),
]

GROUP_PATCH_3: list[RunConfig] = [
    RunConfig(filters=24, blocks=2, description="patch f24 b2"),
    RunConfig(filters=24, blocks=4, description="patch f24 b4"),
    RunConfig(filters=32, blocks=2, description="patch f32 b2"),
    RunConfig(filters=48, blocks=2, description="patch f48 b2"),
    RunConfig(filters=48, blocks=3, description="patch f48 b3"),
    RunConfig(filters=48, blocks=8, description="patch f48 b8"),
    RunConfig(filters=72, blocks=2, description="patch f72 b2"),
    RunConfig(filters=80, blocks=7, description="patch f80 b7"),
    RunConfig(filters=88, blocks=2, description="patch f88 b2"),
    RunConfig(filters=88, blocks=3, description="patch f88 b3"),
    RunConfig(filters=88, blocks=7, description="patch f88 b7"),
    RunConfig(filters=104, blocks=7, description="patch f104 b7"),
    RunConfig(filters=128, blocks=2, description="patch f128 b2"),
    RunConfig(filters=160, blocks=3, description="patch f160 b3"),
    RunConfig(filters=168, blocks=7, description="patch f168 b7"),
    RunConfig(filters=172, blocks=4, description="patch f172 b4"),
    RunConfig(filters=172, blocks=5, description="patch f172 b5"),
    RunConfig(filters=172, blocks=8, description="patch f172 b8"),
    RunConfig(filters=192, blocks=3, description="patch f192 b3"),
    RunConfig(filters=192, blocks=7, description="patch f192 b7"),
]

GROUP_PATCH_4: list[RunConfig] = [
    RunConfig(filters=16, blocks=2, description="patch f16 b2"),
    RunConfig(filters=16, blocks=7, description="patch f16 b7"),
    RunConfig(filters=24, blocks=8, description="patch f24 b8"),
    RunConfig(filters=32, blocks=7, description="patch f32 b7"),
    RunConfig(filters=40, blocks=2, description="patch f40 b2"),
    RunConfig(filters=40, blocks=8, description="patch f40 b8"),
    RunConfig(filters=56, blocks=3, description="patch f56 b3"),
    RunConfig(filters=56, blocks=7, description="patch f56 b7"),
    RunConfig(filters=72, blocks=8, description="patch f72 b8"),
    RunConfig(filters=80, blocks=8, description="patch f80 b8"),
    RunConfig(filters=96, blocks=2, description="patch f96 b2"),
    RunConfig(filters=96, blocks=8, description="patch f96 b8"),
    RunConfig(filters=104, blocks=3, description="patch f104 b3"),
    RunConfig(filters=112, blocks=7, description="patch f112 b7"),
    RunConfig(filters=112, blocks=8, description="patch f112 b8"),
    RunConfig(filters=160, blocks=5, description="patch f160 b5"),
    RunConfig(filters=160, blocks=8, description="patch f160 b8"),
    RunConfig(filters=168, blocks=5, description="patch f168 b5"),
    RunConfig(filters=184, blocks=7, description="patch f184 b7"),
    RunConfig(filters=224, blocks=5, description="patch f224 b5"),
]

# Объединение с дедупликацией по slug.
_all_with_duplicates = (
    GROUP_A
    + GROUP_B
    + GROUP_C
    + GROUP_D
    + GROUP_LAST1
    + GROUP_LAST2
    + GROUP_PATCH_1
    + GROUP_PATCH_2
    + GROUP_PATCH_3
    + GROUP_PATCH_4
)
_seen: set[str] = set()
ALL_RUNS: list[RunConfig] = []
for r in _all_with_duplicates:
    if r.slug not in _seen:
        _seen.add(r.slug)
        ALL_RUNS.append(r)


def selected_runs() -> list[RunConfig]:
    """Runs выбранные по KWS_GROUP."""
    group = os.environ.get("KWS_GROUP", "").upper().strip()
    if group == "A":
        return GROUP_A
    if group == "B":
        return GROUP_B
    if group == "C":
        return GROUP_C
    if group == "D":
        return GROUP_D
    if group == "LAST_1":
        return GROUP_LAST1
    if group == "LAST_2":
        return GROUP_LAST2
    if group == "PATCH_1":
        return GROUP_PATCH_1
    if group == "PATCH_2":
        return GROUP_PATCH_2
    if group == "PATCH_3":
        return GROUP_PATCH_3
    if group == "PATCH_4":
        return GROUP_PATCH_4
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
