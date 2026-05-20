"""
bulk_profile_to_stats.py

Прогоняет агрегацию по всем profile_*.csv в measurements/best/profile/
и сохраняет соответствующие stats_*.csv в measurements/best/stats/.

Логика идентична тому, что в исходном profile_analysis.py:
  - отбрасываем warmup (run_id < 0)
  - агрегируем по (op_index, op_tag)
  - считаем mean/std/min/max/median времени слоя в мс
  - считаем pct_time от общего среднего инференса

Запуск:
    python bulk_profile_to_stats.py
"""

from __future__ import annotations
from pathlib import Path

import pandas as pd

# В Colab пути будут другие — поправь корень, если нужно
PROFILE_DIR = Path("../measurements/best/profile")
STATS_DIR   = Path("../measurements/best/stats")


def profile_to_stats(profile_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(profile_csv, comment="#")
    df = df[df.run_id >= 0].copy()
    df["ticks_ms"] = df.ticks_us / 1000.0

    agg = (
        df.groupby(["op_index", "op_tag"])
          .agg(
              mean_ms   = ("ticks_ms", "mean"),
              std_ms    = ("ticks_ms", "std"),
              min_ms    = ("ticks_ms", "min"),
              max_ms    = ("ticks_ms", "max"),
              median_ms = ("ticks_ms", "median"),
          )
          .reset_index()
          .sort_values("op_index")
    )
    total = agg.mean_ms.sum()
    agg["pct_time"] = 100.0 * agg.mean_ms / total
    return agg


def main() -> None:
    STATS_DIR.mkdir(parents=True, exist_ok=True)

    profiles = sorted(PROFILE_DIR.glob("profile_*.csv"))
    if not profiles:
        print(f"нет файлов в {PROFILE_DIR}")
        return

    print(f"обрабатываю {len(profiles)} файлов\n")
    for p in profiles:
        # profile_f104_b5_ptq.csv -> stats_f104_b5_ptq.csv
        model_id = p.stem.replace("profile_", "")
        out = STATS_DIR / f"stats_{model_id}.csv"
        agg = profile_to_stats(p)
        agg.to_csv(out, index=False, float_format="%.3f")
        total = agg.mean_ms.sum()
        print(f"  {model_id:20s}  слоёв={len(agg):2d}  total={total:7.1f} ms  -> {out.name}")

    print(f"\nсохранено в {STATS_DIR}")


if __name__ == "__main__":
    main()
