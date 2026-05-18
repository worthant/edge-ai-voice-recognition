"""
Мердж 5 CSV файлов в один итоговый all_models_merged.csv.

Источники:
    all_models_final.csv  — базовый, уже нормализованный (старые + первые волны)
    _index.csv            — Colab аккаунт 1 (PATCH_1)
    _index_1_.csv         — Colab аккаунт 2 (PATCH_2)
    _index_2_.csv         — Colab аккаунт 3 (PATCH_3)
    _index__1_.csv        — Colab аккаунт 4 (PATCH_4)

Логика:
  1. Конкатенируем все CSV в один длинный DataFrame.
  2. Группируем по (filters, blocks).
  3. Внутри группы консолидируем колонки:
       - fp32_acc_pct, ptq_acc_pct, qat_acc_pct, int8_acc_pct: max()
       - fp32_size_kb, ptq_size_kb, qat_size_kb, int8_size_kb: первое не-NaN
       - params, slug, description, simd_aligned: первое не-NaN
       - даты: max()
  4. Сохраняем результат + лог: сколько ячеек, сколько источников per cell.

Запуск:
    python merge_csvs.py
"""

import pandas as pd
import numpy as np
from pathlib import Path

# === Настрой пути под себя ===
INPUT_DIR = Path(".")  # папка с CSV
FILES = {
    "orig": INPUT_DIR / "all_models_a-d_groups.csv",
    "p1": INPUT_DIR / "_index1.csv",
    "p2": INPUT_DIR / "_index2.csv",
    "p3": INPUT_DIR / "_index3.csv",
    "p4": INPUT_DIR / "_index4.csv",
}
OUTPUT_CSV = INPUT_DIR / "all_models_short.csv"
LOG_CSV = INPUT_DIR / "merge_log.csv"


def main():
    # ---------- 1. Загружаем все файлы, добавляем колонку источника ----------
    frames = []
    for name, path in FILES.items():
        if not path.exists():
            print(f"!! ВНИМАНИЕ: файл не найден: {path}")
            continue
        df = pd.read_csv(path)
        df["_source"] = name
        print(
            f"[load] {name}: {len(df)} строк, "
            f"{df.groupby(['filters','blocks']).ngroups} уникальных (f,b)"
        )
        frames.append(df)

    all_df = pd.concat(frames, ignore_index=True, sort=False)
    print(f"\n[concat] всего строк: {len(all_df)}")

    # ---------- 2. Колонки и стратегии агрегации ----------
    acc_cols = ["fp32_acc_pct", "ptq_acc_pct", "qat_acc_pct", "int8_acc_pct"]
    size_cols = ["fp32_size_kb", "ptq_size_kb", "qat_size_kb", "int8_size_kb"]
    meta_cols = ["slug", "description", "simd_aligned", "params"]
    date_cols = ["train_date_utc", "updated_utc"]
    other_cols = ["quant"]  # старый формат, оставим если есть

    # Добавим недостающие колонки как NaN, чтобы не падать
    for col in acc_cols + size_cols + meta_cols + date_cols + other_cols:
        if col not in all_df.columns:
            all_df[col] = np.nan

    def first_non_null(s):
        s = s.dropna()
        return s.iloc[0] if len(s) > 0 else np.nan

    def max_or_nan(s):
        s = s.dropna()
        return s.max() if len(s) > 0 else np.nan

    def sources_list(s):
        return ",".join(sorted(set(s.dropna())))

    # ---------- 3. Группируем и агрегируем ----------
    agg_dict = {}
    for col in acc_cols:
        agg_dict[col] = max_or_nan
    for col in size_cols:
        agg_dict[col] = first_non_null
    for col in meta_cols:
        agg_dict[col] = first_non_null
    for col in date_cols:
        agg_dict[col] = max_or_nan
    for col in other_cols:
        agg_dict[col] = first_non_null
    agg_dict["_source"] = sources_list

    merged = all_df.groupby(["filters", "blocks"], as_index=False).agg(agg_dict)

    print(f"\n[merge] итог: {len(merged)} уникальных (filters, blocks)")

    # ---------- 4. Проверки и предупреждения ----------
    # Если в одной (f,b) совершенно разные точности (разница > 1.5%), warning
    print("\n[check] ищем подозрительные расхождения между источниками...")
    warnings = []
    for (f, b), grp in all_df.groupby(["filters", "blocks"]):
        for col in ["ptq_acc_pct", "qat_acc_pct"]:
            vals = grp[col].dropna().unique()
            if len(vals) >= 2:
                spread = vals.max() - vals.min()
                if spread > 1.5:
                    warnings.append(
                        {
                            "filters": f,
                            "blocks": b,
                            "metric": col,
                            "spread_pct": round(spread, 2),
                            "values": ", ".join(f"{v:.2f}" for v in vals),
                            "sources": ",".join(grp["_source"].unique()),
                        }
                    )
    if warnings:
        print(f"  Найдено {len(warnings)} расхождений > 1.5%:")
        for w in warnings[:10]:
            print(
                f"    f={w['filters']:3d} b={w['blocks']} {w['metric']}: "
                f"[{w['values']}] (Δ={w['spread_pct']}%, src={w['sources']})"
            )
        if len(warnings) > 10:
            print(f"    ... и ещё {len(warnings)-10}")
    else:
        print("  расхождений > 1.5% не найдено")

    # ---------- 5. Сводка по покрытию ----------
    def has_any_int8(row):
        return (
            pd.notna(row["ptq_acc_pct"])
            or pd.notna(row["qat_acc_pct"])
            or pd.notna(row["int8_acc_pct"])
        )

    merged["has_int8"] = merged.apply(has_any_int8, axis=1)
    n_int8 = merged["has_int8"].sum()
    n_fp32 = merged["fp32_acc_pct"].notna().sum()

    all_filters = sorted(merged["filters"].unique())
    all_blocks = sorted(merged["blocks"].unique())
    full_grid = len(all_filters) * len(all_blocks)

    print(f"\n[coverage] filters: {len(all_filters)}, blocks: {len(all_blocks)}")
    print(f"           полная сетка: {full_grid} ячеек")
    print(f"           заполнено INT8: {n_int8}")
    print(f"           заполнено FP32: {n_fp32}")

    merged = merged.drop(columns=["has_int8"])

    # ---------- 6. Сохранение ----------
    # Упорядочим колонки человекочитаемо
    col_order = (
        ["slug", "filters", "blocks", "description", "simd_aligned", "params"]
        + ["fp32_acc_pct", "fp32_size_kb"]
        + ["ptq_acc_pct", "ptq_size_kb"]
        + ["qat_acc_pct", "qat_size_kb"]
        + ["int8_acc_pct", "int8_size_kb", "quant"]
        + ["train_date_utc", "updated_utc"]
        + ["_source"]
    )
    col_order = [c for c in col_order if c in merged.columns]
    merged = merged[col_order]
    merged = merged.sort_values(["filters", "blocks"]).reset_index(drop=True)

    merged.to_csv(OUTPUT_CSV, index=False)
    print(f"\n[save] {OUTPUT_CSV} — {len(merged)} строк")

    # Лог расхождений
    if warnings:
        pd.DataFrame(warnings).to_csv(LOG_CSV, index=False)
        print(f"[save] {LOG_CSV} — {len(warnings)} расхождений (для ручного контроля)")


if __name__ == "__main__":
    main()
