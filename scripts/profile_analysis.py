"""
profile_analysis.py

Parses kws profile CSV exported from ESP32-S3 and produces the main figure
for the thesis: per-layer execution time with error bars, coloured by op type.

Run:
    python profile_analysis.py profile.csv

Output:
    figure_layer_time.pdf  — main figure
    figure_layer_time.png  — same, raster
    layer_stats.csv        — summary table (mean, std, min, max per layer)
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ---------- 1. Load ----------
csv_path = sys.argv[1] if len(sys.argv) > 1 else "scripts/stats.csv"
df = pd.read_csv(csv_path, comment="#")

# Drop warmup runs (run_id == -1)
df = df[df.run_id >= 0].copy()
print(f"loaded {len(df)} events across {df.run_id.nunique()} runs")
print(f"ops per run: {df.groupby('run_id').size().unique()}")

# Convert µs → ms for readability
df["ticks_ms"] = df.ticks_us / 1000.0

# ---------- 2. Aggregate per layer ----------
agg = (
    df.groupby(["op_index", "op_tag"])
    .agg(
        mean_ms=("ticks_ms", "mean"),
        std_ms=("ticks_ms", "std"),
        min_ms=("ticks_ms", "min"),
        max_ms=("ticks_ms", "max"),
        median_ms=("ticks_ms", "median"),
    )
    .reset_index()
    .sort_values("op_index")
)

total_mean = agg.mean_ms.sum()
agg["pct_time"] = 100 * agg.mean_ms / total_mean
print(agg.to_string(index=False))
print(f"\ntotal mean inference: {total_mean:.1f} ms")

agg.to_csv("layer_stats.csv", index=False, float_format="%.3f")

# # ---------- 3. Plot ----------
# # Op-type colours.
# colors = {
#     "CONV_2D": "#d94545",  # red — the bottleneck
#     "DEPTHWISE_CONV_2D": "#4a90d9",  # blue
#     "MEAN": "#9b59b6",  # purple
#     "FULLY_CONNECTED": "#777777",  # grey — basically zero
# }
#
#
# # Friendly labels for X-axis: distinguish the standard first Conv from PW convs.
# def short_label(row):
#     tag = row.op_tag
#     if tag == "CONV_2D":
#         # The 0th CONV_2D is the standard 10x4 Conv1; remaining CONV_2Ds at
#         # even op_indices are 1x1 pointwise convs paired with DW-convs.
#         return "Conv1" if row.op_index == 0 else f"PW{row.op_index // 2}"
#     if tag == "DEPTHWISE_CONV_2D":
#         return f"DW{(row.op_index + 1) // 2}"
#     if tag == "MEAN":
#         return "GAP"
#     if tag == "FULLY_CONNECTED":
#         return "FC"
#     return tag
#
#
# agg["label"] = agg.apply(short_label, axis=1)
# agg["color"] = agg.op_tag.map(colors)
#
# # Figure
# fig, ax = plt.subplots(figsize=(10, 4.5))
# xs = np.arange(len(agg))
# bars = ax.bar(
#     xs,
#     agg.mean_ms,
#     yerr=agg.std_ms,
#     color=agg.color,
#     edgecolor="black",
#     linewidth=0.5,
#     error_kw=dict(ecolor="black", elinewidth=0.8, capsize=2),
# )
#
# ax.set_xticks(xs)
# ax.set_xticklabels(agg.label, rotation=0, fontsize=9)
# ax.set_ylabel("Время выполнения слоя, мс", fontsize=11)
# ax.set_xlabel("Слой графа вычислений", fontsize=11)
# ax.set_title(
#     f"Распределение времени инференса DS-CNN по слоям на ESP32-S3\n"
#     f"(n={df.run_id.nunique()} прогонов, ср. инференс = {total_mean:.0f} мс)",
#     fontsize=11,
# )
#
# # Percentages above each bar
# for x, m, p in zip(xs, agg.mean_ms, agg.pct_time):
#     ax.text(
#         x, m + agg.std_ms.max() * 1.5, f"{p:.1f}%", ha="center", va="bottom", fontsize=8
#     )
#
# # Legend with op-type colours
# legend_handles = [
#     mpatches.Patch(
#         color=colors["CONV_2D"], label="CONV_2D (стандартная + точечные 1×1)"
#     ),
#     mpatches.Patch(color=colors["DEPTHWISE_CONV_2D"], label="DEPTHWISE_CONV_2D"),
#     mpatches.Patch(color=colors["MEAN"], label="MEAN (Global Avg Pooling)"),
#     mpatches.Patch(color=colors["FULLY_CONNECTED"], label="FULLY_CONNECTED"),
# ]
# ax.legend(handles=legend_handles, loc="upper right", fontsize=9, framealpha=0.95)
#
# ax.grid(axis="y", linestyle=":", alpha=0.5)
# ax.set_axisbelow(True)
# ax.set_ylim(0, agg.mean_ms.max() * 1.18)
#
# plt.tight_layout()
# plt.savefig("figure_layer_time.pdf", bbox_inches="tight")
# plt.savefig("figure_layer_time.png", bbox_inches="tight", dpi=150)
# print("\nsaved: figure_layer_time.pdf, figure_layer_time.png, layer_stats.csv")
