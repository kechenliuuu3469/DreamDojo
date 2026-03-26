import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# Data from CSV results
# ============================================================

methods = ["Original LAM", "Bridge-only FT", "DROID-only FT", "Cross-embod. FT\n(Ours)"]

# Frame-to-frame, skip=3, PSNR
f2f_bridge_psnr = [26.91, 31.12, 25.62, 29.30]
f2f_droid_psnr  = [23.03, 20.43, 27.03, 26.76]

# Rollout, PSNR
roll_bridge_psnr = [23.23, 26.22, 19.93, 24.04]
roll_droid_psnr  = [18.09, 15.19, 19.62, 19.82]

# ============================================================
# Colors
# ============================================================
colors = [
    "#9E9E9E",   # Original LAM — gray
    "#5B9BD5",   # Bridge-only FT — blue
    "#70AD47",   # DROID-only FT — green
    "#E74C3C",   # Cross-embod. FT (Ours) — red/highlight
]

edge_colors = [
    "#757575",
    "#3A7BBF",
    "#4E8B31",
    "#C0392B",
]

# ============================================================
# Plot
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=False)

bar_width = 0.16
datasets = ["Bridge V2", "DROID\n(stacked)"]

for ax_idx, (ax, title, bridge_vals, droid_vals) in enumerate(zip(
    axes,
    ["Frame-to-Frame Reconstruction (skip=3)", "Open-Loop Rollout"],
    [f2f_bridge_psnr, roll_bridge_psnr],
    [f2f_droid_psnr, roll_droid_psnr],
)):
    x = np.arange(len(datasets))

    for i, (method, color, ec) in enumerate(zip(methods, colors, edge_colors)):
        vals = [bridge_vals[i], droid_vals[i]]
        offset = (i - 1.5) * bar_width
        bars = ax.bar(
            x + offset, vals, bar_width,
            label=method if ax_idx == 0 else None,
            color=color,
            edgecolor=ec,
            linewidth=1.2,
            zorder=3,
        )

        # Add value labels on top of bars
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.3,
                f"{val:.1f}",
                ha='center', va='bottom',
                fontsize=7.5, fontweight='bold' if "Ours" in method else 'normal',
                color=ec,
            )

    # Add a subtle highlight box around cross-embodiment bars
    # for j, dataset_idx in enumerate(x):
    #     ours_x = dataset_idx + (3 - 1.5) * bar_width
    #     ax.add_patch(plt.Rectangle(
    #         (ours_x - bar_width / 2 , 0),
    #         bar_width ,
    #         max(bridge_vals[3], droid_vals[3]) + 1.5,
    #         fill=False, edgecolor='#E74C3C', linewidth=1.5,
    #         linestyle='--', alpha=0.6, zorder=2,
    #     ))

    ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontsize=11)
    ax.set_ylabel("PSNR ↑ (dB)", fontsize=11)
    ax.grid(axis='y', alpha=0.3, zorder=0)
    ax.set_axisbelow(True)

    # Set y-axis range
    all_vals = bridge_vals + droid_vals
    y_min = min(all_vals) - 3
    y_max = max(all_vals) + 3
    ax.set_ylim(y_min, y_max)

    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# Legend
fig.legend(
    loc='upper center',
    bbox_to_anchor=(0.5, 1.02),
    ncol=4,
    fontsize=10,
    frameon=True,
    fancybox=True,
    shadow=False,
    edgecolor='#CCCCCC',
)

plt.tight_layout(rect=[0, 0, 1, 0.92])
plt.savefig("outputs/lam_eval_barplot.pdf", bbox_inches='tight', dpi=300)
plt.savefig("outputs/lam_eval_barplot.png", bbox_inches='tight', dpi=300)
print("Saved to outputs/lam_eval_barplot.pdf and .png")