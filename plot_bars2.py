import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# Data
# ============================================================
methods = ["Original\nLAM", "Bridge-only\nFT", "DROID-only\nFT", "Cross-embod.\nFT (Ours)"]

# Frame-to-frame, skip=3, PSNR
f2f_bridge = [26.91, 31.12, 25.62, 29.30]
f2f_droid  = [23.03, 20.43, 27.03, 26.76]

# Rollout, PSNR
roll_bridge = [23.23, 26.22, 19.93, 24.04]
roll_droid  = [18.09, 15.19, 19.62, 19.82]

# ============================================================
# Colors
# ============================================================
bridge_color = "#5B9BD5"   # blue for Bridge
droid_color  = "#70AD47"   # green for DROID

# Ours gets a subtle glow / different treatment
bar_alpha = [0.55, 0.55, 0.55, 1.0]  # muted for others, full for ours
bar_edgewidth = [0.8, 0.8, 0.8, 2.0]

# ============================================================
# Plot: 2 panels, each with 4 method groups × 2 dataset bars
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

bar_width = 0.32
x = np.arange(len(methods))

for ax_idx, (ax, title, bridge_vals, droid_vals) in enumerate(zip(
    axes,
    ["Frame-to-Frame Reconstruction (skip=3)", "Open-Loop Rollout"],
    [f2f_bridge, roll_bridge],
    [f2f_droid, roll_droid],
)):
    for i in range(len(methods)):
        # Bridge bar
        ax.bar(
            x[i] - bar_width / 2, bridge_vals[i], bar_width,
            color=bridge_color, alpha=bar_alpha[i],
            edgecolor='#3A7BBF', linewidth=bar_edgewidth[i],
            label='Bridge V2' if i == 0 and ax_idx == 0 else None,
            zorder=3,
        )
        # DROID bar
        ax.bar(
            x[i] + bar_width / 2, droid_vals[i], bar_width,
            color=droid_color, alpha=bar_alpha[i],
            edgecolor='#4E8B31', linewidth=bar_edgewidth[i],
            label='DROID (stacked)' if i == 0 and ax_idx == 0 else None,
            zorder=3,
        )

        # Value labels
        ax.text(
            x[i] - bar_width / 2, bridge_vals[i] + 0.25,
            f"{bridge_vals[i]:.1f}",
            ha='center', va='bottom', fontsize=7.5,
            fontweight='bold' if i == 3 else 'normal',
            color='#2B5E8C',
        )
        ax.text(
            x[i] + bar_width / 2, droid_vals[i] + 0.25,
            f"{droid_vals[i]:.1f}",
            ha='center', va='bottom', fontsize=7.5,
            fontweight='bold' if i == 3 else 'normal',
            color='#3B6E28',
        )

    # Highlight "Ours" background
    ax.axvspan(x[3] - 0.45, x[3] + 0.45, color='#FFE0E0', alpha=0.4, zorder=0)

    # Styling
    ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=9.5)
    ax.set_ylabel("PSNR ↑ (dB)", fontsize=11)
    ax.grid(axis='y', alpha=0.3, zorder=0)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Y-axis range
    all_vals = bridge_vals + droid_vals
    ax.set_ylim(min(all_vals) - 3, max(all_vals) + 3)

    # Add "imbalanced" / "balanced" annotations
    for i in range(len(methods)):
        gap = abs(bridge_vals[i] - droid_vals[i])
        if i == 3:
            label = f"Δ={gap:.1f}"
            color = '#27AE60'
        else:
            label = f"Δ={gap:.1f}"
            color = '#E74C3C' if gap > 4 else '#F39C12'

        mid_y = min(bridge_vals[i], droid_vals[i]) - 1.2
        ax.text(
            x[i], mid_y, label,
            ha='center', va='top', fontsize=7,
            color=color, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor=color, alpha=0.7),
        )

# Legend
fig.legend(
    loc='upper center',
    bbox_to_anchor=(0.5, 1.02),
    ncol=2,
    fontsize=11,
    frameon=True,
    fancybox=True,
    edgecolor='#CCCCCC',
)

plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig("outputs/lam_eval_barplot2.pdf", bbox_inches='tight', dpi=300)
plt.savefig("outputs/lam_eval_barplot2.png", bbox_inches='tight', dpi=300)
print("Done!")