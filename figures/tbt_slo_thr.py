"""tbt_slo variant with RR-threshold legend wording, matching the (b)-panel
style of write_lifetime_total_v5: grey = "RR threshold = inf (ideal)"
(no read-reclaim), red = "RR threshold = 1e6 (typical SLC)". Output length
starts at 2K. Reuses the SLO-sized-concurrency compute from paper_tbt_slo.

Defaults: 162-layer NAND, input = 1M, weights on HBF, LLaMA-3.1-405B.
"""
from __future__ import annotations

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from dataclasses import replace
from pathlib import Path
from matplotlib.transforms import blended_transform_factory

from defaults import DEFAULT_BASE
import tbt_slo as m

LAYERS = int(os.environ.get("HBF_LAYERS", "162"))
PPB = LAYERS * 4
I = int(os.environ.get("TBT_INPUT", 1_048_576))
OUTPUTS = [2_048, 8_192, 32_768, 131_072, 524_288]     # from 2K
m.OUTPUTS = OUTPUTS                                     # compute() iterates this

C_LO = m.C_HBM              # grey — ideal (no read-reclaim)
# typical-SLC bar colour; default vivid red (matches write_lifetime_total_v5 b).
# Override via TBT_HI_COLOR (+ TBT_TAG to keep the variant as a separate file).
C_HI = os.environ.get("TBT_HI_COLOR", "#d81e05")
TAG = os.environ.get("TBT_TAG", "")
LAB_LO = r"RR threshold = $\infty$ (ideal)"
LAB_HI = r"RR threshold = $10^6$ (typical SLC)"
SLO_MS = m.SLO_MS
_fmt_tok = m._fmt_tok


def _fig(I, p99_lo, p99_hi, max_lo, max_hi, ymax, out_dir):
    STEP = 1.3
    x = np.arange(len(OUTPUTS)) * STEP
    labels = [_fmt_tok(o) for o in OUTPUTS]
    fig, ax = plt.subplots(figsize=(9.8, 3.3))
    u, bw = 0.2, 0.19
    b1 = ax.bar(x - 1.7 * u, p99_lo, bw, color=C_LO, label=LAB_LO,
                edgecolor="white", linewidth=0.4, zorder=3)
    b2 = ax.bar(x - 0.7 * u, p99_hi, bw, color=C_HI, label=LAB_HI,
                edgecolor="white", linewidth=0.4, zorder=3)
    b3 = ax.bar(x + 0.7 * u, max_lo, bw, color=C_LO,
                edgecolor="white", linewidth=0.4, zorder=3)
    b4 = ax.bar(x + 1.7 * u, max_hi, bw, color=C_HI,
                edgecolor="white", linewidth=0.4, zorder=3)
    for bars in (b1, b2, b3, b4):
        ax.bar_label(bars, fmt="%.0f", fontsize=8.5, padding=1)

    ax.set_ylabel("TBT  [ms]", fontsize=13)
    ax.tick_params(axis="y", labelsize=11.5)
    ax.set_ylim(0, ymax)
    ax.grid(axis="y", alpha=0.22, zorder=0)

    # tabular x axis: p99/max over output length
    trans = blended_transform_factory(ax.transData, ax.transAxes)
    ax.set_xticks([])
    ax.set_xlim(x[0] - STEP / 2, x[-1] + STEP / 2)
    Y_AX, Y_MID, Y_BOT = 0.0, -0.125, -0.255
    edges = [x[0] - STEP / 2] + [xi + STEP / 2 for xi in x]
    GRID = "#9a9a9a"
    for bx in edges:
        ax.plot([bx, bx], [Y_AX, Y_BOT], transform=trans, color=GRID,
                linewidth=1.0, clip_on=False, zorder=1)
    for xi in x:
        ax.text(xi - 1.2 * u, (Y_AX + Y_MID) / 2, "p99", ha="center", va="center",
                transform=trans, fontsize=13)
        ax.text(xi + 1.2 * u, (Y_AX + Y_MID) / 2, "max", ha="center", va="center",
                transform=trans, fontsize=13)
    for xi, lab in zip(x, labels):
        ax.text(xi, (Y_MID + Y_BOT) / 2, lab, ha="center", va="center",
                transform=trans, fontsize=12.5)
    lx = edges[0] - 0.05 * STEP
    ax.text(lx, (Y_AX + Y_MID) / 2, "metric", ha="right", va="center",
            transform=trans, fontsize=13)
    ax.text(lx, (Y_MID + Y_BOT) / 2, "output length\n[tokens]", ha="right",
            va="center", transform=trans, fontsize=13)

    # legend inside, upper-left; extra y headroom keeps it off the bars
    ax.legend(fontsize=12, loc="upper left", framealpha=0.95, ncol=1)
    fig.subplots_adjust(left=0.14, right=0.985, top=0.955, bottom=0.26)
    out = out_dir / f"tbt_slo_thr__I={_fmt_tok(I)}{TAG}.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return out


def main():
    base = replace(DEFAULT_BASE, hbf=replace(DEFAULT_BASE.hbf, pages_per_block=PPB,
                   weights_storage="hbf"))
    p99_lo, p99_hi, max_lo, max_hi, Bs = m.compute(base, I)
    ymax = float(np.array(max_hi).max()) * 1.45   # headroom so the legend clears the bar labels
    out_dir = Path(__file__).resolve().parent.parent / "plots" / "paper" / f"tbt_slo_thr__{LAYERS}layer"
    out_dir.mkdir(parents=True, exist_ok=True)
    p = _fig(I, p99_lo, p99_hi, max_lo, max_hi, ymax, out_dir)
    print(f"  wrote {p}")


if __name__ == "__main__":
    main()
