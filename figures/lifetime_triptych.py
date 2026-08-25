"""Three-panel HBF-lifetime figure (a)(b)(c), unified in the reclaim_decomp
style (output-length x axis, fixed input = 64K & batch = 16, linear years axis,
red 5-year limit, compact fonts) for a 2-column page width.

  (a) NAND layer count — with-RR lifetime vs output length, one line per layer
  (b) HBF bandwidth    — with-RR lifetime vs output length, one line per bandwidth
  (c) KV vs weight RR  — only-KV / only-weight / both, vs output length

All three share one linear y axis and the same x axis, so the lifetimes sit on
a common scale. LLaMA-3.1-405B, input = 64K, batch = 16, weights on HBF.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from dataclasses import replace
from pathlib import Path

from defaults import DEFAULT_BASE
from hbf_rr.simulator import simulate
from hbf_rr.models import LLAMA_3_1_405B

SPY = 365.25 * 86400
B = 16
INPUT = 65_536
OUTPUTS = [2_048, 32_768, 131_072, 524_288]          # 4 x-axis points
TARGET_YEARS = 5.0
YMAX = 8.0

LAYERS = [112, 164, 218, 332]                       # (a); bw held at 1.6 TB/s
BANDWIDTHS = [0.8e12, 1.6e12, 2.4e12, 3.2e12]        # (b); layers held at 162
BW_FIXED = 1.6e12
LAYERS_FIXED = 162

# Okabe-Ito for the swept lines in (a)/(b); reclaim_decomp colours in (c).
COLORS = ["#0072B2", "#009E73", "#E69F00", "#D55E00"]
MARKERS = ["o", "s", "^", "D"]
C_KV, C_WT, C_BOTH = "#D55E00", "#5A9E30", "#d81e05"
LIMIT_C = "#c0392b"

FS_LABEL, FS_TICK, FS_LEG, FS_LIMIT, FS_CAP = 11.5, 9.5, 9.0, 10.0, 11.5


def _fmt(n: int) -> str:
    if n >= 1_048_576 and n % 1_048_576 == 0:
        return f"{n // 1_048_576}M"
    if n >= 1024 and n % 1024 == 0:
        return f"{n // 1024}K"
    return str(n)


def _base(layers, bw):
    return replace(DEFAULT_BASE, model=LLAMA_3_1_405B,
                   hbf=replace(DEFAULT_BASE.hbf, pages_per_block=layers * 4,
                               bandwidth_bytes_per_sec=bw,
                               hbm_bandwidth_bytes_per_sec=bw,
                               weights_storage="hbf"))


def _life_curve(layers, bw):
    base = _base(layers, bw)
    out = []
    for O in OUTPUTS:
        r = simulate(replace(base, workload=replace(base.workload,
                     input_tokens=INPUT, output_tokens=O, batch_size=B)))
        out.append(r.hbf_lifetime_years)                 # with read-reclaim
    return np.array(out)


def _decomp():
    base = _base(LAYERS_FIXED, BW_FIXED)
    budget = base.reclaim.pe_cycle_limit * base.hbf.capacity_bytes
    kv, wt, both = [], [], []
    for O in OUTPUTS:
        r = simulate(replace(base, workload=replace(base.workload,
                     input_tokens=INPUT, output_tokens=O, batch_size=B)))
        kvw = r.kv_write_bw_per_hbf_bytes_per_sec
        kvr = r.reclaim_write_bw_per_hbf_bytes_per_sec
        wtr = r.weight_reclaim_write_bw_per_hbf_bytes_per_sec
        f = lambda bw: budget / bw / SPY if bw > 0 else np.nan
        kv.append(f(kvw + kvr)); wt.append(f(kvw + wtr)); both.append(r.hbf_lifetime_years)
    return np.array(kv), np.array(wt), np.array(both)


def _common(ax, show_y):
    x = np.arange(len(OUTPUTS))
    ax.set_xticks(x)
    ax.set_xticklabels([_fmt(o) for o in OUTPUTS], fontsize=FS_TICK)
    ax.set_xlim(-0.5, len(x) - 0.5)
    ax.set_ylim(0, YMAX)
    ax.set_yticks([0, 2, 4, 6, 8])
    # labelleft per-axis (not set_yticklabels, which would clear the shared axis)
    ax.tick_params(labelsize=FS_TICK, labelleft=show_y)
    ax.set_xlabel("output length  [tokens]", fontsize=FS_LABEL)
    ax.grid(axis="y", alpha=0.22, which="major", zorder=0)
    if show_y:
        ax.set_ylabel("HBF lifetime  [years]", fontsize=FS_LABEL)
    ax.axhline(TARGET_YEARS, color=LIMIT_C, linestyle="--", linewidth=1.4, zorder=4)
    ax.text(0.97, TARGET_YEARS + 0.22, "5-year limit", ha="right", va="bottom",
            fontsize=FS_LIMIT, fontweight="bold", color=LIMIT_C,
            transform=ax.get_yaxis_transform(), zorder=8)


def _sweep_panel(ax, curves, labels, show_y):
    x = np.arange(len(OUTPUTS))
    for y, lab, c, m in zip(curves, labels, COLORS, MARKERS):
        ax.plot(x, y, color=c, marker=m, markersize=4.0, linewidth=1.5,
                markeredgecolor="white", markeredgewidth=0.5, label=lab, zorder=6)
    _common(ax, show_y)
    ax.legend(loc="upper left", fontsize=FS_LEG, framealpha=0.9, handlelength=1.4,
              labelspacing=0.25, borderpad=0.3)


def _decomp_panel(ax, kv, wt, both, show_y):
    x = np.arange(len(OUTPUTS))
    l_kv, = ax.plot(x, kv, color=C_KV, marker="s", markersize=4.0, linewidth=1.5,
                    markeredgecolor="white", markeredgewidth=0.5, label="only KV RR", zorder=6)
    l_wt, = ax.plot(x, wt, color=C_WT, marker="^", markersize=4.5, linewidth=1.5,
                    markeredgecolor="white", markeredgewidth=0.5, label="only weight RR", zorder=6)
    l_both, = ax.plot(x, both, color=C_BOTH, marker="D", markersize=4.0, linewidth=1.9,
                      markeredgecolor="white", markeredgewidth=0.5, label="both RR", zorder=7)
    _common(ax, show_y)
    ax.legend(handles=[l_both, l_kv, l_wt], loc="upper left", fontsize=FS_LEG,
              framealpha=0.9, handlelength=1.4, labelspacing=0.25, borderpad=0.3)


def main():
    a_curves = [_life_curve(L, BW_FIXED) for L in LAYERS]
    b_curves = [_life_curve(LAYERS_FIXED, bw) for bw in BANDWIDTHS]
    kv, wt, both = _decomp()

    fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(7.16, 3.05), sharey=True,
                                        layout="constrained")
    fig.get_layout_engine().set(w_pad=0.02, wspace=0.03)
    _sweep_panel(axA, a_curves, [f"{L} layers" for L in LAYERS], show_y=True)
    _sweep_panel(axB, b_curves, [f"{bw / 1e12:g} TB/s" for bw in BANDWIDTHS],
                 show_y=False)
    _decomp_panel(axC, kv, wt, both, show_y=False)

    # bold, enlarged (a)/(b)/(c) captions on a shared baseline below the panels
    fig.canvas.draw()
    inv = fig.transFigure.inverted()
    rend = fig.canvas.get_renderer()
    caps = ["(a)  NAND layer count", "(b)  HBF bandwidth", "(c)  KV vs weight reclaim"]
    y_cap = min(ax.get_tightbbox(rend).transformed(inv).y0
                for ax in (axA, axB, axC)) - 0.015
    cap_dx = [-0.008, -0.02, -0.008]            # (a),(c) a touch right of (b)
    for ax, cap, dx in zip((axA, axB, axC), caps, cap_dx):
        p = ax.get_position()
        fig.text((p.x0 + p.x1) / 2 + dx, y_cap, cap, ha="center", va="top",
                 fontsize=FS_CAP, fontweight="bold")

    out_dir = Path(__file__).resolve().parent.parent / "plots" / "paper" / "lifetime_triptych"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "lifetime_triptych.png"
    fig.savefig(out, dpi=400, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
