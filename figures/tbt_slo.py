"""Paper figure: TBT under SLO-sized concurrency.

For each workload we pick the MAX concurrency such that the no-reclaim p99 TBT
meets a 200 ms SLO (binary search), then report p50 / p99(no RR) / p99(with RR)
at that concurrency. Message: sizing concurrency to a 200 ms SLO *without*
reclaim still leaves read-reclaim pushing p99 over the SLO.

High request rate (saturated), 112-layer NAND (ppb=448), weights on HBF,
LLaMA-3.1-405B.
"""
from __future__ import annotations

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from dataclasses import replace
from pathlib import Path

from defaults import DEFAULT_BASE, _auto_rates_for_workload
from hbf_rr.queueing import (simulate_request_stream, _simulate_one_fast,
                             _wpercentile, compute_capacity_max_concurrency)

# One colour per storage: HBM = grey (baseline, no read-reclaim), HBF = warm
# vermillion (the story). p99 vs max is read off the x-axis sub-labels.
C_HBM = "#c2c2c2"   # light grey (= KV write grey in write_lifetime)
C_HBF = "#D55E00"   # vermillion

# NAND layer count via HBF_LAYERS (default 112); PPB = layers x 4. Non-default
# layers get a folder suffix so results sit side-by-side.
LAYERS = int(os.environ.get("HBF_LAYERS", "112"))
PPB = LAYERS * 4
SUFFIX = "" if LAYERS == 112 else f"__{LAYERS}layer"
SLO_MS = 200.0
INPUTS = [4_096, 32_768, 65_536, 131_072, 262_144, 1_048_576]
OUTPUTS = [512, 2_048, 8_192, 32_768, 131_072, 524_288]
HARD_CAP = 1024


def _fmt_tok(n: int) -> str:
    if n >= 1_048_576 and n % 1_048_576 == 0:
        return f"{n // 1_048_576}M"
    if n >= 1024 and n % 1024 == 0:
        return f"{n // 1024}K"
    return str(n)


def _p99_no(cfg, I, O, B):
    """Weighted p99 TBT (ms), reclaim OFF, at concurrency B (saturated)."""
    rate = _auto_rates_for_workload(I, O, B, n=6)[-1]
    nreq = int(min(2500, max(80, 2 * B)))
    r = _simulate_one_fast(cfg, rate, nreq, B, 0, False, 30.0)
    return _wpercentile(r[0], r[10], 99) * 1e3


def _slo_concurrency(cfg, I, O, cap):
    """Largest B in [1, cap] with no-reclaim p99 <= SLO_MS (monotonic in B)."""
    if _p99_no(cfg, I, O, 1) > SLO_MS:
        return 1
    lo, hi = 1, cap
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if _p99_no(cfg, I, O, mid) <= SLO_MS:
            lo = mid
        else:
            hi = mid - 1
    return lo


def compute(base, I):
    p99_hbm, p99_hbf, max_hbm, max_hbf, Bs = [], [], [], [], []
    for O in OUTPUTS:
        cfg = replace(base, workload=replace(base.workload,
                      input_tokens=I, output_tokens=O, batch_size=1))
        cap = compute_capacity_max_concurrency(cfg)["max_concurrency_from_capacity"]
        cap = max(1, min(cap, HARD_CAP))
        B = _slo_concurrency(cfg, I, O, cap)
        rate = _auto_rates_for_workload(I, O, B, n=6)[-1]
        nreq = int(min(4000, max(200, 3 * B)))
        r = simulate_request_stream(cfg, rate, nreq, B, 0)
        p99_hbm.append(r.tbt_p99_noreclaim_ms); p99_hbf.append(r.tbt_p99_ms)
        max_hbm.append(r.tbt_max_noreclaim_ms); max_hbf.append(r.tbt_max_ms)
        Bs.append(B)
        print(f"  I={I:>8} O={O:>8}  SLO-conc={B:>5}  "
              f"p99[HBM]={r.tbt_p99_noreclaim_ms:6.1f}  p99[HBF]={r.tbt_p99_ms:6.1f}  "
              f"max[HBM]={r.tbt_max_noreclaim_ms:6.1f}  max[HBF]={r.tbt_max_ms:6.1f}")
    return (np.array(p99_hbm), np.array(p99_hbf),
            np.array(max_hbm), np.array(max_hbf), Bs)


def _fig(I, p99_hbm, p99_hbf, max_hbm, max_hbf, ymax, out_dir):
    from matplotlib.transforms import blended_transform_factory
    STEP = 1.3                              # group pitch (>1 spreads groups out)
    x = np.arange(len(OUTPUTS)) * STEP
    labels = [_fmt_tok(o) for o in OUTPUTS]
    fig, ax = plt.subplots(figsize=(9.8, 5.1))
    u, bw = 0.2, 0.19   # offset unit, bar width
    # left cluster = p99 (HBM, HBF); right cluster = max (HBM, HBF).
    # One colour per storage; only the p99-HBM / HBF bars carry a legend entry.
    b1 = ax.bar(x - 1.7 * u, p99_hbm, bw, color=C_HBM, label="HBM",
                edgecolor="white", linewidth=0.4, zorder=3)
    b2 = ax.bar(x - 0.7 * u, p99_hbf, bw, color=C_HBF, label="HBF",
                edgecolor="white", linewidth=0.4, zorder=3)
    b3 = ax.bar(x + 0.7 * u, max_hbm, bw, color=C_HBM,
                edgecolor="white", linewidth=0.4, zorder=3)
    b4 = ax.bar(x + 1.7 * u, max_hbf, bw, color=C_HBF,
                edgecolor="white", linewidth=0.4, zorder=3)
    for bars in (b1, b2, b3, b4):
        ax.bar_label(bars, fmt="%.0f", fontsize=6, padding=1)

    ax.set_ylabel("TBT  [ms]", fontsize=11)
    ax.set_ylim(0, ymax)
    ax.grid(axis="y", alpha=0.22, zorder=0)

    # ----- tabular x axis: a label table below the plot -----
    #   top row   : p99 / max   (metric, one cell per sub-cluster)
    #   bottom row : output length (group name, spans a group's two cells)
    # with grid lines boxing the table and left-hand row labels.
    trans = blended_transform_factory(ax.transData, ax.transAxes)
    ax.set_xticks([])
    ax.set_xlim(x[0] - STEP / 2, x[-1] + STEP / 2)
    Y_AX, Y_MID, Y_BOT = 0.0, -0.115, -0.235      # axes-fraction row boundaries
    edges = [x[0] - STEP / 2] + [xi + STEP / 2 for xi in x]   # group boundaries
    GRID = "#9a9a9a"
    for bx in edges:                          # vertical group separators only
        ax.plot([bx, bx], [Y_AX, Y_BOT], transform=trans, color=GRID,
                linewidth=1.0, clip_on=False, zorder=1)
    for xi in x:                               # p99 / max, top row
        ax.text(xi - 1.2 * u, (Y_AX + Y_MID) / 2, "p99", ha="center", va="center",
                transform=trans, fontsize=11)
        ax.text(xi + 1.2 * u, (Y_AX + Y_MID) / 2, "max", ha="center", va="center",
                transform=trans, fontsize=11)
    for xi, lab in zip(x, labels):             # output length, bottom row
        ax.text(xi, (Y_MID + Y_BOT) / 2, lab, ha="center", va="center",
                transform=trans, fontsize=10.5)
    # left-hand row labels, just left of the table
    lx = edges[0] - 0.06 * STEP
    ax.text(lx, (Y_AX + Y_MID) / 2, "metric", ha="right", va="center",
            transform=trans, fontsize=9.5, fontweight="bold")
    ax.text(lx, (Y_MID + Y_BOT) / 2, "output len", ha="right", va="center",
            transform=trans, fontsize=9.5, fontweight="bold")

    ax.legend(fontsize=9.5, loc="upper left", framealpha=0.95, ncol=2, title=None)
    fig.suptitle(f"TBT at SLO-sized concurrency (HBM p99 = {SLO_MS:.0f} ms)  —  "
                 f"HBF read-reclaim breaks the SLO  (input = {_fmt_tok(I)})",
                 fontsize=11.5, y=0.98)
    fig.subplots_adjust(left=0.13, right=0.98, top=0.9, bottom=0.2)
    out = out_dir / f"tbt_slo_sized__I={_fmt_tok(I)}.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def main():
    base = replace(DEFAULT_BASE, hbf=replace(DEFAULT_BASE.hbf, pages_per_block=PPB,
                   weights_storage="hbf"))
    out_dir = Path(__file__).resolve().parent.parent / "plots" / "paper" / f"tbt_slo{SUFFIX}"
    out_dir.mkdir(parents=True, exist_ok=True)
    data = {I: compute(base, I) for I in INPUTS}
    ymax = max(v[3].max() for v in data.values()) * 1.12   # by HBF max-TBT
    for I in INPUTS:
        p99_hbm, p99_hbf, max_hbm, max_hbf, Bs = data[I]
        p = _fig(I, p99_hbm, p99_hbf, max_hbm, max_hbf, ymax, out_dir)
        print(f"  wrote {p}")


if __name__ == "__main__":
    main()
