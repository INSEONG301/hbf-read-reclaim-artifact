"""Paper figure: per-HBF write traffic (stacked bars) vs HBF lifetime (lines)
across output length, one figure per input length.

Model / assumptions
-------------------
  * 112-layer NAND: pages_per_block = 112 x 4 = 448
  * Weights resident on HBF (weights_storage = "hbf")
  * Fixed batch / concurrency B = 16
  * LLaMA-3.1-405B, default HBF topology (8 HBF/GPU x 8 GPU)

Left axis (stacked bars, linear): per-HBF write bandwidth [MB/s]
  - "KV write (decode)"  = B x O x bytes_per_token / decode_time
                           (decode-phase KV writes only; prefill excluded)
  - "read-reclaim write" = KV read-reclaim + weight read-reclaim writes
Right axis (lines, log): HBF lifetime [years]
  - "no reclaim"   (dashed) = PE_budget x capacity / KV-write BW
  - "with reclaim" (solid)  = PE_budget x capacity / (KV-write + reclaim) BW
"""
from __future__ import annotations

import math
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from dataclasses import replace
from pathlib import Path

from defaults import DEFAULT_BASE
from hbf_rr.simulator import simulate
from hbf_rr.policies import compute_geometry

# ---- Okabe-Ito, CVD-safe (validated) ----
# The read-reclaim quantities are the story, so they stay vivid; the KV-write
# baseline and the no-reclaim lifetime are muted to grey so they recede.
C_KV      = "#c2c2c2"   # light grey — KV write (baseline, de-emphasised)
C_RECLAIM = "#E69F00"   # orange     — reclaim write
C_LIFE_NO = "#9a9a9a"   # grey       — lifetime, no reclaim (de-emphasised)
C_LIFE_YES= "#D55E00"   # vermil     — lifetime, with reclaim

SEC_PER_YEAR = 365.25 * 86400
B = 16
# NAND layer count selectable via HBF_LAYERS (default 112). PPB = layers x 4.
# 112 -> 448 (default), 162 -> 648, 218 -> 872. Non-default layers get a folder
# suffix so results sit side-by-side instead of overwriting.
LAYERS = int(os.environ.get("HBF_LAYERS", "112"))
PPB = LAYERS * 4
# Read-reclaim threshold (page reads per block) selectable via RR_THRESHOLD.
# Default 1e6 matches ReclaimConfig. Non-default values get a folder suffix so
# threshold variants sit side-by-side instead of overwriting, and the threshold
# is named in the figure title (the default set keeps its original title).
DEFAULT_THRESHOLD = 1_000_000
THRESHOLD = int(float(os.environ.get("RR_THRESHOLD", DEFAULT_THRESHOLD)))


def _fmt_th(t: int) -> str:
    """Compact mantissa-exponent label: 1e6 -> '1e6', 2000000 -> '2e6'.

    Strips trailing zeros into the exponent so threshold sweep values (which are
    multiples of 1e6, not powers of ten) stay readable in titles and paths.
    """
    v, e = int(t), 0
    while v >= 10 and v % 10 == 0:
        v //= 10
        e += 1
    return f"{v}e{e}" if e else str(v)


def _th_note() -> str:
    """Title fragment naming the threshold, empty for the default set."""
    return "" if THRESHOLD == DEFAULT_THRESHOLD else f", RR threshold = {_fmt_th(THRESHOLD)}"


SUFFIX = "" if LAYERS == 112 else f"__{LAYERS}layer"
if THRESHOLD != DEFAULT_THRESHOLD:
    SUFFIX += f"__th={_fmt_th(THRESHOLD)}"
INPUTS  = [4_096, 32_768, 65_536, 131_072, 262_144, 1_048_576]
OUTPUTS = [128, 512, 2_048, 8_192, 32_768, 131_072, 524_288]

# Every write_lifetime* figure set (this script's, the stacked/total variants,
# the v2..v5 layouts and their sweeps) is collected under one parent folder so
# plots/paper/ is not flooded with a dozen sibling directories. Scripts import
# this rather than rebuilding the path, so the grouping moves in one place.
# `parent.parent` because these scripts live in <repo>/write_lifetime/.
REPO_ROOT = Path(__file__).resolve().parent.parent
PAPER_DIR = REPO_ROOT / "plots" / "paper"
WRITE_LIFETIME_DIR = PAPER_DIR / "write_lifetime"
# Prefix for the "how to regenerate" lines the scripts write into each output
# folder's README, so the command works when run from the repository root.
SCRIPT_DIR_NAME = Path(__file__).resolve().parent.name


def _fmt_tok(n: int) -> str:
    if n >= 1_048_576 and n % 1_048_576 == 0:
        return f"{n // 1_048_576}M"
    if n >= 1024 and n % 1024 == 0:
        return f"{n // 1024}K"
    return str(n)


def _compute(base, I: int):
    """Return per-O arrays: kv_bytes, reclaim_bytes (per HBF), life_no, life_yes
    (years). KV write includes prefill + decode (I+O tokens). Lifetime is
    derived from write *bandwidth*; bars display total write bytes over the run.
    Uses simulator.simulate() directly (fixed batch = B = concurrency)."""
    kv, rec, ln, ly = [], [], [], []
    for O in OUTPUTS:
        cfg = replace(base, workload=replace(base.workload,
                      input_tokens=I, output_tokens=O, batch_size=B))
        r = simulate(cfg)
        dt = r.total_decode_time_s
        kv_bytes = r.kv_write_bytes_per_hbf                  # B*(I+O)*bpt, prefill+decode
        rec_bw = (r.reclaim_write_bw_per_hbf_bytes_per_sec
                  + r.weight_reclaim_write_bw_per_hbf_bytes_per_sec)
        rec_bytes = rec_bw * dt                              # reclaim writes over the run
        kv.append(kv_bytes)
        rec.append(rec_bytes)
        ln.append(r.hbf_lifetime_years_kv_only)              # no reclaim (KV writes only)
        ly.append(r.hbf_lifetime_years)                      # with reclaim (KV + all reclaim)
    return (np.array(kv), np.array(rec), np.array(ln), np.array(ly))


def _fig_for_input(base, I: int, out_dir: Path):
    kv, rec, life_no, life_yes = _compute(base, I)
    kv_gb, rec_gb = kv / 1e9, rec / 1e9          # bytes -> GB
    x = np.arange(len(OUTPUTS))
    labels = [_fmt_tok(o) for o in OUTPUTS]

    fig, axL = plt.subplots(figsize=(10.0, 5.0))
    axR = axL.twinx()

    # ----- left: grouped write-byte bars (log; stacking is misleading on log) -----
    w = 0.40
    axL.bar(x - w / 2, kv_gb, w, color=C_KV, label="KV write",
            edgecolor="white", linewidth=0.5, zorder=3)
    axL.bar(x + w / 2, rec_gb, w, color=C_RECLAIM, label="read-reclaim write",
            edgecolor="white", linewidth=0.5, zorder=3)

    axL.set_yscale("log")
    axL.set_ylabel("per-HBF write volume  [GB]", fontsize=11)
    axL.set_xlabel("output length  [tokens]", fontsize=11)
    axL.set_xticks(x)
    axL.set_xticklabels(labels, fontsize=9.5)
    axL.set_ylim(1e-1, 2e6)                    # fixed across the 4 figures
    axL.grid(axis="y", alpha=0.22, which="major", zorder=0)

    # ----- right: lifetime lines (log) -----
    axR.plot(x, life_no, color=C_LIFE_NO, linestyle="--", linewidth=2.2,
             marker="o", markersize=7.5, label="lifetime — no RR", zorder=6)
    axR.plot(x, life_yes, color=C_LIFE_YES, linestyle="-", linewidth=2.2,
             marker="s", markersize=7.5, label="lifetime — with RR", zorder=6)
    axR.set_yscale("log")
    axR.set_ylim(0.1, 1e4)                     # fixed across the 4 figures
    axR.set_ylabel("HBF lifetime  [years]", fontsize=11)
    # 5-year lifetime target
    axR.axhline(5.0, color="#c0392b", linestyle="--", linewidth=1.6, zorder=4)
    axR.text(0.98, 5.0, "5-year limit", va="bottom", ha="right", fontsize=10,
             fontweight="bold", color="#c0392b", transform=axR.get_yaxis_transform(),
             bbox=dict(facecolor="white", alpha=0.85, edgecolor="none", pad=1.5),
             zorder=7)

    # ----- combined legend -----
    hL, lL = axL.get_legend_handles_labels()
    hR, lR = axR.get_legend_handles_labels()
    axL.legend(hL + hR, lL + lR, loc="upper center", fontsize=9,
               framealpha=0.95, ncol=4, columnspacing=1.4,
               bbox_to_anchor=(0.5, 1.0))

    fig.suptitle(f"Write traffic & HBF lifetime vs output length "
                 f"(input = {_fmt_tok(I)}, batch = {B}{_th_note()})",
                 fontsize=12.5, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = out_dir / f"write_lifetime__I={_fmt_tok(I)}.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return (out, kv, rec, life_no, life_yes)


def main():
    import csv
    base = replace(DEFAULT_BASE,
                   hbf=replace(DEFAULT_BASE.hbf, pages_per_block=PPB,
                               weights_storage="hbf"),
                   reclaim=replace(DEFAULT_BASE.reclaim,
                                   threshold_page_reads=THRESHOLD))
    out_dir = WRITE_LIFETIME_DIR / f"write_lifetime{SUFFIX}"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "write_lifetime_data.csv"
    with open(csv_path, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["input_tokens", "output_tokens", "batch",
                     "kv_write_GB_per_hbf", "reclaim_write_GB_per_hbf",
                     "lifetime_years_no_reclaim", "lifetime_years_with_reclaim"])
        for I in INPUTS:
            p, kv, rec, ln, ly = _fig_for_input(base, I, out_dir)
            print(f"  wrote {p}")
            for O, k, rc, a, b in zip(OUTPUTS, kv, rec, ln, ly):
                wr.writerow([I, O, B, f"{k/1e9:.4g}", f"{rc/1e9:.4g}",
                             f"{a:.4g}", f"{b:.4g}"])
    print(f"  wrote {csv_path}")


if __name__ == "__main__":
    main()
