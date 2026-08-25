"""v4 with the read-reclaim bar split into its KV and Weight components.

Companion to paper_figs_total_v4.py, which is left untouched. Identical in
every other respect — same (a)|(b) side-by-side layout, same output range
(2K..512K), same lifetime panel, same sizing knobs. The only change is in
panel (a):

    v4:  KV store | Read reclaim (RR)          [RR = KV reclaim + weight reclaim]
    v5:  KV store | RR (Weight) | RR (KV)      + a Total marker

Grouped, not stacked. The y axis is logarithmic, and stacking segments on a log
axis misrepresents the upper segment's height (the same reason paper_figs.py
plots grouped bars and paper_figs_stacked.py has to draw its stack total-first).

Two things the split makes visible:

  * Weight reclaim depends only on output length, not on input length — the
    weight shard is re-read once per forward pass regardless of context, so its
    bar is the same 16.8 / 67 / 269 / 1075 / 4300 GB column at every input.
  * Which component dominates flips with the workload. At short context and
    short output the weight shard is the entire reclaim load (KV reclaim is
    exactly zero there); by I=1M, O=512K KV reclaim is 14x the weight term.

Where KV reclaim is exactly 0 no bar can be drawn on a log axis, so those slots
are annotated "0" — the integer-event model (floor(reads/threshold)) means no
KV block crossed the threshold within the decode window, not missing data.

    HBF_LAYERS=162 python3.11 write_lifetime/paper_figs_total_v5.py
    HBF_LAYERS=162 FIG_W=3.45 FIG_H=2.0 python3.11 write_lifetime/paper_figs_total_v5.py
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
import _common as pf

B, PPB, SUFFIX = pf.B, pf.PPB, pf.SUFFIX
THRESHOLD = pf.THRESHOLD
C_KV = pf.C_KV           # light grey  — KV store
C_RR_W = "#E69F00"       # orange      — read reclaim, weight blocks
C_RR_KV = "#D55E00"      # vermillion  — read reclaim, KV cache blocks
C_LIFE_NO = "#9a9a9a"    # light grey  — ideal (no reclaim)
C_LIFE_YES = "#d81e05"   # vivid red   — with reclaim
C_LIMIT = "#c0392b"      # red         — 5-year limit line
INPUTS = pf.INPUTS
OUTPUTS = [o for o in pf.OUTPUTS if 2_048 <= o <= 524_288]
_fmt_tok = pf._fmt_tok

# ---- sizing: default to full text width, as in v4 ----
FIG_W = float(os.environ.get("FIG_W", "7.16"))
# Height is set by the legend: the tallest Total diamond (I=1M, O=512K) sits at
# ~83% of panel (a)'s y range, and the one-row legend above it needs the rest.
# Below ~2.5 in the two collide; 2.65 in leaves a ~3.4 pt gap.
FIG_H = float(os.environ.get("FIG_H", "2.65"))
_S = min(1.0, (FIG_W / 2.0) / 3.45)
FS_LABEL = 10.5 * _S
FS_TICK = 9.5 * _S
FS_LEG = 9.0 * _S
FS_PANEL = 10.5 * _S
FS_LIMIT = 10.5 * _S
FS_ZERO = 6.5 * _S
WSUFFIX = "" if abs(FIG_W - 7.16) < 1e-9 else f"__w={FIG_W:g}in"

# Panel captions, printed BELOW each panel as a second x-label line. Kept in the
# x-label (rather than a free-floating text) so constrained_layout reserves room
# for them and nothing is clipped at these widths. Mathtext gives the "(a)" its
# bold weight, since a single label string cannot mix font weights.
PANEL_A = r"$\mathbf{(a)\ Write\ overhead}$"
PANEL_B = r"$\mathbf{(b)\ HBF\ lifetime}$"
# Gap between the axis label and the caption beneath it. This is baseline-to-
# baseline distance in font heights, not the gap itself: the visible whitespace
# is (linespacing - 1) x fontsize, so 2.02 leaves 9.2 pt where 2.6 left 13.8 pt.
CAPTION_LINESPACING = 2.02


def _th_math(t: int) -> str:
    e = int(round(math.log10(t))) if t > 0 else 0
    if 10 ** e == t:
        return rf"$10^{{{e}}}$"
    return rf"${t / 10 ** e:g}\times10^{{{e}}}$"


def _th_legend_label() -> str:
    note = " (typical SLC)" if THRESHOLD == 1_000_000 else ""
    return f"RR threshold = {_th_math(THRESHOLD)}{note}"


def _compute_split(base, I: int):
    """Like pf._compute, but returns the KV and weight reclaim writes
    separately instead of their sum. Bytes are BW x decode time, matching how
    pf._compute turns the simulator's bandwidths into per-run volumes."""
    kv, rr_kv, rr_w, ln, ly = [], [], [], [], []
    for O in OUTPUTS:
        cfg = replace(base, workload=replace(base.workload,
                      input_tokens=I, output_tokens=O, batch_size=B))
        r = simulate(cfg)
        dt = r.total_decode_time_s
        kv.append(r.kv_write_bytes_per_hbf)
        rr_kv.append(r.reclaim_write_bw_per_hbf_bytes_per_sec * dt)
        rr_w.append(r.weight_reclaim_write_bw_per_hbf_bytes_per_sec * dt)
        ln.append(r.hbf_lifetime_years_kv_only)      # no reclaim
        ly.append(r.hbf_lifetime_years)              # KV write + all reclaim
    return tuple(np.array(a) for a in (kv, rr_kv, rr_w, ln, ly))


def _fig_for_input(base, I: int, out_dir: Path,
                   fname_tag: str | None = None,
                   ylim_a: tuple | None = None, ylim_b: tuple | None = None):
    """Draw one figure for input length `I` from `base`.

    The optional arguments exist for sweep drivers that collect many variants in
    a single folder: `fname_tag` goes into the filename, and `ylim_a`/`ylim_b`
    override the panel limits so every figure in a sweep shares one scale. All
    default to None, which reproduces the standalone figure exactly."""
    kv, rr_kv, rr_w, life_no, life_yes = _compute_split(base, I)
    kv_gb, rr_kv_gb, rr_w_gb = kv / 1e9, rr_kv / 1e9, rr_w / 1e9
    # Total write = the three bars summed. This is the quantity the "with RR"
    # lifetime in (b) is derived from, hence the shared C_LIFE_YES red.
    total_gb = kv_gb + rr_kv_gb + rr_w_gb
    x = np.arange(len(OUTPUTS))
    labels = [_fmt_tok(o) for o in OUTPUTS]

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(FIG_W, FIG_H),
                                   layout="constrained")
    fig.get_layout_engine().set(w_pad=0.03, wspace=0.04,
                                h_pad=0.02, hspace=0.0)

    # ============ (a) write volume: KV store | RR (KV) | RR (Weight) ============
    lim_a = ylim_a or (1e-1, 1e6)
    ylo = lim_a[0]
    w = 0.27
    b_kv = axA.bar(x - w, kv_gb, w, color=C_KV,
                   edgecolor="white", linewidth=0.4, zorder=3)
    b_rw = axA.bar(x, rr_w_gb, w, color=C_RR_W,
                   edgecolor="white", linewidth=0.4, zorder=3)
    b_rk = axA.bar(x + w, rr_kv_gb, w, color=C_RR_KV,
                   edgecolor="white", linewidth=0.4, zorder=3)
    # Total marker over each group, in (b)'s "with RR" red so the two panels
    # visibly refer to the same quantity. Markers only — a connecting line
    # would read as a fourth series rather than a per-group total.
    (m_tot,) = axA.plot(x, total_gb, linestyle="none", marker="D",
                        markersize=6.0 * _S, color=C_LIFE_YES,
                        markeredgecolor="white", markeredgewidth=0.7, zorder=8)
    # A log axis cannot render a zero-height bar; mark those slots explicitly so
    # an absent bar is not read as missing data.
    for xi, v in zip(x, rr_kv_gb):
        if v <= 0:
            axA.text(xi + w, ylo * 1.35, "0", ha="center", va="bottom",
                     fontsize=FS_ZERO, color=C_RR_KV, fontweight="bold",
                     zorder=5)

    axA.set_yscale("log")
    axA.set_ylim(*lim_a)
    axA.set_ylabel("per-HBF write  [GB]", fontsize=FS_LABEL)
    axA.set_xlabel("output length  [tokens]\n" + PANEL_A, fontsize=FS_PANEL)
    axA.xaxis.label.set_linespacing(CAPTION_LINESPACING)
    axA.set_xticks(x)
    axA.set_xticklabels(labels, fontsize=FS_TICK)
    axA.tick_params(labelsize=FS_TICK)
    axA.grid(axis="y", alpha=0.22, which="major", zorder=0)
    # Handles passed explicitly: get_legend_handles_labels() collects Line2D
    # before BarContainer, which would float "Total" to the front.
    axA.legend([b_kv, b_rw, b_rk, m_tot],
               ["KV store", "RR (Weight)", "RR (KV)", "Total"],
               loc="upper left", fontsize=FS_LEG, framealpha=0.95,
               ncol=2, columnspacing=0.9, handlelength=1.2,
               handletextpad=0.4, borderaxespad=0.4, labelspacing=0.3)

    # ================= (b) lifetime: ideal vs actual threshold =================
    axB.plot(x, life_no, color=C_LIFE_NO, linestyle="--", linewidth=1.6,
             marker="o", markersize=4.2,
             label=r"RR threshold = $\infty$ (ideal)", zorder=6)
    axB.plot(x, life_yes, color=C_LIFE_YES, linestyle="-", linewidth=1.6,
             marker="s", markersize=4.2,
             label=_th_legend_label(), zorder=6)
    axB.set_yscale("log")
    axB.set_ylim(*(ylim_b or (1e0, 1e4)))
    axB.set_ylabel("HBF lifetime  [years]", fontsize=FS_LABEL)
    axB.set_xlabel("output length  [tokens]\n" + PANEL_B, fontsize=FS_PANEL)
    axB.xaxis.label.set_linespacing(CAPTION_LINESPACING)
    axB.set_xticks(x)
    axB.set_xticklabels(labels, fontsize=FS_TICK)
    axB.tick_params(labelsize=FS_TICK)
    axB.grid(axis="y", alpha=0.22, which="major", zorder=0)
    axB.axhline(5.0, color=C_LIMIT, linestyle="--", linewidth=1.1, zorder=4)
    axB.text(0.985, 6.6, "5-year limit", va="bottom", ha="right",
             fontsize=FS_LIMIT, fontweight="bold", color=C_LIMIT,
             transform=axB.get_yaxis_transform(), zorder=7)
    axB.legend(loc="upper left", fontsize=FS_LEG, framealpha=0.95,
               ncol=1, handlelength=1.8, handletextpad=0.5,
               borderaxespad=0.4, labelspacing=0.3)

    # No suptitle: the input length and batch are recorded in the folder's
    # README.md instead, so the figure carries only what a paper caption needs.
    stem = "write_lifetime_total_v5"
    if fname_tag is not None:
        stem += f"__{fname_tag}"
    out = out_dir / f"{stem}__I={_fmt_tok(I)}.png"
    fig.savefig(out, dpi=400)
    plt.close(fig)
    return out


def _write_readme(out_dir: Path, base):
    """Record the settings the figures no longer print on themselves.

    Generated from the live config objects rather than hand-written, so the
    README cannot drift from what the figures were actually rendered with."""
    h, m, rc = base.hbf, base.model, base.reclaim
    rows = [
        ("Model", f"{m.name} ({m.num_layers} layers, {m.num_kv_heads} KV heads, "
                  f"head_dim {m.head_dim}, FP16)"),
        ("Batch / concurrency", f"{B}"),
        ("Input lengths", ", ".join(_fmt_tok(i) for i in INPUTS) + "  (one figure each)"),
        ("Output lengths", ", ".join(_fmt_tok(o) for o in OUTPUTS) + "  (x axis)"),
        ("Weights", f"resident on HBF (`weights_storage=\"{h.weights_storage}\"`)"),
        ("NAND", f"{h.pages_per_block // 4}-layer, pages_per_block = {h.pages_per_block}, "
                 f"page = {h.page_size_bytes // 1024} KB, tR = {h.nand_tR_ns / 1000:g} us"),
        ("RR threshold", f"{rc.threshold_page_reads:,} page reads per block"),
        ("P/E budget", f"{rc.pe_cycle_limit:,} cycles per block (SLC)"),
        ("HBF", f"{h.bandwidth_bytes_per_sec / 1e12:g} TB/s, "
                f"{h.capacity_bytes // 1024**3} GiB, "
                f"{h.stacks_per_hbf} stacks x {h.dies_per_stack} dies x "
                f"{h.planes_per_die} planes = {h.total_planes_per_hbf} planes"),
        ("System", f"{h.num_gpus} GPUs x {h.hbfs_per_gpu} HBF each"),
        ("Figure size", f"{FIG_W:g} x {FIG_H:g} in"
                        + ("  (LaTeX `figure*`, full text width)" if not WSUFFIX else "")),
    ]
    width = max(len(k) for k, _ in rows)
    lines = [
        f"# {out_dir.name}",
        "",
        "Per-HBF write volume and the HBF lifetime it implies, versus output length.",
        "One figure per input length, PNG at 400 dpi.",
        "",
        "## Panels",
        "",
        f"- **(a) Write overhead** — per-HBF write volume as grouped bars:",
        f"  `KV store`, `RR (Weight)`,",
        f"  `RR (KV)`. The red diamond is their sum. Log axis, so the bars are grouped",
        f"  rather than stacked. A slot marked `0` means KV reclaim never fired: under",
        f"  the integer-event model (`floor(reads / threshold)`) no KV block crossed the",
        f"  threshold within the decode window — it is a real zero, not missing data.",
        f"- **(b) HBF lifetime** — lifetime under KV writes only",
        f"  (`RR threshold = inf`, the ideal with no read-disturb reclaim) versus under",
        f"  KV writes plus all reclaim (`RR threshold = {rc.threshold_page_reads:,}`).",
        f"  The dashed red line is the 5-year target.",
        "",
        "## Settings",
        "",
    ]
    lines += [f"| {'Parameter'.ljust(width)} | Value |",
              f"| {'-' * width} | --- |"]
    lines += [f"| {k.ljust(width)} | {v} |" for k, v in rows]
    lines += [
        "",
        "## Regenerate",
        "",
        "```bash",
        f"HBF_LAYERS={h.pages_per_block // 4} "
        + (f"RR_THRESHOLD={rc.threshold_page_reads:g} " if rc.threshold_page_reads != 1_000_000 else "")
        + (f"FIG_W={FIG_W:g} FIG_H={FIG_H:g} " if WSUFFIX else "")
        + f"python3.11 {pf.SCRIPT_DIR_NAME}/paper_figs_total_v5.py",
        "```",
        "",
        f"This file is written by `{pf.SCRIPT_DIR_NAME}/paper_figs_total_v5.py`; "
        "edit the script, not this file.",
        "",
    ]
    path = out_dir / "README.md"
    path.write_text("\n".join(lines))
    return path


def main():
    base = replace(DEFAULT_BASE,
                   hbf=replace(DEFAULT_BASE.hbf, pages_per_block=PPB,
                               weights_storage="hbf"),
                   reclaim=replace(DEFAULT_BASE.reclaim,
                                   threshold_page_reads=THRESHOLD))
    out_dir = pf.WRITE_LIFETIME_DIR / f"write_lifetime_total_v5{SUFFIX}{WSUFFIX}"
    out_dir.mkdir(parents=True, exist_ok=True)
    for I in INPUTS:
        p = _fig_for_input(base, I, out_dir)
        print(f"  wrote {p}")
    print(f"  wrote {_write_readme(out_dir, base)}")


if __name__ == "__main__":
    main()
