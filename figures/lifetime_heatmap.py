"""Paper figure: HBF lifetime heatmaps over (input x output) length.

2 x 2 panel: rows = model (LLaMA-3.1-405B, Qwen3-235B-A22B),
columns = reclaim OFF (KV writes only) vs reclaim ON (KV + KV-reclaim +
weight-reclaim). x-axis = input length, y-axis = output length. A shared
log color scale lets all four panels be compared directly; a red contour
marks the 5-year lifetime target.

Fixed batch = 16 (matches the write_lifetime paper figure), 112-layer NAND
(pages_per_block = 448), weights resident on HBF.
"""
from __future__ import annotations

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
from dataclasses import replace
from pathlib import Path

from defaults import DEFAULT_BASE
from hbf_rr.simulator import simulate
from hbf_rr.models import LLAMA_3_1_405B, QWEN3_235B_A22B

B = 16
# NAND layer count via HBF_LAYERS (default 112); PPB = layers x 4.
LAYERS = int(os.environ.get("HBF_LAYERS", "112"))
PPB = LAYERS * 4
SUFFIX = "" if LAYERS == 112 else f"__{LAYERS}layer"
# NAND page size via PAGE_SIZE bytes (default 2048); non-default gets a folder tag.
PAGE_SIZE = int(os.environ.get("PAGE_SIZE", "4096"))
if PAGE_SIZE != 2048:
    SUFFIX += f"__{PAGE_SIZE // 1024}KBpage"
INPUTS = [1_024, 4_096, 16_384, 65_536, 262_144, 1_048_576]   # 1K..1M, x4 steps
OUTPUTS = [512, 2_048, 8_192, 32_768, 131_072, 524_288]       # 512..512K
MODELS = [LLAMA_3_1_405B, QWEN3_235B_A22B]
TARGET_YEARS = 5.0


def _fmt_tok(n: int) -> str:
    if n >= 1_048_576 and n % 1_048_576 == 0:
        return f"{n // 1_048_576}M"
    if n >= 1024 and n % 1024 == 0:
        return f"{n // 1024}K"
    return str(n)


def _grids(base):
    """Return {model_name: (life_no[O,I], life_yes[O,I])} in years."""
    out = {}
    for model in MODELS:
        no = np.zeros((len(OUTPUTS), len(INPUTS)))
        yes = np.zeros((len(OUTPUTS), len(INPUTS)))
        cfg0 = replace(base, model=model)
        for j, I in enumerate(INPUTS):
            for i, O in enumerate(OUTPUTS):
                cfg = replace(cfg0, workload=replace(cfg0.workload,
                              input_tokens=I, output_tokens=O, batch_size=B))
                r = simulate(cfg)
                no[i, j] = r.hbf_lifetime_years_kv_only
                yes[i, j] = r.hbf_lifetime_years
        out[model.name] = (no, yes)
        print(f"  {model.name}: no-RR {no.min():.2g}-{no.max():.2g} yr, "
              f"with-RR {yes.min():.2g}-{yes.max():.2g} yr")
    return out


def _panel(ax, grid, norm, title):
    # Diverging colour on log-lifetime, white pinned at the 5-year target:
    # below 5 yr -> red (bad), above 5 yr -> blue (good).
    im = ax.imshow(np.log10(grid), origin="lower", aspect="auto",
                   cmap="RdBu", norm=norm)
    ax.set_xticks(range(len(INPUTS)))
    ax.set_xticklabels([_fmt_tok(v) for v in INPUTS], fontsize=12)
    ax.set_yticks(range(len(OUTPUTS)))
    ax.set_yticklabels([_fmt_tok(v) for v in OUTPUTS], fontsize=12)
    ax.set_title(title, fontsize=15, fontweight="bold")
    # per-cell value annotations; white text only on the saturated extremes
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            v = grid[i, j]
            t = norm(np.log10(v))
            ax.text(j, i, f"{v:.3g}" if v < 100 else f"{v:.0f}",
                    ha="center", va="center", fontsize=11,
                    color="white" if (t < 0.18 or t > 0.82) else "black")
    return im


def main():
    base = replace(DEFAULT_BASE,
                   hbf=replace(DEFAULT_BASE.hbf, pages_per_block=PPB,
                               page_size_bytes=PAGE_SIZE, weights_storage="hbf"))
    data = _grids(base)

    allv = np.concatenate([np.concatenate([g[0].ravel(), g[1].ravel()])
                           for g in data.values()])
    vmin = max(allv[allv > 0].min(), 1e-2)
    vmax = allv.max()
    # Diverging norm on log10(years), white centred at the 5-year target.
    norm = TwoSlopeNorm(vmin=np.log10(vmin),
                        vcenter=np.log10(TARGET_YEARS),
                        vmax=np.log10(vmax))

    fig, axes = plt.subplots(2, 2, figsize=(11.2, 9.0))
    # column labels shown at the bottom (a) left / (b) right
    col_labels = [r"(a)  RR threshold = $\infty$ (ideal)",
                  r"(b)  RR threshold = $10^{6}$ (typical SLC)"]
    for r, model in enumerate(MODELS):
        no, yes = data[model.name]
        for c, grid in enumerate((no, yes)):
            ax = axes[r, c]
            im = _panel(ax, grid, norm, model.name.replace("LLaMA", "Llama"))
            if r == 1:
                ax.set_xlabel("input length  [tokens]", fontsize=15)
            if c == 0:
                ax.set_ylabel("output length  [tokens]", fontsize=15)
    # (a)/(b) RR-threshold labels below each column, on a shared baseline and
    # centred under each panel's full visual extent (incl. axis labels/ticks).
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    y_lab = axes[1, 0].get_position().y0 - 0.075
    x_off = [0.0, -0.025]                       # nudge (b) slightly left
    for c, lab in enumerate(col_labels):
        bb = axes[1, c].get_tightbbox(rend).transformed(inv)
        fig.text((bb.x0 + bb.x1) / 2 + x_off[c], y_lab, lab, ha="center",
                 va="top", fontsize=16, fontweight="bold")

    cbar = fig.colorbar(im, ax=axes, fraction=0.046, pad=0.03,
                        location="right")
    ticks_years = [0.5, 1, 2, 5, 10, 50, 100, 500]
    ticks_years = [t for t in ticks_years if vmin <= t <= vmax]
    cbar.set_ticks([np.log10(t) for t in ticks_years])
    cbar.set_ticklabels([f"{t:g}" for t in ticks_years])
    cbar.ax.tick_params(labelsize=11.5)
    cbar.set_label("HBF lifetime  [years]", fontsize=15)

    out_dir = Path(__file__).resolve().parent.parent / "plots" / "paper" / f"lifetime_heatmap{SUFFIX}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "lifetime_heatmap.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")

    # ---- CSV ----
    import csv
    with open(out_dir / "lifetime_heatmap_data.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["model", "input_tokens", "output_tokens", "batch",
                     "lifetime_years_no_reclaim", "lifetime_years_with_reclaim"])
        for model in MODELS:
            no, yes = data[model.name]
            for i, O in enumerate(OUTPUTS):
                for j, I in enumerate(INPUTS):
                    wr.writerow([model.name, I, O, B,
                                 f"{no[i, j]:.4g}", f"{yes[i, j]:.4g}"])
    print(f"  wrote {out_dir / 'lifetime_heatmap_data.csv'}")


if __name__ == "__main__":
    main()
