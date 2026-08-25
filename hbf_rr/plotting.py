"""Matplotlib helpers for sweep outputs.

Two plot types:
  - `plot_sweep(results, x_key, y_key, line_key=None, ...)`  — 1-D sweep plot.
  - `plot_heatmap(results, x_key, y_key, value_key, ...)`    — 2-D heatmap,
    optionally faceted into multiple PNGs by a 3rd key.

`make_default_plots` produces a curated set of 1-D plots for one finished
sweep. Each plot annotates the *fixed* configuration in a small footer so the
sweep context is always visible.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .simulator import SimulationResult


# Keys that may end up on x / y / line / facet axes.
_SWEEP_KEYS = [
    "model_name", "write_policy", "weights_storage",
    "input_tokens", "output_tokens", "batch_size",
    "page_size_bytes", "pages_per_block",
    "planes_per_die", "dies_per_stack", "stacks_per_hbf",
    "total_dies_per_hbf", "total_planes_per_hbf",
    "hbfs_per_gpu", "num_gpus",
    "hbf_bandwidth_bytes_per_sec", "bandwidth_utilization",
    "hbm_bandwidth_bytes_per_sec", "nand_tR_ns",
    "reclaim_threshold",
]


# Curated 1-D metrics. (key, label, log_y?)
CORE_METRICS: List[tuple[str, str, bool]] = [
    ("decode_steps_between_reclaims_hottest", "decode steps between reclaims (hot block)", True),
    ("hottest_block_reclaim_period_s",        "hot-block reclaim period [s]",               True),
    ("first_reclaim_time_s",                  "time to first reclaim [s]",                  True),
    ("reclaim_write_bw_per_hbf_bytes_per_sec", "avg reclaim write BW per HBF [B/s]",        True),
    ("avg_token_latency_ms",                  "avg per-token decode latency [ms]",          False),
    ("planes_needed_for_target_bw",           "planes needed to sustain HBF BW [count]",    True),
    ("plane_oversubscription",                "planes needed / planes available",           False),
]


# Keys included in the "fixed config" caption shown beneath every plot.
_CAPTION_KEYS = [
    "model_name", "write_policy",
    "input_tokens", "output_tokens", "batch_size",
    "page_size_bytes", "pages_per_block",
    "planes_per_die", "dies_per_stack", "stacks_per_hbf",
    "hbfs_per_gpu", "num_gpus",
    "hbf_bandwidth_bytes_per_sec", "bandwidth_utilization",
    "hbm_bandwidth_bytes_per_sec", "weights_storage", "nand_tR_ns",
    "reclaim_threshold",
]


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _row(r: SimulationResult) -> dict:
    return r.to_row()


def _fmt(v) -> str:
    """Compact value formatting for captions / cell labels."""
    if v is None:
        return "—"
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, float):
        if not math.isfinite(v):
            return "inf"
        if abs(v) >= 1e9:
            return f"{v:.2e}"
        if abs(v) >= 1000:
            return f"{v:.0f}"
        if abs(v) < 0.01 and v != 0:
            return f"{v:.2e}"
        return f"{v:.2g}"
    return str(v)


# Unit-aware formatters per metric key suffix.

def _fmt_bytes_per_sec(v: float) -> tuple[str, str]:
    """Format bytes/sec into a short label and unit string (used for colorbar)."""
    if v is None or not math.isfinite(v):
        return ("—", "")
    units = [(1e12, "TB/s"), (1e9, "GB/s"), (1e6, "MB/s"), (1e3, "KB/s"), (1, "B/s")]
    for scale, unit in units:
        if abs(v) >= scale:
            return (f"{v/scale:.2f} {unit}", unit)
    return (f"{v:.2f} B/s", "B/s")


def _fmt_seconds(v: float) -> str:
    if v is None or not math.isfinite(v):
        return "—"
    if abs(v) < 1e-3:
        return f"{v*1e6:.1f} us"
    if abs(v) < 1.0:
        return f"{v*1e3:.1f} ms"
    if abs(v) < 60:
        return f"{v:.2g} s"
    if abs(v) < 3600:
        return f"{v/60:.1f} min"
    return f"{v/3600:.1f} h"


def _fmt_bytes(v: float) -> tuple[str, str]:
    if v is None or not math.isfinite(v):
        return ("—", "")
    units = [(1024 ** 4, "TB"), (1024 ** 3, "GB"), (1024 ** 2, "MB"), (1024, "KB"), (1, "B")]
    for scale, unit in units:
        if abs(v) >= scale:
            return (f"{v/scale:.2f} {unit}", unit)
    return (f"{v:.0f} B", "B")


def _fmt_cell(v: float, value_key: str) -> str:
    """Choose a unit-aware compact string for a heatmap cell."""
    if v is None or not math.isfinite(v):
        return "—"
    if value_key.endswith("bytes_per_sec"):
        return _fmt_bytes_per_sec(v)[0]
    if value_key.endswith("_bytes"):
        return _fmt_bytes(v)[0]
    if value_key.endswith("_s") and "period" in value_key or value_key.endswith("_time_s"):
        return _fmt_seconds(v)
    return _fmt(v)


def _colorbar_unit(value_key: str, sample_max: float) -> str:
    """Return a unit string suitable for the colorbar label."""
    if value_key.endswith("bytes_per_sec"):
        return _fmt_bytes_per_sec(sample_max)[1] or "B/s"
    if value_key.endswith("_s") and ("period" in value_key or "time" in value_key):
        if sample_max < 1: return "s (sub-second)"
        if sample_max < 60: return "s"
        return "s"
    return ""


def _config_caption(rows: List[dict], exclude: Sequence[str]) -> str:
    """Build a single-line 'fixed parameters' caption from the first row.

    Any key whose value varies across `rows` is auto-excluded too (so e.g. a
    bandwidth sweep that auto-links HBM doesn't pretend HBM is fixed).
    """
    if not rows:
        return ""
    r0 = rows[0]
    ex = set(exclude)
    for k in _CAPTION_KEYS:
        vals = {r.get(k) for r in rows}
        if len(vals) > 1:
            ex.add(k)
    bits: list[str] = []
    for k in _CAPTION_KEYS:
        if k in ex:
            continue
        v = r0.get(k)
        if v is None:
            continue
        bits.append(f"{k}={_fmt(v)}")
    return "  ".join(bits)


def _wrap_caption(caption: str, max_chars_per_line: int = 95) -> str:
    """Wrap caption text into N lines, splitting on double-space boundaries."""
    if not caption or len(caption) <= max_chars_per_line:
        return caption
    bits = caption.split("  ")
    lines: list[str] = []
    cur = ""
    for b in bits:
        if not cur:
            cur = b
        elif len(cur) + 2 + len(b) <= max_chars_per_line:
            cur = cur + "  " + b
        else:
            lines.append(cur)
            cur = b
    if cur:
        lines.append(cur)
    return "\n".join(lines)


def _attach_caption(fig, caption: str) -> None:
    """Reserve bottom margin and stamp the wrapped caption there.

    fig.tight_layout() does NOT account for fig.text, so we have to control the
    bottom margin manually. We measure number of lines and reserve accordingly.
    """
    if not caption:
        return
    wrapped = _wrap_caption(caption)
    n_lines = wrapped.count("\n") + 1
    # Bottom margin reserved: ~0.10 base for xlabel + 0.035 per caption line.
    bottom = 0.12 + 0.038 * n_lines
    fig.subplots_adjust(bottom=bottom)
    fig.text(0.5, 0.01, wrapped, ha="center", va="bottom",
             fontsize=7, alpha=0.7)


def _has_variation(values: Sequence) -> bool:
    seen: set = set()
    for v in values:
        if v is None:
            continue
        seen.add(v)
        if len(seen) > 1:
            return True
    return False


def _detect_axes(rows: List[dict]) -> List[str]:
    return [k for k in _SWEEP_KEYS if len({row.get(k) for row in rows}) > 1]


# ---------------------------------------------------------------------------
# 1-D line sweep
# ---------------------------------------------------------------------------

def plot_sweep(
    results: List[SimulationResult],
    x_key: str,
    y_key: str,
    out_path: Path,
    line_key: Optional[str] = None,
    title: Optional[str] = None,
    y_label: Optional[str] = None,
    log_x: bool = False,
    log_y: bool = False,
    max_lines: int = 12,
) -> Optional[Path]:
    rows = [_row(r) for r in results]
    if not _has_variation([row.get(y_key) for row in rows]):
        return None

    series: dict = {}
    for row in rows:
        x = row.get(x_key)
        y = row.get(y_key)
        if x is None or y is None:
            continue
        gk = row.get(line_key) if line_key else None
        series.setdefault(gk, []).append((x, y))

    if len(series) > max_lines:
        return None

    # Optional rescale for B/s metrics so y-axis numbers are readable.
    y_scale = 1.0
    y_unit = ""
    sample_y = max([row.get(y_key) or 0 for row in rows] or [0])
    if y_key.endswith("bytes_per_sec") and math.isfinite(sample_y) and sample_y > 0:
        _, y_unit = _fmt_bytes_per_sec(sample_y)
        unit_to_scale = {"B/s": 1, "KB/s": 1e3, "MB/s": 1e6, "GB/s": 1e9, "TB/s": 1e12}
        y_scale = unit_to_scale.get(y_unit, 1.0)

    fig, ax = plt.subplots(figsize=(8, 5))
    cmap = plt.get_cmap("viridis")
    keys_sorted = sorted(series.keys(), key=lambda k: (k is None, k))
    for i, gk in enumerate(keys_sorted):
        pts = sorted(series[gk], key=lambda p: p[0])
        xs = [p[0] for p in pts]
        ys = [p[1] / y_scale for p in pts]
        label = f"{line_key}={_fmt(gk)}" if (line_key and gk is not None) else None
        color = cmap(i / max(1, len(keys_sorted) - 1))
        ax.plot(xs, ys, marker="o", label=label, color=color)

    ax.set_xlabel(x_key, fontsize=11)
    y_axis_label = y_label or y_key
    if y_unit:
        # If the label already carries a "[B/s]" suffix from CORE_METRICS, swap
        # it for the rescaled unit so we don't end up with "[B/s] [MB/s]".
        import re as _re
        y_axis_label = _re.sub(r"\s*\[[^\]]*B/s[^\]]*\]\s*$", "", y_axis_label)
        y_axis_label = f"{y_axis_label} [{y_unit}]"
    ax.set_ylabel(y_axis_label, fontsize=11)
    if log_y:
        ax.set_yscale("log")
    if log_x:
        ax.set_xscale("log")
    ax.grid(True, alpha=0.3, which="both")
    ax.tick_params(labelsize=10)
    if line_key and any(k is not None for k in keys_sorted):
        ax.legend(fontsize=9, loc="best", framealpha=0.9)

    if title:
        fig.suptitle(title, fontsize=13, y=0.985)

    exclude = [x_key]
    if line_key:
        exclude.append(line_key)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    _attach_caption(fig, _config_caption(rows, exclude))

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


def plot_bar(
    results: List[SimulationResult],
    names: List[str],
    value_key: str,
    out_path: Path,
    title: Optional[str] = None,
    value_label: Optional[str] = None,
    log_y: bool = False,
    annotate: bool = True,
) -> Optional[Path]:
    """Bar chart of one metric across a list of named results.

    Use case: a small set of explicit (named) workload points such as the
    Deep Research input/output pairs. Bars are colored by viridis position and
    each bar gets a numeric annotation in the metric's natural unit
    (ms / s / min / MB/s etc. via `_fmt_cell`).
    """
    if not results:
        return None
    rows = [_row(r) for r in results]
    values = [row.get(value_key) for row in rows]
    if all(v is None for v in values):
        return None

    fig, ax = plt.subplots(figsize=(max(8, 1.3 * len(names)), 5))
    cmap = plt.get_cmap("viridis")
    colors = [cmap(i / max(1, len(names) - 1)) for i in range(len(names))]
    xs = list(range(len(names)))
    plot_values = [v if (v is not None and (not log_y or v > 0)) else None for v in values]
    bars = ax.bar(xs, [v if v is not None else 0 for v in plot_values],
                  color=colors)

    ax.set_xticks(xs)
    ax.set_xticklabels(names, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel(value_label or value_key, fontsize=11)
    if log_y:
        ax.set_yscale("log")
    ax.grid(True, axis="y", alpha=0.3, which="both")
    ax.tick_params(labelsize=10)

    if annotate:
        for bar, v in zip(bars, values):
            if v is None:
                continue
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h,
                    _fmt_cell(v, value_key),
                    ha="center", va="bottom", fontsize=9)

    if title:
        fig.suptitle(title, fontsize=13, y=0.985)

    # Caption skips workload axes (those vary per bar) plus any varying key.
    exclude = ["input_tokens", "output_tokens"]
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    _attach_caption(fig, _config_caption(rows, exclude))

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


def plot_planes_demand(
    results: List[SimulationResult],
    x_key: str,
    out_path: Path,
    line_key: Optional[str] = None,
    title: Optional[str] = None,
    log_x: bool = False,
    log_y: bool = True,
) -> Optional[Path]:
    """Overlay `planes_needed_for_target_bw` (required) and
    `total_planes_per_hbf` (available) on the same y-axis so the user can see
    at what x-value the HBF can actually deliver its nominal bandwidth.

    Solid line = planes_needed, dashed line = total_planes_per_hbf (reference).
    Their crossing point marks the feasibility boundary.
    """
    rows = [_row(r) for r in results]
    if not rows:
        return None

    series_needed: dict = {}
    series_avail: dict = {}
    for row in rows:
        x = row.get(x_key)
        need = row.get("planes_needed_for_target_bw")
        avail = row.get("total_planes_per_hbf")
        if x is None or need is None or avail is None:
            continue
        gk = row.get(line_key) if line_key else None
        series_needed.setdefault(gk, []).append((x, need))
        series_avail.setdefault(gk, []).append((x, avail))

    if not series_needed:
        return None

    fig, ax = plt.subplots(figsize=(8, 5))
    cmap = plt.get_cmap("viridis")
    keys = sorted(series_needed.keys(), key=lambda k: (k is None, k))
    for i, gk in enumerate(keys):
        pts_n = sorted(series_needed[gk], key=lambda p: p[0])
        pts_a = sorted(series_avail[gk], key=lambda p: p[0])
        xs = [p[0] for p in pts_n]
        ys_n = [p[1] for p in pts_n]
        ys_a = [p[1] for p in pts_a]
        color = cmap(i / max(1, len(keys) - 1))
        suffix = f"  ({line_key}={_fmt(gk)})" if (line_key and gk is not None) else ""
        ax.plot(xs, ys_n, marker="o", linestyle="-",
                label=f"planes needed{suffix}", color=color)
        ax.plot(xs, ys_a, marker="s", linestyle="--",
                label=f"planes available{suffix}", color=color, alpha=0.55)

    ax.set_xlabel(x_key, fontsize=11)
    ax.set_ylabel("planes per HBF [count]", fontsize=11)
    if title:
        fig.suptitle(title, fontsize=13, y=0.985)
    if log_x:
        ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")
    ax.grid(True, alpha=0.3, which="both")
    ax.tick_params(labelsize=10)
    ax.legend(fontsize=8, loc="best", framealpha=0.9)

    exclude = [x_key]
    if line_key:
        exclude.append(line_key)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    _attach_caption(fig, _config_caption(rows, exclude))

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


def make_default_plots(
    results: List[SimulationResult],
    out_dir: str | Path,
    x_key: Optional[str] = None,
    line_key: Optional[str] = None,
    title_prefix: str = "",
) -> List[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [_row(r) for r in results]
    if not rows:
        return []

    varying = sorted(_detect_axes(rows), key=lambda k: -len({row.get(k) for row in rows}))
    if x_key is None:
        x_key = varying[0] if varying else None
    if x_key is None:
        return []
    if line_key is None:
        remaining = [k for k in varying if k != x_key]
        line_key = remaining[0] if remaining else None

    log_x_keys = {
        "input_tokens", "output_tokens",
        "hbf_bandwidth_bytes_per_sec", "hbm_bandwidth_bytes_per_sec",
        "reclaim_threshold", "page_size_bytes", "pages_per_block",
        "total_planes_per_hbf", "nand_tR_ns",
    }
    written: List[Path] = []
    for y_key, label, log_y in CORE_METRICS:
        out_path = out_dir / f"{y_key}.png"
        title = f"{title_prefix}{label} vs {x_key}".strip()
        p = plot_sweep(
            results, x_key=x_key, y_key=y_key,
            out_path=out_path, line_key=line_key,
            title=title, y_label=label,
            log_x=(x_key in log_x_keys), log_y=log_y,
        )
        if p is not None:
            written.append(p)
    return written


# ---------------------------------------------------------------------------
# 2-D heatmap
# ---------------------------------------------------------------------------

def plot_heatmap(
    results: List[SimulationResult],
    x_key: str,
    y_key: str,
    value_key: str,
    out_dir: str | Path,
    facet_key: Optional[str] = None,
    title_prefix: str = "",
    value_label: Optional[str] = None,
    log_value: bool = False,
    cmap: str = "viridis",
    annotate: bool = True,
) -> List[Path]:
    """Draw a heatmap of `value_key` vs (x_key, y_key).

    If `facet_key` is given, one PNG is written per facet value
    (filename: `<value_key>__<facet>=<val>.png`). Otherwise a single PNG.
    `log_value=True` plots log10(value); non-positive / nan / inf cells stay blank.
    Each cell is annotated with its numeric value.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [_row(r) for r in results]

    facets: dict = {}
    for row in rows:
        fv = row.get(facet_key) if facet_key else None
        facets.setdefault(fv, []).append(row)

    written: List[Path] = []
    for fv in sorted(facets.keys(), key=lambda v: (v is None, v)):
        facet_rows = facets[fv]
        xs = sorted({r[x_key] for r in facet_rows if r.get(x_key) is not None})
        ys = sorted({r[y_key] for r in facet_rows if r.get(y_key) is not None})
        if not xs or not ys:
            continue

        raw = np.full((len(ys), len(xs)), np.nan)
        for r in facet_rows:
            x = r.get(x_key); y = r.get(y_key); v = r.get(value_key)
            if x is None or y is None or v is None:
                continue
            if isinstance(v, float) and not math.isfinite(v):
                continue
            ix = xs.index(x); iy = ys.index(y)
            raw[iy, ix] = float(v)

        if log_value:
            plot_grid = np.full_like(raw, np.nan)
            mask = (raw > 0) & np.isfinite(raw)
            plot_grid[mask] = np.log10(raw[mask])
        else:
            plot_grid = raw

        # Generous cell size for readable annotations.
        cell_w, cell_h = 1.4, 0.85
        fig_w = max(8.0, 2.0 + cell_w * len(xs))
        fig_h = max(5.0, 2.0 + cell_h * len(ys))
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        im = ax.imshow(plot_grid, aspect="auto", origin="lower", cmap=cmap)

        def _axis_label(v, key):
            if key.endswith("_bytes"):
                return _fmt_bytes(v)[0]
            if key.endswith("bytes_per_sec"):
                return _fmt_bytes_per_sec(v)[0]
            return _fmt(v)

        ax.set_xticks(range(len(xs)))
        ax.set_xticklabels([_axis_label(x, x_key) for x in xs],
                           rotation=35, ha="right", fontsize=10)
        ax.set_yticks(range(len(ys)))
        ax.set_yticklabels([_axis_label(y, y_key) for y in ys], fontsize=10)
        ax.set_xlabel(x_key, fontsize=11)
        ax.set_ylabel(y_key, fontsize=11)

        cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
        finite_raw = raw[np.isfinite(raw)]
        cb_unit = _colorbar_unit(value_key, float(np.nanmax(finite_raw))) if finite_raw.size else ""
        cb_label = value_label or value_key
        if log_value:
            cb.set_label(f"log10  ({cb_label})", fontsize=10)
        else:
            if cb_unit and cb_unit not in cb_label:
                cb_label = f"{cb_label} [{cb_unit}]"
            cb.set_label(cb_label, fontsize=10)
        cb.ax.tick_params(labelsize=9)

        # Per-cell labels (unit-aware, larger font)
        if annotate:
            finite = plot_grid[np.isfinite(plot_grid)]
            if finite.size:
                vmin, vmax = float(np.nanmin(plot_grid)), float(np.nanmax(plot_grid))
                mid = (vmin + vmax) / 2 if vmax > vmin else vmin
            else:
                mid = 0.0
            for iy in range(raw.shape[0]):
                for ix in range(raw.shape[1]):
                    v_raw = raw[iy, ix]
                    if not np.isfinite(v_raw):
                        ax.text(ix, iy, "—", ha="center", va="center",
                                color="grey", fontsize=10)
                        continue
                    v_plot = plot_grid[iy, ix]
                    txt_color = "white" if (np.isfinite(v_plot) and v_plot < mid) else "black"
                    ax.text(ix, iy, _fmt_cell(v_raw, value_key),
                            ha="center", va="center",
                            color=txt_color, fontsize=10)

        # Title via suptitle so it never collides with the axes.
        title_bits = [t for t in [title_prefix, value_label or value_key] if t]
        if facet_key and fv is not None:
            title_bits.append(f"{facet_key} = {_fmt(fv)}")
        fig.suptitle("   |   ".join(title_bits), fontsize=13, y=0.985)

        exclude = [x_key, y_key]
        if facet_key:
            exclude.append(facet_key)
        # Use tight_layout first, then attach caption (which reserves bottom).
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
        _attach_caption(fig, _config_caption(facet_rows, exclude))

        if facet_key and fv is not None:
            fname = f"{value_key}__{facet_key}={_fmt(fv)}.png"
        else:
            fname = f"{value_key}.png"
        out_path = out_dir / fname
        fig.savefig(out_path, dpi=140)
        plt.close(fig)
        written.append(out_path)

    return written
