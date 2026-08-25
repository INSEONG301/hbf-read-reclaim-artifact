"""Parameter sweep driver.

Given a dict of {param_name: [values...]}, runs the simulator over the
cartesian product and returns a list of SimulationResult plus a CSV row table.

Recognised sweep keys (any subset; missing keys take their default from the
relevant config dataclass / the supplied base SimulationConfig):

  Model / workload:
    model                      ModelConfig instance OR model name in MODEL_REGISTRY
    input_tokens, output_tokens, batch_size

  HBF hardware:
    page_size_bytes, pages_per_block, planes_per_die,
    dies_per_stack, stacks_per_hbf,
    hbfs_per_gpu, num_gpus, capacity_bytes,
    bandwidth_bytes_per_sec, bandwidth_utilization, nand_tR_ns

  Reclaim:
    reclaim_threshold

  Policy:
    write_policy   ("plane_balanced_block_fill" or "uniform")
"""
from __future__ import annotations

import csv
import itertools
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

from .configs import (
    HBFConfig, ModelConfig, WorkloadConfig, ReclaimConfig, SimulationConfig,
)
from .models import MODEL_REGISTRY
from .simulator import simulate, SimulationResult


_HBF_KEYS = {
    "page_size_bytes", "pages_per_block",
    "planes_per_die", "dies_per_stack", "stacks_per_hbf",
    "hbfs_per_gpu", "num_gpus", "capacity_bytes",
    "bandwidth_bytes_per_sec", "bandwidth_utilization", "nand_tR_ns",
    "hbm_bandwidth_bytes_per_sec", "weights_storage",
}
_WORKLOAD_KEYS = {"input_tokens", "output_tokens", "batch_size"}
_RECLAIM_KEYS = {"reclaim_threshold", "pe_cycle_limit"}


def _resolve_model(v) -> ModelConfig:
    if isinstance(v, ModelConfig):
        return v
    if isinstance(v, str):
        if v not in MODEL_REGISTRY:
            raise KeyError(f"Unknown model '{v}'. Known: {sorted(MODEL_REGISTRY)}")
        return MODEL_REGISTRY[v]
    raise TypeError(f"model must be ModelConfig or registry name, got {type(v)}")


def _apply_point(base: SimulationConfig, point: Mapping[str, Any]) -> SimulationConfig:
    hbf_kwargs, work_kwargs, reclaim_kwargs = {}, {}, {}
    model = base.model
    policy = base.write_policy
    for k, v in point.items():
        if k == "model":
            model = _resolve_model(v)
        elif k == "write_policy":
            policy = v
        elif k in _HBF_KEYS:
            hbf_kwargs[k] = v
        elif k in _WORKLOAD_KEYS:
            work_kwargs[k] = v
        elif k == "reclaim_threshold":
            reclaim_kwargs["threshold_page_reads"] = v
        elif k == "pe_cycle_limit":
            reclaim_kwargs["pe_cycle_limit"] = v
        else:
            raise KeyError(f"Unknown sweep key '{k}'")

    # Auto-link HBM bandwidth to HBF bandwidth: when a sweep varies the HBF
    # bandwidth but does not explicitly set the HBM bandwidth, the HBM moves
    # together (HBM and HBF are both 8 stacks/GPU with matched per-stack BW).
    if "bandwidth_bytes_per_sec" in point and "hbm_bandwidth_bytes_per_sec" not in point:
        hbf_kwargs["hbm_bandwidth_bytes_per_sec"] = point["bandwidth_bytes_per_sec"]

    return SimulationConfig(
        model=model,
        hbf=replace(base.hbf, **hbf_kwargs),
        workload=replace(base.workload, **work_kwargs),
        reclaim=replace(base.reclaim, **reclaim_kwargs),
        write_policy=policy,
    )


def sweep_grid(grid: Mapping[str, Iterable[Any]]) -> List[Dict[str, Any]]:
    """Cartesian product of grid as a list of point dicts."""
    keys = list(grid.keys())
    values = [list(v) for v in grid.values()]
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


def run_sweep(
    base: SimulationConfig,
    grid: Mapping[str, Iterable[Any]],
    csv_path: str | Path | None = None,
) -> List[SimulationResult]:
    """Run every grid point, optionally writing a CSV.

    Returns the list of `SimulationResult` (includes block-level stats).
    """
    points = sweep_grid(grid)
    results: List[SimulationResult] = []
    rows: List[dict] = []
    for pt in points:
        cfg = _apply_point(base, pt)
        res = simulate(cfg)
        results.append(res)
        rows.append(res.to_row())

    if csv_path is not None and rows:
        csv_path = Path(csv_path)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = sorted({k for row in rows for k in row.keys()})
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in rows:
                w.writerow(r)
    return results
