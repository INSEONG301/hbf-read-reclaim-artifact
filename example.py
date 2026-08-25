"""Minimal example: HBF read-reclaim endurance and latency for one workload.

Run:
    PYTHONPATH=. python3 example.py

Shows, for a single (model, HBF, workload) configuration:
  1. the per-HBF write bandwidth split into KV write / KV reclaim / weight reclaim,
  2. the resulting HBF lifetime with and without read-reclaim, and
  3. the per-token TBT distribution (p50 / p99 / max) with vs without reclaim,
     obtained from the event-driven request-stream model.
"""
from dataclasses import replace

from defaults import DEFAULT_BASE, _auto_rates_for_workload
from hbf_rr.simulator import simulate
from hbf_rr.queueing import simulate_request_stream, compute_capacity_max_concurrency

SPY = 365.25 * 86400

# ---- configuration -------------------------------------------------------
# 162-layer SLC NAND (pages_per_block = 4 x layers), model weights on HBF,
# LLaMA-3.1-405B (the default model), input = 64K, output = 32K, batch = 16.
LAYERS, INPUT, OUTPUT, BATCH = 162, 65_536, 32_768, 16

base = replace(
    DEFAULT_BASE,
    hbf=replace(DEFAULT_BASE.hbf, pages_per_block=LAYERS * 4, weights_storage="hbf"),
    workload=replace(DEFAULT_BASE.workload,
                     input_tokens=INPUT, output_tokens=OUTPUT, batch_size=BATCH),
)

# ---- 1 & 2: analytic write traffic + lifetime ----------------------------
r = simulate(base)
budget = base.reclaim.pe_cycle_limit * base.hbf.capacity_bytes   # writes the HBF can absorb

print(f"Model      : {base.model.name}")
print(f"NAND       : {LAYERS}-layer SLC, {base.hbf.pages_per_block} pages/block, "
      f"{base.hbf.capacity_bytes // 1024**3} GiB/HBF")
print(f"Workload   : input={INPUT}, output={OUTPUT}, batch={BATCH}\n")

print("per-HBF write bandwidth [GB/s]")
print(f"  KV write        : {r.kv_write_bw_per_hbf_bytes_per_sec/1e9:8.3f}")
print(f"  KV reclaim      : {r.reclaim_write_bw_per_hbf_bytes_per_sec/1e9:8.3f}")
print(f"  weight reclaim  : {r.weight_reclaim_write_bw_per_hbf_bytes_per_sec/1e9:8.3f}\n")

print("HBF lifetime [years]")
print(f"  no read-reclaim (KV writes only) : {r.hbf_lifetime_years_kv_only:8.2f}")
print(f"  weight reclaim only              : {r.hbf_lifetime_years_weight_reclaim_only:8.2f}")
print(f"  with read-reclaim (all sources)  : {r.hbf_lifetime_years:8.2f}\n")

# ---- 3: TBT distribution under a saturating request stream ---------------
cap = compute_capacity_max_concurrency(base)["max_concurrency_from_capacity"]
conc = max(1, min(cap, 256))
rate = _auto_rates_for_workload(INPUT, OUTPUT, conc, n=6)[-1]   # ~saturating rate
q = simulate_request_stream(base, request_rate_per_sec=rate,
                            n_requests=max(200, 3 * conc), max_concurrent=conc, seed=0)

print(f"TBT [ms] at concurrency={conc}, rate={rate:.4g} req/s")
print(f"  no reclaim   :  p50={q.tbt_p50_noreclaim_ms:6.1f}  "
      f"p99={q.tbt_p99_noreclaim_ms:6.1f}  max={q.tbt_max_noreclaim_ms:6.1f}")
print(f"  with reclaim :  p50={q.tbt_p50_ms:6.1f}  "
      f"p99={q.tbt_p99_ms:6.1f}  max={q.tbt_max_ms:6.1f}")
