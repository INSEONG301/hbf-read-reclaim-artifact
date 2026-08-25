"""Poisson-arrival request stream + continuous-batching decode simulator.

The closed-form simulator in `simulator.py` computes long-run aggregates
(reclaim BW, lifetime, …) for ONE fixed (I, O, B) workload. For latency-side
questions (p50 / p90 / p99 TBT, queueing tail) we need a time-stepped run
because:

  * requests arrive at Poisson(λ) — concurrency varies over time
  * step time depends on the *current* active batch's total KV bytes
  * reclaim cycles fire intermittently and stall the HBF for the time
    needed to (read the source block out) + (write the destination block).
    With NAND read BW ≫ NAND write BW (typical ratio ≈ 30:1 for SLC pSLC /
    Z-NAND), the write side dominates the stall:
        stall = burst_bytes × (1/read_bw + 1/write_bw)
              = burst_bytes × (1 + read_bw/write_bw) / read_bw
              = (1 + WRITE_TO_READ_RATIO) × burst_bytes / read_bw
    Every active request sees this stall because all HBFs in the GPU run
    in lock-step.

The user-facing run takes `n_requests` Poisson arrivals (rather than a wall
duration) so the percentile statistics have a fixed sample budget regardless
of arrival rate.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import List

import numpy as np

from .configs import SimulationConfig
from .policies import compute_geometry, _G


def _wpercentile(values, weights, q: float) -> float:
    """Weighted percentile: percentile of the multiset where value[i] occurs
    weight[i] times. Used to get per-output-token TBT percentiles from one
    (value, weight=active_count) sample per forward pass, without expanding."""
    v = np.asarray(values, dtype=float)
    if v.size == 0:
        return 0.0
    w = np.asarray(weights, dtype=float)
    order = np.argsort(v, kind="mergesort")
    v = v[order]
    w = w[order]
    cw = np.cumsum(w)
    total = cw[-1]
    if total <= 0:
        return float(v[-1])
    target = q / 100.0 * total
    idx = int(np.searchsorted(cw, target, side="left"))
    if idx >= v.size:
        idx = v.size - 1
    return float(v[idx])


def _wmean(values, weights) -> float:
    v = np.asarray(values, dtype=float)
    if v.size == 0:
        return 0.0
    w = np.asarray(weights, dtype=float)
    tot = w.sum()
    return float((v * w).sum() / tot) if tot > 0 else 0.0


# Stall multiplier: read_bw / write_bw. Default 30 (read 30× faster than write).
DEFAULT_WRITE_TO_READ_RATIO = 30.0


# ---------------------------------------------------------------------------
# Capacity-based concurrency cap
# ---------------------------------------------------------------------------

def compute_capacity_max_concurrency(cfg: SimulationConfig) -> dict:
    """Max number of in-flight requests that fit in HBF capacity.

    Total HBF storage = capacity × hbfs_per_gpu × num_gpus. This holds:
      (a) model weights, if `weights_storage == 'hbf'` (else weights live in HBM
          and consume 0 HBF bytes), and
      (b) per-request KV cache: (I+O) tokens × kv_bytes_per_token_global,
          uniformly sharded across all HBFs.

    Per-HBF view (equivalent since KV is sharded uniformly):
      kv_per_request_per_hbf = (I + O) × bpt_padded_per_hbf
      weight_per_hbf         = total_weights / total_hbfs  (0 if HBM-resident)
      max_concurrency        = floor((capacity − weight_per_hbf) / kv_per_req)
    """
    hbf = cfg.hbf
    model = cfg.model
    I = cfg.workload.input_tokens
    O = cfg.workload.output_tokens

    total_hbfs = hbf.hbfs_per_gpu * hbf.num_gpus
    total_hbf_capacity = hbf.capacity_bytes * total_hbfs

    weight_bytes_total = model.total_weight_bytes() if hbf.weights_storage == "hbf" else 0
    weight_per_hbf = weight_bytes_total // total_hbfs if total_hbfs > 0 else 0

    geom = compute_geometry(model, hbf, I + O)
    bpt_per_hbf_padded = geom.pages_per_token_per_hbf * hbf.page_size_bytes
    kv_per_request_per_hbf = (I + O) * bpt_per_hbf_padded
    kv_per_request_system = kv_per_request_per_hbf * total_hbfs

    available_per_hbf = max(0, hbf.capacity_bytes - weight_per_hbf)
    max_concurrency = (
        available_per_hbf // kv_per_request_per_hbf
        if kv_per_request_per_hbf > 0 else 0
    )

    return {
        "total_hbf_capacity_bytes": total_hbf_capacity,
        "weight_bytes_on_hbf_total": int(weight_bytes_total),
        "weight_bytes_per_hbf": int(weight_per_hbf),
        "available_per_hbf_for_kv_bytes": int(available_per_hbf),
        "kv_per_request_per_hbf_bytes": int(kv_per_request_per_hbf),
        "kv_per_request_system_bytes": int(kv_per_request_system),
        "max_concurrency_from_capacity": int(max_concurrency),
    }


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class TBTResult:
    request_rate_per_sec: float
    n_requests: int
    n_tokens: int
    input_tokens: int
    output_tokens: int
    max_concurrent: int
    # Capacity-derived concurrency budget (whether or not it was used as the cap)
    capacity_max_concurrency: int
    kv_per_request_per_hbf_bytes: int
    weight_bytes_per_hbf: int
    total_hbf_capacity_bytes: int

    avg_concurrent: float
    peak_concurrent: int

    # TBT percentiles WITH reclaim on (ms)
    tbt_p50_ms: float
    tbt_p90_ms: float
    tbt_p99_ms: float
    tbt_p999_ms: float
    tbt_mean_ms: float
    tbt_max_ms: float

    # TBT percentiles WITHOUT reclaim (ms)
    tbt_p50_noreclaim_ms: float
    tbt_p90_noreclaim_ms: float
    tbt_p99_noreclaim_ms: float
    tbt_max_noreclaim_ms: float

    # Ratios = with / without
    p50_overhead_x: float
    p90_overhead_x: float
    p99_overhead_x: float

    # Reclaim counters
    total_reclaim_cycles: int                     # KV reclaim waves
    total_reclaim_write_bytes_per_hbf: int        # KV reclaim burst bytes moved
    total_weight_reclaim_cycles: int              # weight reclaim waves
    total_weight_reclaim_write_bytes_per_hbf: int # weight reclaim burst bytes moved
    sim_wallclock_s: float           # how long the simulation modeled
    avg_stall_per_cycle_ms: float    # 31 × burst / read_bw, averaged
    # Write BW breakdown per HBF
    reclaim_bw_per_hbf_bytes_per_sec: float       # total reclaim (KV + weight)
    kv_reclaim_bw_per_hbf_bytes_per_sec: float    # KV-cache reclaim contribution
    weight_reclaim_bw_per_hbf_bytes_per_sec: float  # weight reclaim contribution
    kv_write_bw_per_hbf_bytes_per_sec: float      # prefill+decode KV write contribution
    total_write_bw_per_hbf_bytes_per_sec: float   # kv write + kv reclaim + weight reclaim
    # Lifetime views: kv-write-only, reclaim-only (kv+weight reclaim),
    # weight-reclaim-only, and total (kv write + all reclaim)
    hbf_lifetime_seconds_kv_only: float
    hbf_lifetime_years_kv_only: float
    hbf_lifetime_seconds_reclaim_only: float
    hbf_lifetime_years_reclaim_only: float
    hbf_lifetime_seconds_weight_reclaim_only: float
    hbf_lifetime_years_weight_reclaim_only: float
    hbf_lifetime_seconds: float                   # total (kv write + all reclaim)
    hbf_lifetime_years: float
    reclaim_wear_overhead_x: float                # (kv write + all reclaim) / kv write
    # Mean cycle-to-cycle time (how often a reclaim wave hits one HBF)
    reclaim_period_s: float
    reclaim_cycles_per_sec: float
    # p99 reclaim-induced delta (how many ms reclaim adds on top of baseline)
    p99_delta_ms: float

    def to_row(self) -> dict:
        """Flat dict view (lets plotting.plot_heatmap consume TBTResult)."""
        return asdict(self)


# ---------------------------------------------------------------------------
# Core simulation
# ---------------------------------------------------------------------------

@dataclass
class _Request:
    arrival_time: float
    step: int = 0                  # 0..O-1 decode-step index
    reads_acc: int = 0             # hot-block reads accumulated since last reclaim
    tbts: List[float] = field(default_factory=list)


def _simulate_one_ref(
    cfg: SimulationConfig,
    request_rate: float,
    n_requests: int,
    max_concurrent: int,
    seed: int,
    reclaim_on: bool,
    write_to_read_ratio: float,
) -> tuple:
    """Reference discrete per-forward-pass simulation (exact but O(total
    tokens); kept as the correctness oracle for `_simulate_one_fast`). Returns
      (TBT list, per_request_tbts, kv reclaim cycles, avg_concurrent, peak_concurrent,
       sim_time, total_stall_time, total_kv_reclaim_bytes,
       total_weight_reclaim_bytes, weight_reclaim_cycles)."""
    rng = np.random.default_rng(seed)

    I = cfg.workload.input_tokens
    O = cfg.workload.output_tokens
    hbf = cfg.hbf
    geom = compute_geometry(cfg.model, hbf, I + O)
    K = geom.tokens_per_block
    P = hbf.total_planes_per_hbf
    ppt = geom.pages_per_token_per_hbf
    bpt_padded = ppt * hbf.page_size_bytes
    block_size = hbf.block_size_bytes
    threshold = cfg.reclaim.threshold_page_reads
    read_bw = hbf.effective_bandwidth
    eff_hbm_bw_per_gpu = hbf.hbm_bandwidth_bytes_per_sec * hbf.hbfs_per_gpu
    eff_hbf_bw_per_gpu = read_bw * hbf.hbfs_per_gpu

    stall_factor = 1.0 + write_to_read_ratio  # ~31 when write = read/30

    def _weight_time(n_active: int) -> float:
        active_w = cfg.model.active_weight_bytes_per_gpu(hbf.num_gpus, max(1, n_active))
        if cfg.hbf.weights_storage == "hbm":
            return active_w / eff_hbm_bw_per_gpu if eff_hbm_bw_per_gpu > 0 else 0.0
        return active_w / eff_hbf_bw_per_gpu if eff_hbf_bw_per_gpu > 0 else 0.0

    # --- Weight read-reclaim setup ---------------------------------------
    # If weights live on HBF they are the hottest data in the system: every
    # forward pass (one decode step for the whole active batch) re-reads the
    # weight shard on each HBF. Those reads accumulate as read-disturbance on
    # the weight blocks exactly like KV reads do, so weight blocks reclaim on
    # a fast, near-uniform cadence. Because every weight block is read equally,
    # they all cross the threshold together → one reclaim wave rewrites the
    # whole weight shard on that HBF.
    weights_on_hbf = hbf.weights_storage == "hbf"
    total_hbfs = hbf.hbfs_per_gpu * hbf.num_gpus
    weight_stored_per_hbf = (
        cfg.model.total_weight_bytes() / total_hbfs
        if (weights_on_hbf and total_hbfs > 0) else 0.0
    )
    num_weight_blocks = int(weight_stored_per_hbf // block_size)
    weight_burst_bytes = num_weight_blocks * block_size   # whole shard per wave
    weight_reads_acc = 0.0
    total_weight_reclaim_bytes = 0
    n_weight_reclaim_cycles = 0

    # Generate exactly n_requests Poisson arrivals.
    inter = rng.exponential(1.0 / max(request_rate, 1e-12), size=n_requests)
    arrivals = np.cumsum(inter).tolist()
    requests = [_Request(arrival_time=a) for a in arrivals]

    active: List[_Request] = []
    waiting: List[_Request] = []
    pending_idx = 0

    sim_t = 0.0
    n_reclaim_cycles = 0
    total_stall = 0.0
    total_reclaim_bytes = 0
    peak_concurrent = 0
    concurrent_time_weighted = 0.0
    completed = 0
    # TBT is aggregated per output token (per request): a batch forward pass
    # contributes `len(active)` samples, all equal to that pass's step time.
    # We store one value per pass plus a weight (= active count) and compute
    # weighted percentiles later — identical to expanding per token, without
    # the memory blow-up. Per-request trajectories (r.tbts) feed the plots.
    step_tbts: List[float] = []
    step_w: List[int] = []

    while completed < n_requests:
        # Admit arrivals.
        while pending_idx < len(requests) and requests[pending_idx].arrival_time <= sim_t:
            waiting.append(requests[pending_idx])
            pending_idx += 1
        while waiting and len(active) < max_concurrent:
            active.append(waiting.pop(0))

        if not active:
            # Jump to next arrival.
            if pending_idx < len(requests):
                sim_t = requests[pending_idx].arrival_time
                continue
            break

        peak_concurrent = max(peak_concurrent, len(active))

        # Step time = weight load + KV reads of all active requests (HBFs
        # operate in lock-step so the slowest channel sets the pace; here
        # all HBFs have identical pattern, so per-HBF time = global time).
        kv_step_time = sum(
            (I + a.step + 1) * bpt_padded / read_bw for a in active
        )
        step_time = _weight_time(len(active)) + kv_step_time

        # Reclaim. Per-block access count:
        #   At seq_len = I+step+1, the hot block (stripe 0, plane 0) holds
        #     num_stored = min(K, ceil(seq_len / P)) tokens.
        #   Per decode step it reads exactly num_stored × ppt pages.
        #   For partial stripe (num_stored < K) this is much less than the
        #   "fully saturated" K × ppt rate, so the cycle fires later.
        # When the hot block crosses threshold:
        #   * fully-saturated stripes 0..sat-1 all reach threshold at the
        #     same step → fire together = sat × P blocks
        #   * if sat == 0 (whole-system is in partial stripe), only the
        #     hot block itself fires = 1 block
        # burst_bytes therefore scales with current saturation level.
        burst_stall = 0.0
        if reclaim_on:
            for a in active:
                seq_len = I + a.step + 1
                num_stored_hot = min(K, max(1, (seq_len + P - 1) // P))  # ceil(seq_len/P)
                a.reads_acc += num_stored_hot * ppt
                while a.reads_acc >= threshold:
                    a.reads_acc -= threshold
                    sat = max(0, seq_len // (K * P))
                    blocks_cycle = sat * P if sat > 0 else 1
                    burst_bytes = blocks_cycle * block_size
                    # stall = burst × (1/read_bw + 1/write_bw)
                    stall_one = burst_bytes * stall_factor / read_bw
                    burst_stall += stall_one
                    total_stall += stall_one
                    total_reclaim_bytes += burst_bytes
                    n_reclaim_cycles += 1

            # Weight reclaim: charged once per forward pass (not per request),
            # since the weight shard is loaded once and shared across the batch.
            if weights_on_hbf and num_weight_blocks > 0:
                # Bytes of weights physically read this pass on one HBF. For a
                # dense model this equals the whole shard; for MoE it is the
                # activated-expert fraction (grows toward the shard with batch).
                w_read_per_hbf = (
                    cfg.model.active_weight_bytes_per_gpu(hbf.num_gpus, len(active))
                    / hbf.hbfs_per_gpu
                )
                # Reads landing on each (uniformly hot) weight block this step,
                # in page-reads: pages_per_block × (read fraction of the shard).
                weight_reads_acc += hbf.pages_per_block * (
                    w_read_per_hbf / weight_stored_per_hbf
                )
                while weight_reads_acc >= threshold:
                    weight_reads_acc -= threshold
                    stall_one = weight_burst_bytes * stall_factor / read_bw
                    burst_stall += stall_one
                    total_stall += stall_one
                    total_weight_reclaim_bytes += weight_burst_bytes
                    n_weight_reclaim_cycles += 1

        actual_step = step_time + burst_stall
        step_tbts.append(actual_step)   # one value per forward pass
        step_w.append(len(active))      # weight = tokens emitted this pass

        # Emit one token for every active request (per-request trajectory only).
        for a in active:
            a.tbts.append(actual_step)
            a.step += 1

        # Concurrency × time accumulator.
        concurrent_time_weighted += len(active) * actual_step
        sim_t += actual_step

        # Retire completed.
        before = len(active)
        active = [a for a in active if a.step < O]
        completed += (before - len(active))

    avg_conc = concurrent_time_weighted / sim_t if sim_t > 0 else 0.0
    all_tbts = np.asarray(step_tbts, dtype=float)
    all_w = np.asarray(step_w, dtype=float)
    per_req_tbts = [r.tbts for r in requests]
    return (all_tbts, per_req_tbts, n_reclaim_cycles, avg_conc, peak_concurrent,
            sim_t, total_stall, total_reclaim_bytes,
            total_weight_reclaim_bytes, n_weight_reclaim_cycles, all_w)


# ---------------------------------------------------------------------------
# Optimized event-driven simulation
# ---------------------------------------------------------------------------
#
# The reference above executes one Python iteration per forward pass, i.e.
# O(total decode tokens / concurrency) work with a per-active inner loop — too
# slow once we need n_requests ≥ max_concurrent at O = 1M to actually saturate.
#
# Key facts exploited here (all exact, not approximations):
#   * While the active set is fixed, the baseline step time is AFFINE in the
#     sub-step index j:  base(j) = weight_time(n) + (bpt/read_bw)·(ΣΣ + n·j).
#     (KV read time sums (I+step+1) over active, which is linear in j.)
#   * Each request's read-disturbance accrual depends only on its OWN decode
#     progress, so its reclaim-threshold crossings can be found in closed form
#     (via the same _G(n,P)=Σ ceil(u/P) prefix used by the analytical policy),
#     rather than by stepping token-by-token.
#   * Reclaim writes are counted per request over its full O-token decode, so
#     total reclaim bytes / cycles are INDEPENDENT of arrival timing and come
#     out bit-identical to the reference.
#
# We therefore advance the sim in segments bounded by the next state change
# (a retirement or an admission); within a segment we emit the affine baseline
# in bulk and splice in the (sparse) reclaim-stall spikes at their exact
# crossing passes. Work is O(total forward passes + total reclaim events),
# a factor ≈ concurrency faster than the reference.
#
# Admission timing uses the baseline (reclaim-free) cumulative time to decide
# the pass at which the next arrival is admitted. Reclaim stalls that precede
# an admission would let the clock reach the arrival a hair sooner; ignoring
# that shifts an admission by at most a few passes and leaves the wear metrics
# exact — validated against the reference to match percentiles within ~1%.

def _simulate_one_fast(
    cfg: SimulationConfig,
    request_rate: float,
    n_requests: int,
    max_concurrent: int,
    seed: int,
    reclaim_on: bool,
    write_to_read_ratio: float,
) -> tuple:
    """Event-driven equivalent of `_simulate_one_ref`. Returns the same tuple,
    except `per_request_tbts` is [] (trajectories are only needed by the
    trajectory entry point, which uses the reference with small n_requests)."""
    rng = np.random.default_rng(seed)

    I = cfg.workload.input_tokens
    O = cfg.workload.output_tokens
    hbf = cfg.hbf
    geom = compute_geometry(cfg.model, hbf, I + O)
    K = geom.tokens_per_block
    P = hbf.total_planes_per_hbf
    ppt = geom.pages_per_token_per_hbf
    block_size = hbf.block_size_bytes
    threshold = cfg.reclaim.threshold_page_reads
    read_bw = hbf.effective_bandwidth
    eff_hbm_bw_per_gpu = hbf.hbm_bandwidth_bytes_per_sec * hbf.hbfs_per_gpu
    eff_hbf_bw_per_gpu = read_bw * hbf.hbfs_per_gpu
    stall_factor = 1.0 + write_to_read_ratio
    bpt_over_bw = (ppt * hbf.page_size_bytes) / read_bw if read_bw > 0 else 0.0
    KP = K * P

    def _weight_time(n_active: int) -> float:
        active_w = cfg.model.active_weight_bytes_per_gpu(hbf.num_gpus, max(1, n_active))
        if hbf.weights_storage == "hbm":
            return active_w / eff_hbm_bw_per_gpu if eff_hbm_bw_per_gpu > 0 else 0.0
        return active_w / eff_hbf_bw_per_gpu if eff_hbf_bw_per_gpu > 0 else 0.0

    weights_on_hbf = hbf.weights_storage == "hbf"
    total_hbfs = hbf.hbfs_per_gpu * hbf.num_gpus
    weight_stored_per_hbf = (
        cfg.model.total_weight_bytes() / total_hbfs
        if (weights_on_hbf and total_hbfs > 0) else 0.0
    )
    num_weight_blocks = int(weight_stored_per_hbf // block_size)
    weight_burst_bytes = num_weight_blocks * block_size
    weight_stall_one = weight_burst_bytes * stall_factor / read_bw

    # step s at/after which num_stored saturates to K (ceil((I+s+1)/P) >= K)
    s_plat = (K - 1) * P - I

    def cum_num_stored(s0: int, d: int) -> int:
        """Σ_{s=s0}^{s0+d-1} min(K, ceil((I+s+1)/P))."""
        if d <= 0:
            return 0
        end = s0 + d - 1
        total = 0
        ramp_end = min(end, s_plat - 1)
        if ramp_end >= s0:
            total += _G(I + ramp_end + 1, P) - _G(I + s0, P)
        ps = s0 if s0 > s_plat else s_plat
        if end >= ps:
            total += (end - ps + 1) * K
        return total

    def cross_pass(s0: int, acc0: float, target: float, d_hi: int) -> int:
        """Smallest d in [1, d_hi] with acc0 + ppt·cum_num_stored(s0,d) >= target."""
        lo, hi = 1, d_hi
        while lo < hi:
            mid = (lo + hi) // 2
            if acc0 + ppt * cum_num_stored(s0, mid) >= target:
                hi = mid
            else:
                lo = mid + 1
        return lo

    inter = rng.exponential(1.0 / max(request_rate, 1e-12), size=n_requests)
    arrivals = np.cumsum(inter)
    n_all = n_requests

    active_steps: List[int] = []
    active_acc: List[float] = []
    weight_reads_acc = 0.0
    pending_idx = 0
    waiting = 0
    completed = 0

    sim_t = 0.0
    n_reclaim_cycles = 0
    total_stall = 0.0
    total_reclaim_bytes = 0
    total_weight_reclaim_bytes = 0
    n_weight_reclaim_cycles = 0
    peak_concurrent = 0
    concurrent_time_weighted = 0.0
    tbt_chunks: List[np.ndarray] = []
    w_chunks: List[np.ndarray] = []

    while completed < n_all:
        while pending_idx < n_all and arrivals[pending_idx] <= sim_t:
            waiting += 1
            pending_idx += 1
        while waiting > 0 and len(active_steps) < max_concurrent:
            active_steps.append(0)
            active_acc.append(0.0)
            waiting -= 1

        n_act = len(active_steps)
        if n_act == 0:
            if pending_idx < n_all:
                sim_t = float(arrivals[pending_idx])
                continue
            break

        if n_act > peak_concurrent:
            peak_concurrent = n_act

        max_step = max(active_steps)
        delta_retire = O - max_step            # passes until leader retires
        A1 = bpt_over_bw * n_act
        S = sum((I + s + 1) for s in active_steps)
        A0 = _weight_time(n_act) + bpt_over_bw * S

        # Determine segment length: min(retire, next-admission).
        actual_L = delta_retire
        admitting = False
        if len(active_steps) < max_concurrent and pending_idx < n_all:
            target_t = float(arrivals[pending_idx]) - sim_t
            # smallest k>=1 with A0·k + A1·k(k-1)/2 >= target_t  (baseline clock)
            if target_t <= A0:
                k_real = 1
            elif A1 > 0:
                disc = (A0 - A1 / 2.0) ** 2 + 2.0 * A1 * target_t
                k_real = int(math.ceil(
                    (-(A0 - A1 / 2.0) + math.sqrt(max(0.0, disc))) / A1
                ))
                # correct rounding
                while k_real > 1 and A0 * (k_real - 1) + A1 * (k_real - 1) * (k_real - 2) / 2.0 >= target_t:
                    k_real -= 1
                while A0 * k_real + A1 * k_real * (k_real - 1) / 2.0 < target_t:
                    k_real += 1
            else:
                k_real = max(1, int(math.ceil(target_t / A0))) if A0 > 0 else delta_retire
            if k_real < delta_retire:
                actual_L = k_real
                admitting = True
        if actual_L < 1:
            actual_L = 1

        # Baseline TBT for the segment (affine).
        j = np.arange(actual_L, dtype=float)
        seg = A0 + A1 * j

        # Reclaim spikes spliced onto the exact crossing passes.
        if reclaim_on:
            for idx in range(n_act):
                s0 = active_steps[idx]
                acc0 = active_acc[idx]
                Ctot = acc0 + ppt * cum_num_stored(s0, actual_L)
                kmax = int(Ctot // threshold)
                for k in range(1, kmax + 1):
                    d = cross_pass(s0, acc0, k * threshold, actual_L)
                    p = d - 1
                    seq_len = I + s0 + p + 1
                    sat = seq_len // KP
                    blocks = sat * P if sat > 0 else 1
                    bb = blocks * block_size
                    st = bb * stall_factor / read_bw
                    seg[p] += st
                    total_stall += st
                    total_reclaim_bytes += bb
                    n_reclaim_cycles += 1
                active_acc[idx] = Ctot - kmax * threshold
                active_steps[idx] = s0 + actual_L

            if weights_on_hbf and num_weight_blocks > 0:
                w_read = (
                    cfg.model.active_weight_bytes_per_gpu(hbf.num_gpus, n_act)
                    / hbf.hbfs_per_gpu
                )
                per_pass_w = hbf.pages_per_block * (w_read / weight_stored_per_hbf)
                if per_pass_w > 0:
                    Cw = weight_reads_acc + actual_L * per_pass_w
                    kwmax = int(Cw // threshold)
                    for k in range(1, kwmax + 1):
                        m = int(math.ceil((k * threshold - weight_reads_acc) / per_pass_w)) - 1
                        if m < 0:
                            m = 0
                        elif m >= actual_L:
                            m = actual_L - 1
                        seg[m] += weight_stall_one
                        total_stall += weight_stall_one
                        total_weight_reclaim_bytes += weight_burst_bytes
                        n_weight_reclaim_cycles += 1
                    weight_reads_acc = Cw - kwmax * threshold
        else:
            for idx in range(n_act):
                active_steps[idx] += actual_L

        seg_sum = float(seg.sum())
        sim_t += seg_sum
        concurrent_time_weighted += n_act * seg_sum
        tbt_chunks.append(seg)
        w_chunks.append(np.full(actual_L, n_act, dtype=float))

        if not admitting:
            # leader(s) reached step O — retire them
            keep_steps: List[int] = []
            keep_acc: List[float] = []
            retired = 0
            for idx in range(len(active_steps)):
                if active_steps[idx] < O:
                    keep_steps.append(active_steps[idx])
                    keep_acc.append(active_acc[idx])
                else:
                    retired += 1
            active_steps = keep_steps
            active_acc = keep_acc
            completed += retired

    all_tbts = np.concatenate(tbt_chunks) if tbt_chunks else np.array([])
    all_w = np.concatenate(w_chunks) if w_chunks else np.array([])
    avg_conc = concurrent_time_weighted / sim_t if sim_t > 0 else 0.0
    return (all_tbts, [], n_reclaim_cycles, avg_conc, peak_concurrent,
            sim_t, total_stall, total_reclaim_bytes,
            total_weight_reclaim_bytes, n_weight_reclaim_cycles, all_w)


# Dispatcher — fast path by default; reference kept for trajectories / validation.
_USE_FAST_SIM = True


def _simulate_one(*args, **kwargs):
    if _USE_FAST_SIM:
        return _simulate_one_fast(*args, **kwargs)
    return _simulate_one_ref(*args, **kwargs)


# ---------------------------------------------------------------------------
# Public entry
# ---------------------------------------------------------------------------

def simulate_request_stream(
    cfg: SimulationConfig,
    request_rate_per_sec: float,
    n_requests: int = 1000,
    max_concurrent: int = 32,
    seed: int = 0,
    write_to_read_ratio: float = DEFAULT_WRITE_TO_READ_RATIO,
) -> TBTResult:
    """Two passes (reclaim ON, reclaim OFF) with identical arrival seed so
    that percentile differences isolate the reclaim contribution."""
    cap_info = compute_capacity_max_concurrency(cfg)
    (tbts_on, _per_req_on, cycles, avg_conc_on, peak_on,
     sim_t, total_stall, reclaim_bytes,
     weight_reclaim_bytes, weight_cycles, w_on) = _simulate_one(
        cfg, request_rate_per_sec, n_requests, max_concurrent, seed,
        reclaim_on=True, write_to_read_ratio=write_to_read_ratio,
    )
    (tbts_off, _per_req_off, _c2, _a2, _p2, _s2, _st2, _rb2,
     _wb2, _wc2, w_off) = _simulate_one(
        cfg, request_rate_per_sec, n_requests, max_concurrent, seed,
        reclaim_on=False, write_to_read_ratio=write_to_read_ratio,
    )

    # Per-output-token (concurrency-weighted) percentiles: each forward pass
    # counts once per active request, so a stall during high concurrency hurts
    # proportionally more users. See _wpercentile.
    def pct(arr, w, q):
        return _wpercentile(arr, w, q)

    p50    = pct(tbts_on, w_on, 50)
    p90    = pct(tbts_on, w_on, 90)
    p99    = pct(tbts_on, w_on, 99)
    p999   = pct(tbts_on, w_on, 99.9)
    p_mean = _wmean(tbts_on, w_on)
    p_max  = float(np.max(tbts_on)) if len(tbts_on) else 0.0

    p50_off = pct(tbts_off, w_off, 50)
    p90_off = pct(tbts_off, w_off, 90)
    p99_off = pct(tbts_off, w_off, 99)
    p_max_off = float(np.max(tbts_off)) if len(tbts_off) else 0.0

    def safe_ratio(a, b):
        return a / b if b > 0 else float("inf")

    # Write BW per HBF. Three sources contribute to PE wear:
    #   (a) KV cache writes — each completed request writes (I+O) tokens of KV,
    #       at bpt_padded bytes/token, to its assigned HBFs. With a balanced
    #       placement across all HBFs, one HBF's share is (I+O) × bpt_padded
    #       per request. n_requests requests over sim_t → kv_bw bytes/sec/HBF.
    #   (b) KV reclaim writes — accumulated in `reclaim_bytes`.
    #   (c) Weight reclaim writes — accumulated in `weight_reclaim_bytes` (only
    #       nonzero when weights_storage == "hbf"; weights are re-read every
    #       forward pass, so their blocks reclaim fastest of all).
    # Lifetime = PE × capacity / (kv_write + kv_reclaim + weight_reclaim).
    cfg_hbf = cfg.hbf
    I_outer = cfg.workload.input_tokens
    O_outer = cfg.workload.output_tokens
    geom_local = compute_geometry(cfg.model, cfg_hbf, I_outer + O_outer)
    bpt_padded_local = geom_local.pages_per_token_per_hbf * cfg_hbf.page_size_bytes
    kv_write_bytes_per_hbf = n_requests * (I_outer + O_outer) * bpt_padded_local
    kv_write_bw = (kv_write_bytes_per_hbf / sim_t) if sim_t > 0 else 0.0
    kv_reclaim_bw = (reclaim_bytes / sim_t) if sim_t > 0 else 0.0
    weight_reclaim_bw = (weight_reclaim_bytes / sim_t) if sim_t > 0 else 0.0
    reclaim_bw = kv_reclaim_bw + weight_reclaim_bw        # total reclaim
    total_write_bw = kv_write_bw + reclaim_bw
    write_budget = cfg.reclaim.pe_cycle_limit * cfg_hbf.capacity_bytes
    SECONDS_PER_YEAR = 365.25 * 86400

    def _life(bw: float) -> tuple[float, float]:
        s = write_budget / bw if bw > 0 else float("inf")
        y = s / SECONDS_PER_YEAR if s != float("inf") else float("inf")
        return s, y

    lifetime_s_kv,      lifetime_y_kv      = _life(kv_write_bw)
    lifetime_s_reclaim, lifetime_y_reclaim = _life(reclaim_bw)
    lifetime_s_wrec,    lifetime_y_wrec    = _life(weight_reclaim_bw)
    lifetime_s,         lifetime_y         = _life(total_write_bw)
    wear_overhead_x = (total_write_bw / kv_write_bw) if kv_write_bw > 0 else 1.0

    # Reclaim cadence (KV waves)
    reclaim_period = (sim_t / cycles) if cycles > 0 else float("inf")
    reclaim_per_sec = cycles / sim_t if sim_t > 0 else 0.0

    return TBTResult(
        request_rate_per_sec=request_rate_per_sec,
        n_requests=n_requests,
        n_tokens=int(np.sum(w_on)) if len(w_on) else 0,
        input_tokens=cfg.workload.input_tokens,
        output_tokens=cfg.workload.output_tokens,
        max_concurrent=max_concurrent,
        capacity_max_concurrency=cap_info["max_concurrency_from_capacity"],
        kv_per_request_per_hbf_bytes=cap_info["kv_per_request_per_hbf_bytes"],
        weight_bytes_per_hbf=cap_info["weight_bytes_per_hbf"],
        total_hbf_capacity_bytes=cap_info["total_hbf_capacity_bytes"],
        avg_concurrent=avg_conc_on,
        peak_concurrent=peak_on,
        tbt_p50_ms=p50 * 1000,
        tbt_p90_ms=p90 * 1000,
        tbt_p99_ms=p99 * 1000,
        tbt_p999_ms=p999 * 1000,
        tbt_mean_ms=p_mean * 1000,
        tbt_max_ms=p_max * 1000,
        tbt_p50_noreclaim_ms=p50_off * 1000,
        tbt_p90_noreclaim_ms=p90_off * 1000,
        tbt_p99_noreclaim_ms=p99_off * 1000,
        tbt_max_noreclaim_ms=p_max_off * 1000,
        p50_overhead_x=safe_ratio(p50, p50_off),
        p90_overhead_x=safe_ratio(p90, p90_off),
        p99_overhead_x=safe_ratio(p99, p99_off),
        total_reclaim_cycles=cycles,
        total_reclaim_write_bytes_per_hbf=int(reclaim_bytes),
        total_weight_reclaim_cycles=weight_cycles,
        total_weight_reclaim_write_bytes_per_hbf=int(weight_reclaim_bytes),
        sim_wallclock_s=sim_t,
        avg_stall_per_cycle_ms=(total_stall / cycles * 1000) if cycles > 0 else 0.0,
        reclaim_bw_per_hbf_bytes_per_sec=reclaim_bw,
        kv_reclaim_bw_per_hbf_bytes_per_sec=kv_reclaim_bw,
        weight_reclaim_bw_per_hbf_bytes_per_sec=weight_reclaim_bw,
        kv_write_bw_per_hbf_bytes_per_sec=kv_write_bw,
        total_write_bw_per_hbf_bytes_per_sec=total_write_bw,
        hbf_lifetime_seconds_kv_only=lifetime_s_kv,
        hbf_lifetime_years_kv_only=lifetime_y_kv,
        hbf_lifetime_seconds_reclaim_only=lifetime_s_reclaim,
        hbf_lifetime_years_reclaim_only=lifetime_y_reclaim,
        hbf_lifetime_seconds_weight_reclaim_only=lifetime_s_wrec,
        hbf_lifetime_years_weight_reclaim_only=lifetime_y_wrec,
        hbf_lifetime_seconds=lifetime_s,
        hbf_lifetime_years=lifetime_y,
        reclaim_wear_overhead_x=wear_overhead_x,
        reclaim_period_s=reclaim_period,
        reclaim_cycles_per_sec=reclaim_per_sec,
        p99_delta_ms=p99 * 1000 - p99_off * 1000,
    )


def simulate_request_stream_with_trajectory(
    cfg: SimulationConfig,
    request_rate_per_sec: float,
    n_requests: int = 100,
    max_concurrent: int = 32,
    seed: int = 0,
    write_to_read_ratio: float = DEFAULT_WRITE_TO_READ_RATIO,
) -> tuple[TBTResult, list[list[float]], list[list[float]]]:
    """Run the simulator once and expose per-request TBT trajectories.

    Returns (result, on_trajectories, off_trajectories) where each trajectory
    is a `n_requests`-long list, each entry a list of length O (one TBT per
    decode step for that request). Use for trajectory plots where the
    step-by-step TBT structure (not just percentiles) is informative."""
    cap_info = compute_capacity_max_concurrency(cfg)
    # Trajectory view needs per-request TBT sequences → use the reference
    # simulator (called with small n_requests here, so cost is negligible).
    (tbts_on, on_traj, cycles, avg_conc_on, peak_on,
     sim_t, total_stall, reclaim_bytes,
     weight_reclaim_bytes, weight_cycles, w_on) = _simulate_one_ref(
        cfg, request_rate_per_sec, n_requests, max_concurrent, seed,
        reclaim_on=True, write_to_read_ratio=write_to_read_ratio,
    )
    (tbts_off, off_traj, _c2, _a2, _p2, _s2, _st2, _rb2,
     _wb2, _wc2, w_off) = _simulate_one_ref(
        cfg, request_rate_per_sec, n_requests, max_concurrent, seed,
        reclaim_on=False, write_to_read_ratio=write_to_read_ratio,
    )

    def pct(arr, w, q):
        return _wpercentile(arr, w, q)

    p50, p90, p99, p999 = (pct(tbts_on, w_on, q) for q in (50, 90, 99, 99.9))
    p_mean = _wmean(tbts_on, w_on)
    p_max = float(np.max(tbts_on)) if len(tbts_on) else 0.0
    p50_off, p90_off, p99_off = (pct(tbts_off, w_off, q) for q in (50, 90, 99))
    p_max_off = float(np.max(tbts_off)) if len(tbts_off) else 0.0

    def safe_ratio(a, b):
        return a / b if b > 0 else float("inf")

    I_outer = cfg.workload.input_tokens
    O_outer = cfg.workload.output_tokens
    geom_local = compute_geometry(cfg.model, cfg.hbf, I_outer + O_outer)
    bpt_padded_local = geom_local.pages_per_token_per_hbf * cfg.hbf.page_size_bytes
    kv_write_bytes_per_hbf = n_requests * (I_outer + O_outer) * bpt_padded_local
    kv_write_bw = (kv_write_bytes_per_hbf / sim_t) if sim_t > 0 else 0.0
    kv_reclaim_bw = (reclaim_bytes / sim_t) if sim_t > 0 else 0.0
    weight_reclaim_bw = (weight_reclaim_bytes / sim_t) if sim_t > 0 else 0.0
    reclaim_bw = kv_reclaim_bw + weight_reclaim_bw
    total_write_bw = kv_write_bw + reclaim_bw
    write_budget = cfg.reclaim.pe_cycle_limit * cfg.hbf.capacity_bytes
    SECONDS_PER_YEAR = 365.25 * 86400

    def _life(bw):
        s = write_budget / bw if bw > 0 else float("inf")
        y = s / SECONDS_PER_YEAR if s != float("inf") else float("inf")
        return s, y

    ls_kv, ly_kv = _life(kv_write_bw)
    ls_rec, ly_rec = _life(reclaim_bw)
    ls_wrec, ly_wrec = _life(weight_reclaim_bw)
    ls, ly = _life(total_write_bw)
    wear_x = (total_write_bw / kv_write_bw) if kv_write_bw > 0 else 1.0
    reclaim_period = (sim_t / cycles) if cycles > 0 else float("inf")
    reclaim_per_sec = cycles / sim_t if sim_t > 0 else 0.0

    result = TBTResult(
        request_rate_per_sec=request_rate_per_sec,
        n_requests=n_requests,
        n_tokens=int(np.sum(w_on)) if len(w_on) else 0,
        input_tokens=I_outer,
        output_tokens=O_outer,
        max_concurrent=max_concurrent,
        capacity_max_concurrency=cap_info["max_concurrency_from_capacity"],
        kv_per_request_per_hbf_bytes=cap_info["kv_per_request_per_hbf_bytes"],
        weight_bytes_per_hbf=cap_info["weight_bytes_per_hbf"],
        total_hbf_capacity_bytes=cap_info["total_hbf_capacity_bytes"],
        avg_concurrent=avg_conc_on,
        peak_concurrent=peak_on,
        tbt_p50_ms=p50 * 1000, tbt_p90_ms=p90 * 1000, tbt_p99_ms=p99 * 1000,
        tbt_p999_ms=p999 * 1000, tbt_mean_ms=p_mean * 1000, tbt_max_ms=p_max * 1000,
        tbt_p50_noreclaim_ms=p50_off * 1000, tbt_p90_noreclaim_ms=p90_off * 1000,
        tbt_p99_noreclaim_ms=p99_off * 1000, tbt_max_noreclaim_ms=p_max_off * 1000,
        p50_overhead_x=safe_ratio(p50, p50_off),
        p90_overhead_x=safe_ratio(p90, p90_off),
        p99_overhead_x=safe_ratio(p99, p99_off),
        total_reclaim_cycles=cycles,
        total_reclaim_write_bytes_per_hbf=int(reclaim_bytes),
        total_weight_reclaim_cycles=weight_cycles,
        total_weight_reclaim_write_bytes_per_hbf=int(weight_reclaim_bytes),
        sim_wallclock_s=sim_t,
        avg_stall_per_cycle_ms=(total_stall / cycles * 1000) if cycles > 0 else 0.0,
        reclaim_bw_per_hbf_bytes_per_sec=reclaim_bw,
        kv_reclaim_bw_per_hbf_bytes_per_sec=kv_reclaim_bw,
        weight_reclaim_bw_per_hbf_bytes_per_sec=weight_reclaim_bw,
        kv_write_bw_per_hbf_bytes_per_sec=kv_write_bw,
        total_write_bw_per_hbf_bytes_per_sec=total_write_bw,
        hbf_lifetime_seconds_kv_only=ls_kv, hbf_lifetime_years_kv_only=ly_kv,
        hbf_lifetime_seconds_reclaim_only=ls_rec, hbf_lifetime_years_reclaim_only=ly_rec,
        hbf_lifetime_seconds_weight_reclaim_only=ls_wrec,
        hbf_lifetime_years_weight_reclaim_only=ly_wrec,
        hbf_lifetime_seconds=ls, hbf_lifetime_years=ly,
        reclaim_wear_overhead_x=wear_x,
        reclaim_period_s=reclaim_period,
        reclaim_cycles_per_sec=reclaim_per_sec,
        p99_delta_ms=p99 * 1000 - p99_off * 1000,
    )
    return result, on_traj, off_traj
