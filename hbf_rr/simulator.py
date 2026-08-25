"""Core simulator: takes a SimulationConfig, returns a SimulationResult.

Single-HBF, single-sequence analysis (batch dimensions are accounted for by
scaling, since each batch element has its own KV cache block table).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import Optional, List

from .configs import HBFConfig, ModelConfig, WorkloadConfig, ReclaimConfig, SimulationConfig
from .policies import POLICIES


@dataclass
class BlockReclaimStats:
    stripe: int
    plane: int
    num_tokens_stored: int
    total_page_reads: int
    reclaim_events: int
    first_reclaim_step: Optional[int]


@dataclass
class SimulationResult:
    # ----- echo of inputs (for sweep tables) -----
    model_name: str
    write_policy: str
    input_tokens: int
    output_tokens: int
    batch_size: int

    page_size_bytes: int
    pages_per_block: int
    planes_per_die: int
    dies_per_stack: int
    stacks_per_hbf: int
    total_dies_per_hbf: int
    total_planes_per_hbf: int
    hbfs_per_gpu: int
    num_gpus: int
    hbf_capacity_bytes: int
    hbf_bandwidth_bytes_per_sec: float
    bandwidth_utilization: float
    nand_tR_ns: float
    hbm_bandwidth_bytes_per_sec: float
    weights_storage: str
    reclaim_threshold: int

    # ----- KV cache geometry -----
    kv_bytes_per_token_global: int
    kv_bytes_per_token_per_gpu: float
    kv_bytes_per_token_per_hbf: float
    pages_per_token_per_hbf: int
    tokens_per_block: int
    blocks_per_sequence_per_hbf: int
    seq_kv_bytes_per_hbf: float
    total_kv_bytes_per_hbf: float
    capacity_fit: bool
    blocks_per_plane_per_hbf: int        # capacity / (planes × block_size)
    tokens_to_fill_first_stripe: int     # seq_len at which stripe 0 (block 0 of every plane) is full

    # ----- plane bandwidth -----
    per_plane_bandwidth_bytes_per_sec: float
    planes_needed_for_target_bw: int
    plane_oversubscription: float

    # ----- read-reclaim outcomes -----
    hottest_block_stripe: int
    hottest_block_plane: int
    hottest_block_total_reads: int
    hottest_block_reclaim_events: int
    first_reclaim_step: Optional[int]
    first_reclaim_token: Optional[int]
    first_reclaim_time_s: Optional[float]
    total_reclaim_events_per_seq: int
    total_reclaim_events_per_gpu: int
    total_reclaim_events_system: int
    avg_reads_per_block: float
    blocks_touched_by_reads: int

    # ----- decode latency -----
    active_weight_bytes_per_gpu: float
    weight_load_time_per_step_s: float
    avg_kv_step_time_s: float
    avg_step_time_s: float
    total_decode_time_s: float
    avg_token_latency_ms: float
    avg_kv_read_bandwidth_per_hbf_bytes_per_sec: float
    bottleneck: str

    # ----- read reclaim throughput -----
    reclaim_bytes_per_event: int
    # Cadence on the hot block: number of decode steps that elapse between
    # successive reclaim events, and the wall-clock time that corresponds to.
    decode_steps_between_reclaims_hottest: float
    hottest_block_reclaim_period_s: float
    # "Per-cycle burst" — under PBBF every fully-saturated stripe's P blocks
    # cross threshold within ~P/pages_per_block decode steps of each other, so
    # they reclaim together. The total bytes moved in that wave is the
    # per-cycle reclaim burst size.
    saturated_stripes_at_end: int
    blocks_reclaimed_per_cycle_per_hbf: int   # × batch_size
    reclaim_size_per_cycle_per_hbf_bytes: int
    reclaim_size_per_cycle_per_gpu_bytes: int
    reclaim_size_per_cycle_system_bytes: int
    reclaim_cycles_during_decode: float       # decode_time / period
    # Overhead the reclaim traffic imposes on the HBF that is also serving KV reads.
    reclaim_overhead_fraction_per_hbf: float  # reclaim_BW / HBF_BW  (1.0 means 100 %)
    burst_duration_per_cycle_s: float         # time to move one cycle's burst on one HBF
    # P/E budget consumed by reclaim. Each reclaim writes one block, so total
    # writes per second per HBF = reclaim_BW (bytes/sec). Wearing out the HBF
    # = consuming the per-block P/E budget on every block. Lifetime is then
    # PE_limit × capacity / reclaim_BW.
    pe_cycle_limit: int
    hbf_total_write_budget_bytes: int         # = pe_cycle_limit × capacity_bytes
    # Three lifetime estimates, in increasing realism:
    #   _kv_only      — counts only the KV-cache writes (prefill + decode)
    #   _reclaim_only — counts only reclaim writes (the previous default)
    #   _total        — counts KV writes + reclaim writes
    # reclaim_wear_overhead_x is (kv + reclaim) / kv, i.e. "lifetime gets X
    # times shorter when reclaim is on". 1.0 means reclaim is negligible.
    kv_write_bytes_per_hbf: int               # I+O prefill+decode KV bytes, batch summed
    kv_write_bw_per_hbf_bytes_per_sec: float
    # Weight read-reclaim (nonzero only when weights_storage == "hbf"). Weights
    # are re-read every forward pass, so their blocks are the hottest of all.
    weight_reclaim_events_per_hbf: float
    weight_reclaim_write_bw_per_hbf_bytes_per_sec: float
    hbf_lifetime_seconds_kv_only: float
    hbf_lifetime_years_kv_only: float
    hbf_lifetime_seconds_reclaim_only: float  # KV reclaim + weight reclaim
    hbf_lifetime_years_reclaim_only: float
    hbf_lifetime_seconds_weight_reclaim_only: float
    hbf_lifetime_years_weight_reclaim_only: float
    hbf_lifetime_seconds: float               # total: KV write + all reclaim
    hbf_lifetime_years: float                 # total
    reclaim_wear_overhead_x: float            # (kv write + all reclaim) / kv write  (≥ 1)
    # Reclaim BW is computed using *fractional* events: total page reads /
    # threshold (rather than the integer count of completed reclaims). This
    # keeps the metric continuous and non-zero even when no block has crossed
    # the threshold within the decode window, while still matching the
    # integer-event total in the asymptote.
    expected_reclaim_events_per_hbf: float
    reclaim_write_bw_per_hbf_bytes_per_sec: float       # integer-event based
    reclaim_write_bw_per_gpu_bytes_per_sec: float
    reclaim_write_bw_system_bytes_per_sec: float
    # Fractional / asymptotic estimate for short-workload comparison.
    expected_reclaim_write_bw_per_hbf_bytes_per_sec: float

    # ----- diagnostic block-level dump -----
    block_stats: List[BlockReclaimStats] = field(default_factory=list)

    def to_row(self) -> dict:
        d = asdict(self)
        d.pop("block_stats", None)
        return d


def simulate(cfg: SimulationConfig) -> SimulationResult:
    model, hbf, workload, reclaim = cfg.model, cfg.hbf, cfg.workload, cfg.reclaim

    if cfg.write_policy not in POLICIES:
        raise ValueError(
            f"Unknown write policy '{cfg.write_policy}'. Known: {sorted(POLICIES)}"
        )
    policy_fn = POLICIES[cfg.write_policy]
    geom, summaries = policy_fn(model, hbf, workload, reclaim.threshold_page_reads)

    # --- aggregate over blocks for the single-sequence case ---
    total_reclaim_seq = 0
    first_reclaim_global: Optional[int] = None
    hottest_total = -1
    hottest_idx = 0
    blocks_touched = 0
    block_stats: List[BlockReclaimStats] = []

    for i, sm in enumerate(summaries):
        block_stats.append(BlockReclaimStats(
            stripe=sm.stripe, plane=sm.plane,
            num_tokens_stored=sm.num_tokens_stored,
            total_page_reads=sm.total_page_reads,
            reclaim_events=sm.reclaim_events,
            first_reclaim_step=sm.first_reclaim_step,
        ))
        if sm.total_page_reads > 0:
            blocks_touched += 1
        total_reclaim_seq += sm.reclaim_events
        if sm.first_reclaim_step is not None and (
            first_reclaim_global is None or sm.first_reclaim_step < first_reclaim_global
        ):
            first_reclaim_global = sm.first_reclaim_step
        if sm.total_page_reads > hottest_total:
            hottest_total = sm.total_page_reads
            hottest_idx = i

    hottest = block_stats[hottest_idx] if block_stats else None

    # Hottest block's *per-step* read rate after it saturates: equal to
    # min(num_stored, K) * pages_per_token. For the hottest block in PBBF this
    # equals pages_per_block once seq_len exceeds (num_stored-1)*P + B.
    K_full = geom.tokens_per_block
    hottest_per_step_reads = (
        min(hottest.num_tokens_stored, K_full) * geom.pages_per_token_per_hbf
        if hottest else 0
    )

    # --- decode latency ---
    # GPU memory model: there are `hbfs_per_gpu` HBM stacks AND `hbfs_per_gpu`
    # HBF stacks per GPU, each delivering its respective per-stack bandwidth.
    # KV cache reads always go through HBF. Weights go through whichever store
    # is selected by `weights_storage`. Within a decode step the two phases
    # (weight load, then KV read for attention) execute sequentially on the
    # GPU even though they may be served from different memory channels — we
    # always sum their latencies.
    I, O = workload.input_tokens, workload.output_tokens
    bpt_hbf_padded = geom.pages_per_token_per_hbf * hbf.page_size_bytes
    eff_hbf_bw_per_gpu = hbf.effective_bandwidth * hbf.hbfs_per_gpu          # 8 × HBF
    eff_hbm_bw_per_gpu = hbf.hbm_bandwidth_bytes_per_sec * hbf.hbfs_per_gpu  # 8 × HBM
    weight_bytes_per_gpu = model.active_weight_bytes_per_gpu(
        hbf.num_gpus, workload.batch_size
    )

    avg_seq_len = I + (O + 1) / 2 if O > 0 else I
    avg_kv_bytes_per_gpu = workload.batch_size * avg_seq_len * bpt_hbf_padded * hbf.hbfs_per_gpu

    if hbf.weights_storage == "hbm":
        weight_time = weight_bytes_per_gpu / eff_hbm_bw_per_gpu if eff_hbm_bw_per_gpu > 0 else 0.0
    elif hbf.weights_storage == "hbf":
        weight_time = weight_bytes_per_gpu / eff_hbf_bw_per_gpu if eff_hbf_bw_per_gpu > 0 else 0.0
    else:
        raise ValueError(f"Unknown weights_storage '{hbf.weights_storage}'")

    avg_kv_step_time = avg_kv_bytes_per_gpu / eff_hbf_bw_per_gpu if eff_hbf_bw_per_gpu > 0 else 0.0
    avg_step_time = weight_time + avg_kv_step_time
    total_decode_time = avg_step_time * O
    bottleneck = "weights" if weight_time >= avg_kv_step_time else "kv"

    # Per-HBF average BW used by KV reads
    total_kv_bytes_per_hbf_decode = workload.batch_size * sum(I + t for t in range(1, O + 1)) * bpt_hbf_padded if O > 0 else 0
    avg_bw_per_hbf = (
        total_kv_bytes_per_hbf_decode / total_decode_time if total_decode_time > 0 else 0.0
    )

    # --- first reclaim time ---
    first_reclaim_token = None
    first_reclaim_time = None
    if first_reclaim_global is not None:
        first_reclaim_token = I + first_reclaim_global
        # cumulative step times approximated by uniform avg_step_time
        first_reclaim_time = first_reclaim_global * avg_step_time

    # --- capacity check ---
    seq_kv_bytes_per_hbf = (I + O) * bpt_hbf_padded
    total_kv_bytes_per_hbf = workload.batch_size * seq_kv_bytes_per_hbf
    capacity_fit = total_kv_bytes_per_hbf <= hbf.capacity_bytes
    tokens_to_fill_first_stripe = geom.tokens_per_block * hbf.total_planes_per_hbf

    # --- scale reclaim totals ---
    reclaim_per_gpu = total_reclaim_seq * hbf.hbfs_per_gpu * workload.batch_size
    reclaim_system = reclaim_per_gpu * hbf.num_gpus

    avg_reads_per_block = (
        sum(b.total_page_reads for b in block_stats) / len(block_stats)
        if block_stats else 0.0
    )

    # --- reclaim throughput & periodicity ---
    # Reclaim period on the hot block: once it first hits threshold, how many
    # decode steps elapse before it hits threshold again? With PBBF the hot
    # block reads `hottest_per_step_reads` pages every decode step once it has
    # accumulated num_stored tokens, so that's threshold / per_step_reads
    # decode steps × avg_step_time seconds.
    block_size_bytes = hbf.page_size_bytes * hbf.pages_per_block
    INF = float("inf")
    if hottest_per_step_reads > 0:
        decode_steps_between = reclaim.threshold_page_reads / hottest_per_step_reads
        hottest_period_s = decode_steps_between * avg_step_time if avg_step_time > 0 else INF
    else:
        decode_steps_between = INF
        hottest_period_s = INF

    # Reclaim BW per HBF is computed from the *integer* event count produced
    # by the per-block analysis: for every block b on the HBF, the number of
    # reclaim events that fire over the decode is floor(total_reads_b / threshold),
    # which sums to `total_reclaim_seq` per single sequence (already accumulated
    # above). With batch B, each batch element gets its own block table, so
    # the total integer event count on the HBF is `total_reclaim_seq × B`,
    # and `bytes_moved = events × block_size`.
    #
    # We also retain a fractional (sum_reads / threshold) estimate as a
    # *separate* metric for asymptotic / short-workload comparison — in long
    # workloads the two converge, but in short workloads the integer count
    # may be 0 while the fractional estimate is finite. The CSV exposes both.
    events_per_hbf_integer = total_reclaim_seq * workload.batch_size
    reclaim_bw_per_hbf = (
        events_per_hbf_integer * block_size_bytes / total_decode_time
        if total_decode_time > 0 else 0.0
    )
    reclaim_bw_per_gpu = reclaim_bw_per_hbf * hbf.hbfs_per_gpu
    reclaim_bw_system = reclaim_bw_per_gpu * hbf.num_gpus

    # Fractional / asymptotic estimate, kept for reference.
    sum_seq_len = O * (2 * I + O + 1) // 2 if O > 0 else 0
    total_reads_pages_per_hbf = (
        workload.batch_size * geom.pages_per_token_per_hbf * sum_seq_len
    )
    expected_events_per_hbf = (
        total_reads_pages_per_hbf / reclaim.threshold_page_reads
        if reclaim.threshold_page_reads > 0 else 0.0
    )

    # Per-cycle burst size: how many blocks reclaim simultaneously in one
    # reclaim wave, and how much data is moved per wave.
    stripe_capacity_tokens = max(1, geom.tokens_per_block * hbf.total_planes_per_hbf)
    saturated_stripes_at_end = (I + O) // stripe_capacity_tokens
    blocks_per_cycle_per_hbf = (
        saturated_stripes_at_end * hbf.total_planes_per_hbf * workload.batch_size
    )
    bytes_per_cycle_per_hbf = blocks_per_cycle_per_hbf * block_size_bytes
    bytes_per_cycle_per_gpu = bytes_per_cycle_per_hbf * hbf.hbfs_per_gpu
    bytes_per_cycle_system = bytes_per_cycle_per_gpu * hbf.num_gpus
    cycles_during_decode = (
        total_decode_time / hottest_period_s
        if hottest_period_s > 0 and total_decode_time > 0 else 0.0
    )

    # Overhead fraction = how much of the HBF's nominal bandwidth is consumed
    # just by reclaim writes. >1.0 means reclaim demand can't fit within the
    # HBF's read+write budget at all → reclaim becomes a hard system bottleneck.
    reclaim_overhead_fraction_per_hbf = (
        reclaim_bw_per_hbf / hbf.effective_bandwidth
        if hbf.effective_bandwidth > 0 else 0.0
    )
    burst_duration_per_cycle_s = (
        bytes_per_cycle_per_hbf / hbf.effective_bandwidth
        if hbf.effective_bandwidth > 0 else 0.0
    )

    # HBF lifetime under sustained writes.
    #   Total writes (bytes) the HBF can absorb  =  PE × capacity
    #   Two write sources contribute:
    #     (a) KV cache writes — full I+O tokens for each batch element get
    #         written once per session (prefill writes I bytes at once, decode
    #         adds one token's bytes per step → total = (I+O) × batch × bpt_padded)
    #     (b) Reclaim writes — already accumulated in `reclaim_bw_per_hbf`
    pe = reclaim.pe_cycle_limit
    write_budget = pe * hbf.capacity_bytes
    SECONDS_PER_YEAR = 365.25 * 86400

    kv_write_bytes_per_hbf_total = workload.batch_size * (I + O) * bpt_hbf_padded
    kv_write_bw_per_hbf = (
        kv_write_bytes_per_hbf_total / total_decode_time
        if total_decode_time > 0 else 0.0
    )

    # Weight read-reclaim: when weights live on HBF they are re-read once per
    # forward pass (every decode step). Each weight block therefore accumulates
    # pages_per_block × (activated fraction) read-disturbances per step, and all
    # blocks cross threshold together, rewriting the whole shard per wave.
    total_hbfs = hbf.hbfs_per_gpu * hbf.num_gpus
    weight_stored_per_hbf = (
        model.total_weight_bytes() / total_hbfs
        if (hbf.weights_storage == "hbf" and total_hbfs > 0) else 0.0
    )
    if weight_stored_per_hbf > 0 and reclaim.threshold_page_reads > 0 and O > 0:
        w_read_per_hbf_per_step = weight_bytes_per_gpu / hbf.hbfs_per_gpu
        weight_read_fraction = w_read_per_hbf_per_step / weight_stored_per_hbf
        # Reads on each weight block over the whole decode (fractional events).
        reads_per_wblock_total = O * hbf.pages_per_block * weight_read_fraction
        weight_reclaim_events_per_hbf = (
            reads_per_wblock_total / reclaim.threshold_page_reads
        )
        num_weight_blocks = weight_stored_per_hbf / block_size_bytes
        weight_reclaim_bytes_per_hbf = (
            num_weight_blocks * weight_reclaim_events_per_hbf * block_size_bytes
        )
        weight_reclaim_bw_per_hbf = (
            weight_reclaim_bytes_per_hbf / total_decode_time
            if total_decode_time > 0 else 0.0
        )
    else:
        weight_reclaim_events_per_hbf = 0.0
        weight_reclaim_bytes_per_hbf = 0.0
        weight_reclaim_bw_per_hbf = 0.0

    total_reclaim_bw_per_hbf = reclaim_bw_per_hbf + weight_reclaim_bw_per_hbf
    total_write_bw_per_hbf = kv_write_bw_per_hbf + total_reclaim_bw_per_hbf

    def _life(bw: float) -> tuple[float, float]:
        s = write_budget / bw if bw > 0 else float("inf")
        y = s / SECONDS_PER_YEAR if s != float("inf") else float("inf")
        return s, y

    lifetime_seconds_kv_only,      lifetime_years_kv_only      = _life(kv_write_bw_per_hbf)
    lifetime_seconds_reclaim_only, lifetime_years_reclaim_only = _life(total_reclaim_bw_per_hbf)
    lifetime_seconds_wrec,         lifetime_years_wrec         = _life(weight_reclaim_bw_per_hbf)
    lifetime_seconds,              lifetime_years              = _life(total_write_bw_per_hbf)

    reclaim_wear_overhead_x = (
        total_write_bw_per_hbf / kv_write_bw_per_hbf
        if kv_write_bw_per_hbf > 0 else 1.0
    )

    return SimulationResult(
        model_name=model.name,
        write_policy=cfg.write_policy,
        input_tokens=I,
        output_tokens=O,
        batch_size=workload.batch_size,

        page_size_bytes=hbf.page_size_bytes,
        pages_per_block=hbf.pages_per_block,
        planes_per_die=hbf.planes_per_die,
        dies_per_stack=hbf.dies_per_stack,
        stacks_per_hbf=hbf.stacks_per_hbf,
        total_dies_per_hbf=hbf.total_dies_per_hbf,
        total_planes_per_hbf=hbf.total_planes_per_hbf,
        hbfs_per_gpu=hbf.hbfs_per_gpu,
        num_gpus=hbf.num_gpus,
        hbf_capacity_bytes=hbf.capacity_bytes,
        hbf_bandwidth_bytes_per_sec=hbf.bandwidth_bytes_per_sec,
        bandwidth_utilization=hbf.bandwidth_utilization,
        nand_tR_ns=hbf.nand_tR_ns,
        hbm_bandwidth_bytes_per_sec=hbf.hbm_bandwidth_bytes_per_sec,
        weights_storage=hbf.weights_storage,
        reclaim_threshold=reclaim.threshold_page_reads,

        kv_bytes_per_token_global=model.kv_bytes_per_token_global(),
        kv_bytes_per_token_per_gpu=model.kv_bytes_per_token_per_gpu(hbf.num_gpus),
        kv_bytes_per_token_per_hbf=geom.bytes_per_token_per_hbf,
        pages_per_token_per_hbf=geom.pages_per_token_per_hbf,
        tokens_per_block=geom.tokens_per_block,
        blocks_per_sequence_per_hbf=geom.blocks_per_sequence_per_hbf,
        seq_kv_bytes_per_hbf=seq_kv_bytes_per_hbf,
        total_kv_bytes_per_hbf=total_kv_bytes_per_hbf,
        blocks_per_plane_per_hbf=hbf.blocks_per_plane,
        tokens_to_fill_first_stripe=tokens_to_fill_first_stripe,
        capacity_fit=capacity_fit,

        per_plane_bandwidth_bytes_per_sec=hbf.per_plane_bandwidth,
        planes_needed_for_target_bw=hbf.planes_needed_for_target_bw,
        plane_oversubscription=hbf.plane_oversubscription,

        hottest_block_stripe=hottest.stripe if hottest else -1,
        hottest_block_plane=hottest.plane if hottest else -1,
        hottest_block_total_reads=hottest.total_page_reads if hottest else 0,
        hottest_block_reclaim_events=hottest.reclaim_events if hottest else 0,
        first_reclaim_step=first_reclaim_global,
        first_reclaim_token=first_reclaim_token,
        first_reclaim_time_s=first_reclaim_time,
        total_reclaim_events_per_seq=total_reclaim_seq,
        total_reclaim_events_per_gpu=reclaim_per_gpu,
        total_reclaim_events_system=reclaim_system,
        avg_reads_per_block=avg_reads_per_block,
        blocks_touched_by_reads=blocks_touched,

        active_weight_bytes_per_gpu=weight_bytes_per_gpu,
        weight_load_time_per_step_s=weight_time,
        avg_kv_step_time_s=avg_kv_step_time,
        avg_step_time_s=avg_step_time,
        total_decode_time_s=total_decode_time,
        avg_token_latency_ms=avg_step_time * 1000.0,
        avg_kv_read_bandwidth_per_hbf_bytes_per_sec=avg_bw_per_hbf,
        bottleneck=bottleneck,

        reclaim_bytes_per_event=block_size_bytes,
        decode_steps_between_reclaims_hottest=decode_steps_between,
        hottest_block_reclaim_period_s=hottest_period_s,
        saturated_stripes_at_end=saturated_stripes_at_end,
        blocks_reclaimed_per_cycle_per_hbf=blocks_per_cycle_per_hbf,
        reclaim_size_per_cycle_per_hbf_bytes=bytes_per_cycle_per_hbf,
        reclaim_size_per_cycle_per_gpu_bytes=bytes_per_cycle_per_gpu,
        reclaim_size_per_cycle_system_bytes=bytes_per_cycle_system,
        reclaim_cycles_during_decode=cycles_during_decode,
        reclaim_overhead_fraction_per_hbf=reclaim_overhead_fraction_per_hbf,
        burst_duration_per_cycle_s=burst_duration_per_cycle_s,
        pe_cycle_limit=pe,
        hbf_total_write_budget_bytes=write_budget,
        kv_write_bytes_per_hbf=int(kv_write_bytes_per_hbf_total),
        kv_write_bw_per_hbf_bytes_per_sec=kv_write_bw_per_hbf,
        weight_reclaim_events_per_hbf=weight_reclaim_events_per_hbf,
        weight_reclaim_write_bw_per_hbf_bytes_per_sec=weight_reclaim_bw_per_hbf,
        hbf_lifetime_seconds_kv_only=lifetime_seconds_kv_only,
        hbf_lifetime_years_kv_only=lifetime_years_kv_only,
        hbf_lifetime_seconds_reclaim_only=lifetime_seconds_reclaim_only,
        hbf_lifetime_years_reclaim_only=lifetime_years_reclaim_only,
        hbf_lifetime_seconds_weight_reclaim_only=lifetime_seconds_wrec,
        hbf_lifetime_years_weight_reclaim_only=lifetime_years_wrec,
        hbf_lifetime_seconds=lifetime_seconds,
        hbf_lifetime_years=lifetime_years,
        reclaim_wear_overhead_x=reclaim_wear_overhead_x,
        expected_reclaim_events_per_hbf=expected_events_per_hbf,
        reclaim_write_bw_per_hbf_bytes_per_sec=reclaim_bw_per_hbf,
        reclaim_write_bw_per_gpu_bytes_per_sec=reclaim_bw_per_gpu,
        reclaim_write_bw_system_bytes_per_sec=reclaim_bw_system,
        expected_reclaim_write_bw_per_hbf_bytes_per_sec=(
            expected_events_per_hbf * block_size_bytes / total_decode_time
            if total_decode_time > 0 else 0.0
        ),

        block_stats=block_stats,
    )
