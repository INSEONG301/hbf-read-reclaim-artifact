"""Write placement policies for KV cache on HBF.

A policy is a callable that, given (model, hbf, workload, reclaim_threshold),
returns a `StorageGeometry` plus a list of `BlockReadSummary` objects — one per
KV-cache block on a single HBF, holding one batch element's KV cache.

Each `BlockReadSummary` contains the per-block aggregate metrics needed by the
simulator (total page reads over the entire decode, decode step of the first
reclaim, count of reclaim events). These are computed analytically, so the
policy is O(num_blocks) per simulation rather than O(num_blocks × decode_steps),
which matters once the sweep reaches output_tokens of 1M.

Default policy is `plane_balanced_block_fill` (PBBF): tokens are written
round-robin across all `total_planes_per_hbf` planes (= planes_per_die ×
dies_per_stack × stacks_per_hbf), and within each plane the current block is filled before the
next opens. As a result block 0 of each plane is read on nearly every decode
step, with token sparsity = 1/P inside the block.

`uniform_block_spread` is the planned "even spread" policy in which a token's
pages_per_token pages are split across many distinct blocks so that no single
block is read more than the global mean.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

from .configs import HBFConfig, ModelConfig, WorkloadConfig


# ---------------------------------------------------------------------------
# Per-token storage geometry
# ---------------------------------------------------------------------------

@dataclass
class StorageGeometry:
    bytes_per_token_per_gpu: float
    bytes_per_token_per_hbf: float
    pages_per_token_per_hbf: int          # ceil-padded to whole pages
    raw_pages_per_token_per_hbf: float    # un-padded (for diagnostics)
    tokens_per_block: int                 # floor(pages_per_block / pages_per_token)
    blocks_per_sequence_per_hbf: int      # blocks consumed by one full sequence


def compute_geometry(model: ModelConfig, hbf: HBFConfig, total_tokens: int) -> StorageGeometry:
    bpt_gpu = model.kv_bytes_per_token_per_gpu(hbf.num_gpus)
    bpt_hbf = bpt_gpu / hbf.hbfs_per_gpu
    raw_pages = bpt_hbf / hbf.page_size_bytes
    pages_per_token = max(1, math.ceil(raw_pages))
    tokens_per_block = max(1, hbf.pages_per_block // pages_per_token)
    blocks_per_seq = math.ceil(total_tokens / tokens_per_block)
    return StorageGeometry(
        bytes_per_token_per_gpu=bpt_gpu,
        bytes_per_token_per_hbf=bpt_hbf,
        pages_per_token_per_hbf=pages_per_token,
        raw_pages_per_token_per_hbf=raw_pages,
        tokens_per_block=tokens_per_block,
        blocks_per_sequence_per_hbf=blocks_per_seq,
    )


# ---------------------------------------------------------------------------
# Per-block aggregate read summary
# ---------------------------------------------------------------------------

@dataclass
class BlockReadSummary:
    stripe: int                          # block index within one plane (PBBF) or block id (uniform)
    plane: int                           # plane id (PBBF) or 0 (uniform)
    num_tokens_stored: int               # tokens (or page slots, for uniform) stored
    total_page_reads: int                # cumulative page reads over the entire decode
    first_reclaim_step: Optional[int]    # 1-indexed decode step of first threshold crossing
    reclaim_events: int                  # = total_page_reads // threshold (with reset semantics)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _G(n: int, P: int) -> int:
    """sum_{u=1}^{n} ceil(u/P). Returns 0 for n <= 0.

    Decomposes [1..n] into full P-blocks of identical ceil values plus a tail.
    """
    if n <= 0:
        return 0
    q, r = divmod(n, P)
    return P * q * (q + 1) // 2 + r * (q + 1)


def _first_step_phase_b(
    t_first: int, t_b_end: int,
    u_lo: int, B: int, I: int, P: int,
    pages_per_token: int, threshold: int,
) -> int:
    """Binary search for the smallest t in [t_first, t_b_end] (phase B / ramp)
    such that cumulative page reads on this block >= threshold."""
    lo, hi = t_first, t_b_end
    g_base = _G(u_lo - 1, P)
    while lo < hi:
        mid = (lo + hi) // 2
        u_mid = I + mid - B
        cumsum_pages = (_G(u_mid, P) - g_base) * pages_per_token
        if cumsum_pages >= threshold:
            hi = mid
        else:
            lo = mid + 1
    return lo


# ---------------------------------------------------------------------------
# Plane-Balanced Block-Fill (default)
# ---------------------------------------------------------------------------

def plane_balanced_block_fill(
    model: ModelConfig,
    hbf: HBFConfig,
    workload: WorkloadConfig,
    threshold: int,
) -> Tuple[StorageGeometry, List[BlockReadSummary]]:
    """PBBF block-read summaries computed analytically.

    Layout: token i lands in plane (i mod P), within that plane the current
    block is filled before the next. Block (stripe s, plane p) holds tokens
    {s*K*P + p + j*P : j in [0, K-1]} that exist (i.e. < total_tokens).

    Read pattern at decode step t (1..O): every token in [0, I+t) is read.
    For a block with start B and `num_stored` stored tokens, the per-step
    reads (in tokens) are min(num_stored, ceil(max(0, I+t-B)/P)) — a ramp from
    0 to num_stored followed by a plateau.
    """
    P = hbf.total_planes_per_hbf
    I, O = workload.input_tokens, workload.output_tokens
    total_tokens = I + O
    geom = compute_geometry(model, hbf, total_tokens)
    K = geom.tokens_per_block
    pages_per_token = geom.pages_per_token_per_hbf

    summaries: List[BlockReadSummary] = []
    num_stripes = math.ceil(total_tokens / (K * P))

    for s in range(num_stripes):
        for p in range(P):
            B = s * K * P + p
            if B >= total_tokens:
                continue
            # How many of this block's K possible slots actually receive tokens
            max_j = min(K - 1, (total_tokens - 1 - B) // P)
            num_stored = max_j + 1
            if num_stored <= 0:
                continue

            # Phase boundaries in decode-step coordinates
            t_first = max(1, B - I + 1)             # first step with any tokens visible
            # Smallest t at which reads(t) == num_stored (plateau begins)
            t_plateau = max(t_first, B - I + (num_stored - 1) * P + 1)
            if t_first > O:
                summaries.append(BlockReadSummary(
                    stripe=s, plane=p, num_tokens_stored=num_stored,
                    total_page_reads=0, first_reclaim_step=None, reclaim_events=0,
                ))
                continue

            # --- Phase B (ramp) contribution in tokens ---
            t_b_end = min(O, t_plateau - 1)
            if t_first <= t_b_end:
                u_lo = I + t_first - B
                u_hi = I + t_b_end - B
                phase_b_tokens = _G(u_hi, P) - _G(u_lo - 1, P)
            else:
                u_lo = I + t_first - B  # still needed if we later search phase B
                phase_b_tokens = 0

            # --- Phase C (plateau) contribution ---
            if t_plateau <= O:
                count_c = O - t_plateau + 1
                phase_c_tokens = num_stored * count_c
            else:
                phase_c_tokens = 0

            total_tokens_read = phase_b_tokens + phase_c_tokens
            total_page_reads = total_tokens_read * pages_per_token

            # --- First reclaim step ---
            first_reclaim: Optional[int] = None
            if total_page_reads >= threshold:
                # full phase-B accumulation (even if it stretches past O is fine,
                # but we clip at the actual end of phase B used here)
                if t_first <= t_b_end:
                    phase_b_full_pages = (_G(I + t_b_end - B, P) - _G(u_lo - 1, P)) * pages_per_token
                else:
                    phase_b_full_pages = 0
                if phase_b_full_pages >= threshold:
                    first_reclaim = _first_step_phase_b(
                        t_first=t_first, t_b_end=t_b_end,
                        u_lo=u_lo, B=B, I=I, P=P,
                        pages_per_token=pages_per_token, threshold=threshold,
                    )
                else:
                    deficit = threshold - phase_b_full_pages
                    step_in_c = math.ceil(deficit / (num_stored * pages_per_token))
                    candidate = t_plateau + step_in_c - 1
                    first_reclaim = candidate if candidate <= O else None

            events = total_page_reads // threshold
            summaries.append(BlockReadSummary(
                stripe=s, plane=p, num_tokens_stored=num_stored,
                total_page_reads=int(total_page_reads),
                first_reclaim_step=first_reclaim,
                reclaim_events=int(events),
            ))

    return geom, summaries


# ---------------------------------------------------------------------------
# Uniform block spread (planned future policy)
# ---------------------------------------------------------------------------

def uniform_block_spread(
    model: ModelConfig,
    hbf: HBFConfig,
    workload: WorkloadConfig,
    threshold: int,
) -> Tuple[StorageGeometry, List[BlockReadSummary]]:
    """Round-robin every token's pages_per_token pages across as many blocks as
    possible. In this approximation each block stores N_per_block ≈ total_pages
    / num_blocks page slots, each from a distinct token. During attention at
    seq_len = I+t, every block whose stored token has index < I+t gets one
    page-read for each such stored page.

    Closed-form approximation: assume blocks are filled cyclically in token
    order, so block b holds tokens whose page-slot index falls in
    [b*N, (b+1)*N). Then for I+t > b*N/pages_per_token, the block gets
    min(N, max(0, I+t - b*N/pages_per_token)) page reads at step t.

    For the threshold analysis we only need:
      - total page reads / block over the decode
      - first decode step where cumsum >= threshold (closed form via
        quadratic inversion since reads grow ~linearly with seq_len)
    """
    I, O = workload.input_tokens, workload.output_tokens
    total_tokens = I + O
    geom = compute_geometry(model, hbf, total_tokens)
    pages_per_token = geom.pages_per_token_per_hbf

    total_pages = total_tokens * pages_per_token
    num_blocks = min(hbf.num_blocks_per_hbf, max(1, total_pages))
    # Tokens whose pages land in block b: any token i whose page index range
    # intersects [b*N, (b+1)*N) where N = pages_per_block (block capacity in
    # page slots, capped at total_pages / num_blocks ... but we approximate by
    # filling all blocks). Each block holds N_per_block page slots.
    n_per_block = max(1, total_pages // num_blocks)
    # tokens per block (each block stores n_per_block page slots; each token
    # contributes pages_per_token slots): tokens_per_block_uniform may be < 1
    # but for the sum we treat it as fractional via the formula below.
    tokens_per_block_uniform = n_per_block / pages_per_token

    # Per-step page reads on a typical "fully filled" block at seq_len s:
    # reads_per_step(s) = min(n_per_block, max(0, (s - first_seen) * pages_per_token / num_blocks * num_blocks))
    # We simplify: total page reads across the system per decode step is
    # seq_len * pages_per_token. Divided evenly across num_blocks_in_use, each
    # block sees seq_len * pages_per_token / num_blocks page reads per step.
    # Most blocks will see fractional reads; we report aggregate.

    # For threshold analysis we use the *worst-case* block, which still
    # outperforms PBBF: with even spread the block read rate is reduced by a
    # factor of num_blocks_used / blocks_used_in_pbbf_hottest_stripe.

    # Sum over decode: sum_{t=1..O} (I+t) * pages_per_token / num_blocks
    #                = pages_per_token / num_blocks * O*(2I+O+1)/2
    avg_block_total = pages_per_token * O * (2 * I + O + 1) // (2 * num_blocks)

    # First reclaim: find smallest t with cumulative >= threshold
    # cumulative(t) = pages_per_token / num_blocks * t*(2I+t+1)/2
    # Solve t*(2I+t+1) >= 2 * num_blocks * threshold / pages_per_token
    # Quadratic: t^2 + (2I+1)t - 2*nb*th/ppt >= 0
    if avg_block_total < threshold:
        first_reclaim = None
    else:
        Rhs = 2 * num_blocks * threshold / pages_per_token
        disc = (2 * I + 1) ** 2 + 4 * Rhs
        t_star = (-(2 * I + 1) + math.sqrt(disc)) / 2
        first_reclaim = max(1, math.ceil(t_star))
        if first_reclaim > O:
            first_reclaim = None

    # Emit a single representative "block" (since they're all equivalent under
    # this approximation). Calling code can multiply by num_blocks for system
    # totals if it cares.
    summaries: List[BlockReadSummary] = [BlockReadSummary(
        stripe=0, plane=0,
        num_tokens_stored=int(tokens_per_block_uniform),
        total_page_reads=int(avg_block_total),
        first_reclaim_step=first_reclaim,
        reclaim_events=int(avg_block_total // threshold),
    )]
    # Duplicate for `num_blocks` so reclaim-event totals scale correctly. To
    # keep memory bounded for huge sweeps, cap at 1024 representative blocks
    # and scale the event count.
    n_copies = min(num_blocks, 1024)
    scale = num_blocks / n_copies
    summaries = [BlockReadSummary(
        stripe=i, plane=0,
        num_tokens_stored=int(tokens_per_block_uniform),
        total_page_reads=int(avg_block_total),
        first_reclaim_step=first_reclaim,
        reclaim_events=int((avg_block_total // threshold) * scale),
    ) for i in range(n_copies)]
    return geom, summaries


POLICIES: Dict[str, Callable] = {
    "plane_balanced_block_fill": plane_balanced_block_fill,
    "uniform": uniform_block_spread,
}
