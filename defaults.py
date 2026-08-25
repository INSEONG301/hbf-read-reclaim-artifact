"""Default simulation configuration and the request-rate heuristic.

`DEFAULT_BASE` is the canonical configuration the figure scripts start from:
LLaMA-3.1-405B on an 8x8 HBF system, SLC read-reclaim, weights on HBF. The NAND
page size defaults to 4 KB here.
"""
from __future__ import annotations

from hbf_rr.configs import (
    HBFConfig, WorkloadConfig, ReclaimConfig, SimulationConfig,
)
from hbf_rr.models import LLAMA_3_1_405B


DEFAULT_BASE = SimulationConfig(
    model=LLAMA_3_1_405B,
    hbf=HBFConfig(
        bandwidth_bytes_per_sec=1.6e12,
        capacity_bytes=512 * (1024 ** 3),    # 512 GiB per HBF
        page_size_bytes=4 * 1024,            # 4 KB page (artifact default)
        pages_per_block=196 * 4,             # Z-NAND SLC default (= 784)
        planes_per_die=8,
        dies_per_stack=8,
        stacks_per_hbf=16,
        hbfs_per_gpu=8,
        num_gpus=8,
        bandwidth_utilization=1.0,
        nand_tR_ns=3_000.0,
        hbm_bandwidth_bytes_per_sec=1.6e12,
        weights_storage="hbf",               # weights resident on HBF (KV cache always on HBF)
    ),
    workload=WorkloadConfig(input_tokens=16384, output_tokens=16384, batch_size=1),
    reclaim=ReclaimConfig(threshold_page_reads=1_000_000),
    write_policy="plane_balanced_block_fill",
)


def _auto_rates_for_workload(I: int, O: int, max_concurrent: int,
                             n: int = 6) -> list[float]:
    """`n` log-spaced request rates covering the under-saturated -> saturated
    transition for an (I, O) workload at `max_concurrent` concurrency.

    Heuristic single-active step time (ms):
        step_ms ~ 7.9 + (I + O/2) x 8 KiB / HBF_BW   (LLaMA TP=8 default)
    Saturation rate ~ max_concurrent / (O x step_ms); rates span 0.05x..1.5x.
    """
    bpt = 8192          # per HBF per token, LLaMA padded
    hbf_bw = 1.6e12
    avg_seq = I + O / 2
    step_ms = 7.9 + avg_seq * bpt / hbf_bw * 1000
    decode_s = O * step_ms / 1000
    if decode_s <= 0:
        return [0.001]
    max_rate = max_concurrent / decode_s
    fractions = [0.05, 0.15, 0.3, 0.6, 1.0, 1.5][:n]
    rates = [max_rate * f for f in fractions]

    def _round(r):
        if r >= 1:
            return round(r, 2)
        if r >= 0.01:
            return round(r, 4)
        return round(r, 6)

    return [_round(r) for r in rates]
