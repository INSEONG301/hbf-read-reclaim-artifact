"""Configuration dataclasses for the HBF read-reclaim sweep."""
from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict


@dataclass
class HBFConfig:
    # Top-line HBF spec
    bandwidth_bytes_per_sec: float = 1.6e12        # 1.6 TB/s per HBF
    capacity_bytes: int = 512 * (1024 ** 3)        # 512 GiB per HBF
    page_size_bytes: int = 2 * 1024                # 2 KB
    pages_per_block: int = 196 * 4                     # Z-NAND SLC default (= 784 pages)
    # HBF internal hierarchy:
    #   HBF  =  stacks_per_hbf  ×  dies_per_stack  ×  planes_per_die
    planes_per_die: int = 8                        # planes per NAND die
    dies_per_stack: int = 8                        # dies stacked in one stack
    stacks_per_hbf: int = 16                       # stacks in one HBF
    # System topology
    hbfs_per_gpu: int = 8
    num_gpus: int = 8                              # TP degree
    # Efficiency knobs
    bandwidth_utilization: float = 1.0
    # Plane-level NAND timing (used to estimate how many planes must be
    # activated concurrently to reach the HBF's nominal bandwidth).
    nand_tR_ns: float = 3_000.0                    # Z-NAND SLC ~3us
    # Where model weights live. KV cache is always on HBF in this study.
    # Default: HBM bandwidth matches HBF bandwidth (1.6 TB/s per stack).
    hbm_bandwidth_bytes_per_sec: float = 1.6e12
    weights_storage: str = "hbf"                   # "hbm" or "hbf"

    # ---- derived ----
    @property
    def total_dies_per_hbf(self) -> int:
        return self.dies_per_stack * self.stacks_per_hbf

    @property
    def total_planes_per_hbf(self) -> int:
        return self.planes_per_die * self.total_dies_per_hbf

    @property
    def block_size_bytes(self) -> int:
        return self.page_size_bytes * self.pages_per_block

    @property
    def num_blocks_per_hbf(self) -> int:
        return self.capacity_bytes // self.block_size_bytes

    @property
    def blocks_per_plane(self) -> int:
        """Blocks that fit in one plane on a single HBF.
        = capacity / (total_planes × block_size)."""
        denom = self.total_planes_per_hbf * self.block_size_bytes
        return self.capacity_bytes // denom if denom > 0 else 0

    @property
    def effective_bandwidth(self) -> float:
        return self.bandwidth_bytes_per_sec * self.bandwidth_utilization

    @property
    def per_plane_bandwidth(self) -> float:
        """Bytes/sec a single plane can sustain in fully-pipelined page reads."""
        return self.page_size_bytes / (self.nand_tR_ns * 1e-9)

    @property
    def planes_needed_for_target_bw(self) -> int:
        """Number of planes that must be activated concurrently to sustain
        `bandwidth_bytes_per_sec`. Compare against `total_planes_per_hbf`
        (= planes_per_die × dies_per_stack × stacks_per_hbf) to see if the HBF can reach its
        nominal bandwidth with the given tR."""
        return math.ceil(self.bandwidth_bytes_per_sec / self.per_plane_bandwidth)

    @property
    def plane_oversubscription(self) -> float:
        """planes_needed / total_planes_available. >1 means under-provisioned."""
        return self.planes_needed_for_target_bw / self.total_planes_per_hbf


@dataclass
class ModelConfig:
    name: str
    num_layers: int
    num_kv_heads: int                              # post-GQA
    head_dim: int
    dtype_bytes: int = 2                           # FP16 = 2
    # Parameter counts (used to estimate weight-load latency per decode step).
    # `active_params` = params that must be loaded for ONE token (dense params
    # + activated experts × per-expert params). For dense models set
    # `active_params == total_params`.
    # For MoE, also fill `num_experts` (total experts) and `num_experts_per_token`
    # (top-k). With both set, `effective_active_params(batch_size)` accounts
    # for the union of experts activated across a batch: as batch grows the
    # required weight-load grows toward `total_params`.
    total_params: int = 0
    active_params: int = 0
    num_experts: int = 0
    num_experts_per_token: int = 0
    # Metadata (not used in math)
    hidden_size: int | None = None
    num_attention_heads: int | None = None
    notes: str = ""

    def kv_bytes_per_token_global(self) -> int:
        """Total KV bytes per token across all GPUs / TP ranks."""
        return self.num_layers * 2 * self.num_kv_heads * self.head_dim * self.dtype_bytes

    def kv_bytes_per_token_per_gpu(self, tp: int) -> float:
        """Per-GPU KV bytes per token under TP.

        If tp <= num_kv_heads, KV heads are sharded cleanly.
        If tp > num_kv_heads, KV heads are replicated (typical practice), so
        each GPU still holds 1 head worth (num_kv_heads / min(tp, num_kv_heads))."""
        eff_tp = min(tp, self.num_kv_heads)
        per_head_bytes = self.num_layers * 2 * self.head_dim * self.dtype_bytes
        return per_head_bytes * (self.num_kv_heads / eff_tp)

    def effective_active_params(self, batch_size: int = 1) -> float:
        """Parameters that must be loaded to process a batch of `batch_size`
        tokens on one decode step.

        For dense models (num_experts==0) this is `active_params` regardless
        of batch. For MoE, the union of experts activated across the batch
        grows as `E × (1 - ((E-K)/E)^B)` (random routing assumption); the
        effective load grows from `active_params` (B=1) toward `total_params`
        (B → ∞)."""
        active = float(self.active_params or self.total_params)
        if self.num_experts <= 0 or self.num_experts_per_token <= 0:
            return active
        E = self.num_experts
        K = self.num_experts_per_token
        if K >= E or batch_size <= 1:
            return active if K < E else float(self.total_params)
        # Derive dense + per-expert from (active, total, E, K):
        #   active = dense + K × per_expert
        #   total  = dense + E × per_expert
        per_expert = (self.total_params - self.active_params) / (E - K)
        dense = self.active_params - K * per_expert
        expected_experts = E * (1.0 - ((E - K) / E) ** batch_size)
        return dense + expected_experts * per_expert

    def active_weight_bytes_per_gpu(self, tp: int, batch_size: int = 1) -> float:
        """Bytes of weights loaded per decode step on each GPU.

        Assumes weights tile cleanly across TP ranks. For MoE the per-batch
        union-of-experts adjustment from `effective_active_params` is applied."""
        eff = self.effective_active_params(batch_size)
        return eff * self.dtype_bytes / max(1, tp)

    def total_weight_bytes(self) -> float:
        return self.total_params * self.dtype_bytes


@dataclass
class WorkloadConfig:
    input_tokens: int = 1024
    output_tokens: int = 1024
    batch_size: int = 1


@dataclass
class ReclaimConfig:
    """Read disturbance / read-reclaim threshold.

    Counted as cumulative page reads to any page in a block. When the running
    sum reaches `threshold_page_reads`, the block is reclaimed (counter resets,
    one reclaim event recorded).

    `pe_cycle_limit` is the per-block program/erase budget — each reclaim
    consumes one cycle on the destination block. SLC: ~10^5; MLC: 10^3-10^4;
    TLC: 10^3. Combined with `HBFConfig.capacity_bytes`, this determines the
    HBF lifetime under sustained read-reclaim traffic:
        lifetime [s]  =  pe_cycle_limit × capacity_bytes / reclaim_BW
    """
    threshold_page_reads: int = 1_000_000
    pe_cycle_limit: int = 100_000   # SLC default


@dataclass
class SimulationConfig:
    """Bundle of every input to a single simulation run."""
    model: ModelConfig
    hbf: HBFConfig
    workload: WorkloadConfig
    reclaim: ReclaimConfig
    write_policy: str = "plane_balanced_block_fill"

    def to_flat_dict(self) -> dict:
        out = {"model": self.model.name, "write_policy": self.write_policy}
        out.update({f"hbf_{k}": v for k, v in asdict(self.hbf).items()})
        out.update({f"workload_{k}": v for k, v in asdict(self.workload).items()})
        out.update({f"reclaim_{k}": v for k, v in asdict(self.reclaim).items()})
        return out
