"""Predefined model KV-cache configurations.

Only fields that affect KV cache size are load-bearing:
  num_layers, num_kv_heads, head_dim, dtype_bytes.
MoE specifics (active experts, etc.) do not change KV cache since the attention
projection is dense; they're noted for reference only.
"""
from .configs import ModelConfig


LLAMA_3_1_405B = ModelConfig(
    name="LLaMA-3.1-405B",
    num_layers=126,
    num_kv_heads=8,         # GQA: 128 attn heads / 16 query groups
    head_dim=128,
    dtype_bytes=2,          # FP16
    total_params=405_000_000_000,
    active_params=405_000_000_000,    # dense
    hidden_size=16384,
    num_attention_heads=128,
    notes="Dense. KV cache size per token (global) ≈ 504 KB. Weights ≈ 810 GB FP16.",
)


QWEN3_235B_A22B = ModelConfig(
    name="Qwen3-235B-A22B",
    num_layers=94,
    num_kv_heads=4,         # GQA
    head_dim=128,
    dtype_bytes=2,
    total_params=235_000_000_000,
    active_params=22_000_000_000,     # MoE: 22B activated for ONE token
    num_experts=128,                  # MoE: 128 experts total
    num_experts_per_token=8,          #      top-8 per token
    hidden_size=4096,
    num_attention_heads=64,
    notes=("MoE 128/8. KV cache size per token (global) ≈ 188 KB. "
           "Single-token active weights ≈ 44 GB FP16; effective weight load "
           "grows toward 470 GB (total) as batch_size approaches num_experts."),
)


LLAMA4_MAVERICK = ModelConfig(
    name="Llama-4-Maverick",
    num_layers=48,
    num_kv_heads=8,          # GQA
    head_dim=128,
    dtype_bytes=2,           # BF16
    total_params=400_000_000_000,
    active_params=17_000_000_000,     # MoE: ~17B active per token
    num_experts=128,
    num_experts_per_token=1,          # top-1 routed (+ shared expert), approx
    hidden_size=5120,
    num_attention_heads=40,
    notes=("Matches Kyung et al. (IEEE CAL 2026). GQA KV cache = "
           "128 x 8 x 2 x 48 x 2 B = 192 KB/token (global). MoE 400B total / "
           "~17B active."),
)


MODEL_REGISTRY = {
    LLAMA_3_1_405B.name: LLAMA_3_1_405B,
    QWEN3_235B_A22B.name: QWEN3_235B_A22B,
    LLAMA4_MAVERICK.name: LLAMA4_MAVERICK,
}
