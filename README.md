# HBF Read-Reclaim Simulator

An in-house simulator for **NAND read-disturb reclaim** on **High-Bandwidth
Flash (HBF)** used as KV-cache / weight storage during LLM inference.

When the KV cache (and optionally the model weights) are held on HBF, each
NAND block accumulates read-disturb from every decode step. Once a block's
cumulative page reads cross the read-reclaim threshold, the block must be
rewritten (reclaimed), consuming one program/erase cycle. This tool quantifies
how much reclaim write traffic that generates, how it shortens HBF endurance,
and how the resulting write stalls perturb per-token latency (TBT).

Prior endurance studies of HBF for KV caches count only KV-cache writes and
omit read-disturb reclaim. This simulator adds the reclaim term (both KV-cache
and weight reclaim) and reports the endurance and latency it costs.

## What it models

- **Read-disturb accounting.** KV writes are placed with a *plane-balanced
  block-fill* policy (tokens striped round-robin across all planes; each block
  filled before the next opens). Per-block cumulative page reads over the whole
  decode are computed in closed form — an `O(#blocks)` pass, not
  `O(#blocks x #decode-steps)` — so million-token decodes stay tractable.
- **Reclaim.** A block whose cumulative reads reach the threshold is reclaimed
  (rewritten), costing one P/E cycle. Two sources: KV-cache reclaim (the hot
  block) and weight reclaim (weights re-read every forward pass, so a whole
  weight shard reclaims in a periodic burst).
- **Endurance.** `lifetime = P/E_budget x capacity / total_write_bandwidth`,
  where the write bandwidth sums KV writes, KV reclaim, and weight reclaim.
- **Latency.** An event-driven decode/queueing model injects the reclaim events
  as write stalls and reports per-token TBT percentiles (p50 / p99 / max) with
  and without reclaim, under a Poisson request stream.

## Requirements

- Python >= 3.10
- `numpy`, `matplotlib`  (`pip install -r requirements.txt`)

## Quick start

```bash
pip install -r requirements.txt

# one-configuration example (endurance + latency, printed to stdout)
PYTHONPATH=. python3 example.py

# paper figures (each writes under plots/paper/)
./run/run_write_lifetime.sh     # (a) write overhead + (b) HBF lifetime vs output length
./run/run_tbt_slo.sh            # p99 / max TBT at SLO-sized concurrency, ideal vs typical SLC
./run/run_lifetime_heatmap.sh   # HBF lifetime heatmap over input x output length (2 models)
./run/run_sensitivity.sh        # lifetime sensitivity: NAND layers / bandwidth / KV-vs-weight
```

Requires Python >= 3.10. If your default `python3` is older, point the scripts
at a newer interpreter, e.g. `PYTHON=python3.11 ./run/run_write_lifetime.sh`.
The run scripts accept `HBF_LAYERS` (default 162) and, where relevant,
`PAGE_SIZE` (bytes, default 4096) env overrides.

## Repository layout

```
hbf_rr/
  configs.py     # HBF / model / workload / reclaim config dataclasses
  models.py      # example model configs (dense and MoE)
  policies.py    # write-placement policy + closed-form per-block read summary
  simulator.py   # single-run endurance & write-traffic model  -> simulate()
  queueing.py    # event-driven request-stream / TBT model
  sweep.py       # parameter-sweep helpers
  plotting.py    # plotting helpers
defaults.py      # DEFAULT_BASE config (4 KB page) + request-rate heuristic
example.py       # minimal end-to-end example
figures/
  lifetime_heatmap.py        # lifetime heatmap over (input x output) length
  lifetime_triptych.py       # lifetime sensitivity (layers / bandwidth / KV-vs-weight)
  tbt_slo.py                 # SLO-sized-concurrency TBT (core, imported by tbt_slo_thr)
  tbt_slo_thr.py             # SLO TBT figure: ideal vs typical-SLC threshold
  write_lifetime.py          # (a) write overhead + (b) HBF lifetime figure
  _common.py                 # write-volume / lifetime helpers
run/             # one shell script per paper figure
  run_write_lifetime.sh
  run_tbt_slo.sh
  run_lifetime_heatmap.sh
  run_sensitivity.sh
```

## Key parameters (defaults)

| Parameter | Default |
| --- | --- |
| HBF bandwidth / capacity | 1.6 TB/s, 512 GiB per HBF |
| NAND page / block | 4 KB page, `pages_per_block = 4 x layers` |
| NAND read latency (t_R) | 3 us (SLC) |
| Read-reclaim threshold | 1e6 cumulative page reads per block |
| P/E budget | 1e5 cycles per block (SLC) |
| Topology | 8 HBF stacks/GPU x 8 GPUs (tensor parallel) |
| Weights storage | HBF (re-read every forward pass) |

All are configurable via the dataclasses in `hbf_rr/configs.py` and the
`DEFAULT_BASE` config in `defaults.py`.
