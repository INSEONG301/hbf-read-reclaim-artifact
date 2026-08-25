#!/usr/bin/env bash
# TBT (p99 / max) at SLO-sized concurrency: ideal (threshold = inf) vs typical
# SLC (threshold = 1e6), vs output length, 162-layer, input = 1M. Output:
#   plots/paper/tbt_slo_thr__162layer/tbt_slo_thr__I=1M__vermillion.png
#
# Env: HBF_LAYERS (default 162), PYTHON (default python3).
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=.

HBF_LAYERS="${HBF_LAYERS:-162}" \
TBT_HI_COLOR="${TBT_HI_COLOR:-#D55E00}" TBT_TAG="${TBT_TAG:-__vermillion}" \
    "${PYTHON:-python3}" figures/tbt_slo_thr.py
