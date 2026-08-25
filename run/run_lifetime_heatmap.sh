#!/usr/bin/env bash
# HBF lifetime heatmap over (input x output) length, ideal vs typical-SLC
# reclaim, for two models (dense + MoE). 162-layer, 4 KB page. Output:
#   plots/paper/lifetime_heatmap__162layer__4KBpage/lifetime_heatmap.png
#
# Env: HBF_LAYERS (default 162), PAGE_SIZE bytes (default 4096), PYTHON.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=.

HBF_LAYERS="${HBF_LAYERS:-162}" PAGE_SIZE="${PAGE_SIZE:-4096}" \
    "${PYTHON:-python3}" figures/lifetime_heatmap.py
