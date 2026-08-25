#!/usr/bin/env bash
# (a) write overhead (KV / KV-reclaim / weight-reclaim) and (b) HBF lifetime,
# ideal vs typical-SLC reclaim, vs output length. Output:
#   plots/paper/write_lifetime/write_lifetime_total_v5__162layer/*.png
#
# Env: HBF_LAYERS (default 162), PYTHON (default python3).
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=.

HBF_LAYERS="${HBF_LAYERS:-162}" "${PYTHON:-python3}" figures/write_lifetime.py
