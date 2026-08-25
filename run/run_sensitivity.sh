#!/usr/bin/env bash
# Lifetime sensitivity vs output length, with read-reclaim:
#   (a) NAND layer count   (b) HBF bandwidth   (c) KV vs weight reclaim
# Shared linear years axis with the 5-year limit. Output:
#   plots/paper/lifetime_triptych/lifetime_triptych.{png,pdf}
#
# Swept values are set in figures/lifetime_triptych.py (LAYERS, BANDWIDTHS);
# input length and batch are fixed there. Env: PYTHON (default python3).
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=.

"${PYTHON:-python3}" figures/lifetime_triptych.py
