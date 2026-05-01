#!/usr/bin/env bash
set -euo pipefail
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python train_graph.py --config configs/smoke.json --variant full
