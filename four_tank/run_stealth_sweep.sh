#!/bin/sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$(dirname "$0")"
exec /home/rijad/Anomaly-Detector-under-Encryption/.venv/bin/python stealth_sweep.py > stealth_sweep.log 2>&1
