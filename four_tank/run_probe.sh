#!/bin/sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$(dirname "$0")"
PY=/home/rijad/Anomaly-Detector-under-Encryption/.venv/bin/python
$PY bkz_probe.py 400 8192 8 40 50 60 70 > probe_400.log 2>&1 &
$PY bkz_probe.py 800 16384 8 40 50 60 70 > probe_800.log 2>&1 &
$PY bkz_probe.py 1600 32768 8 40 50 60 70 > probe_1600.log 2>&1 &
wait
