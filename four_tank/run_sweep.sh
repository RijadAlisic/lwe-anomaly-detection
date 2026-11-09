#!/bin/sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
exec /home/rijad/Anomaly-Detector-under-Encryption/.venv/bin/python zero_sweep.py > zero_sweep.log 2>&1
