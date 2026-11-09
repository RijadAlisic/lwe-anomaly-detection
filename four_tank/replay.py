"""Replay of the detector as if it ran in parallel with the plant.

The plant worker controls the plant at the true sample rate and never waits.  Window w (T samples) becomes available
at simulation time (w+1)*T*Ts, when its last residual ciphertext arrives.  Detector workers (default 1) run beside it:
whenever one is free it takes the NEWEST available window that nobody has started.  When the run ends (t_end) the
plant worker stops and the detectors work through the windows they skipped (oldest first).

The detection computation was done sequentially on one core and its time per window was measured (det_results/*.npz,
key `seconds`).  The compute time on the simulation clock is  seconds * time_scale  (1 = the measured wall-clock
seconds are simulation seconds, a larger value emulates a slower detector).

    sched = schedule(avail, seconds, t_end, workers=1, time_scale=1.0)   ->  start, finish, worker (per window)
"""
import numpy as np


def schedule(avail, seconds, t_end, workers=1, time_scale=1.0):
    avail = np.asarray(avail, float)
    dur = np.asarray(seconds, float) * time_scale
    nw = len(avail)
    start, finish, who = np.full(nw, np.nan), np.full(nw, np.nan), np.full(nw, -1)
    free = np.zeros(workers)                       # time at which each detector worker is free
    todo = set(range(nw))
    while todo:
        k = int(np.argmin(free)); t = free[k]
        ready = [w for w in todo if avail[w] <= t]
        if not ready:                              # nothing available yet: wait for the next window
            t = min(avail[w] for w in todo)
            ready = [w for w in todo if avail[w] <= t]
        w = max(ready) if t <= t_end else min(ready)    # newest first during the run, oldest first afterwards
        start[w], finish[w], who[w] = t, t + dur[w], k
        free[k] = t + dur[w]
        todo.remove(w)
    return dict(start=start, finish=finish, worker=who, avail=avail)
