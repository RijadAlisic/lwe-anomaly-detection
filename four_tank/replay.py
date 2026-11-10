"""Replay of the detector as if it ran in parallel with the plant.

The plant worker controls the plant at the true sample rate and never waits.  Window w (T samples) becomes available
at simulation time (w+1)*T*Ts, when its last residual ciphertext arrives.  Detector workers (default 1) run beside it:
whenever one is free it takes the NEWEST available window that nobody has started.  When the run ends (t_end) the
plant worker stops and the detectors work through the windows they skipped (oldest first).

The detection computation was done sequentially on one core and its time per window was measured (det_results/*.npz,
key `seconds`).  The compute time on the simulation clock is  seconds * time_scale  (1 = the measured wall-clock
seconds are simulation seconds, a larger value emulates a slower detector).

    sched = schedule(avail, seconds, t_end, workers=1, time_scale=1.0, drop=False)   ->  start, finish, worker (per window)

With drop=True the detector is a live one: when a worker frees up it takes the newest available window and every older
window that nobody started is missed (never checked); there is no catching up after the run ends.  With catchup=True the
missed windows stay marked as missed during the run, and the detector works through them (oldest first) after the run ends.
"""
import numpy as np


def schedule(avail, seconds, t_end, workers=1, time_scale=1.0, drop=False, catchup=False):
    avail = np.asarray(avail, float)
    dur = np.asarray(seconds, float) * time_scale
    nw = len(avail)
    start, finish, who = np.full(nw, np.nan), np.full(nw, np.nan), np.full(nw, -1)
    free = np.zeros(workers)                       # time at which each detector worker is free
    todo = set(range(nw))
    missed, missed_at = np.zeros(nw, bool), np.full(nw, np.nan)
    while todo:
        k = int(np.argmin(free)); t = free[k]
        ready = [w for w in todo if avail[w] <= t]
        if not ready:                              # nothing available yet: wait for the next window
            t = min(avail[w] for w in todo)
            ready = [w for w in todo if avail[w] <= t]
        w = max(ready) if (t <= t_end or (drop and not catchup)) else min(ready)    # newest first during the run, oldest first afterwards
        if drop:                                   # live detector: windows older than the one taken are never checked
            for o in [o for o in ready if o < w]:
                if not missed[o]:
                    missed[o], missed_at[o] = True, t
                if not catchup:
                    todo.remove(o)
        start[w], finish[w], who[w] = t, t + dur[w], k
        free[k] = t + dur[w]
        todo.remove(w)
    return dict(start=start, finish=finish, worker=who, avail=avail, missed=missed, missed_at=missed_at)
