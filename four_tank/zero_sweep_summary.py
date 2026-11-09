import json, numpy as np, collections
R = json.load(open("det_results/zero_sweep.json"))
G = collections.defaultdict(list)
for r in R:
    G[(r["kind"], r["scen"], r["xi"], r["par"])].append(r)
nseed = len(next(iter(G.values())))
def rate(rs, w): return sum(r["wins"][w]["flag"] for r in rs), len(rs)
print("decryption errors in any run:", sum(r["dec_errors"] for r in R))
print("\nA: attack from step 0 on the true state (0, xi), guess g*xi, one window; alarms out of", nseed, "seeds")
print("kind      xi    g     | win0 (attacked) | win1 | win2 | nz win0 | median Z win0 | max|x2| win0 | x2 at end of win0")
for (kind, scen, xi, g), rs in sorted(G.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2], -1 if kv[0][3] is None else -kv[0][3])):
    if scen != "A": continue
    a = [rate(rs, w) for w in range(3)]
    print(f"{kind:8s} {xi:4.2f} {('none' if g is None else f'{g:4.2f}'):>5s} | {a[0][0]}/{a[0][1]}            | {a[1][0]}/{a[1][1]}  | {a[2][0]}/{a[2][1]}  | "
          f"{int(np.median([r['wins'][0]['nz'] for r in rs])):6d}  | {np.median([r['wins'][0]['Z'] for r in rs]):12.1f}  | "
          f"{np.median([r['wins'][0]['max_x2'] for r in rs]):10.2f}  ")
print("\nB: attack from rest at step 192, ZD sequence of amplitude eps, 192 steps; window 1 attacked, 2 aftermath, 3 later")
print("kind      eps   | win0 (calm) | win1 | win2 | win3 | median nz win1 | max|y| win1-2 | max|x2| win1-2")
for (kind, scen, xi, e), rs in sorted(G.items(), key=lambda kv: (kv[0][0], kv[0][1], -1 if kv[0][3] is None else kv[0][3])):
    if scen != "B": continue
    a = [rate(rs, w) for w in range(4)]
    print(f"{kind:8s} {('none' if e is None else f'{e:4.2f}'):>5s} | {a[0][0]}/{a[0][1]}         | {a[1][0]}/{a[1][1]}  | {a[2][0]}/{a[2][1]}  | {a[3][0]}/{a[3][1]}  | "
          f"{int(np.median([r['wins'][1]['nz'] for r in rs])):8d}       | {max(max(r['wins'][1]['max_y'], r['wins'][2]['max_y']) for r in rs):9.2f}     | "
          f"{max(max(r['wins'][1]['max_x2'], r['wins'][2]['max_x2']) for r in rs):9.2f}")
