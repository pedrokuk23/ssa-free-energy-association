"""Controle de qualidade dos estados (emenda 2): |r| em [6.6e6, 8.0e6] m, |v| em [6.5e3, 8.0e3] m/s, estados finitos e
|a(T) - a(t0)| < 50 km (semieixo maior osculador). Uso: python e12_qc.py IN OUT"""
import json, sys, numpy as np
MU = 3.986004418e14
D = json.load(open(sys.argv[1])); keep = []; bad = {}
def a(x):
    r = np.linalg.norm(x[:3]); v = np.linalg.norm(x[3:]); return 1.0 / (2.0 / r - v * v / MU)
for e in D["events"]:
    ok = True
    for k in ("x0", "yT"):
        x = np.array(e[k], float)
        if not np.all(np.isfinite(x)) or not (6.6e6 < np.linalg.norm(x[:3]) < 8.0e6) or not (6.5e3 < np.linalg.norm(x[3:]) < 8.0e3):
            ok = False
    if ok and abs(a(np.array(e["yT"])) - a(np.array(e["x0"]))) > 5e4:
        ok = False
    if ok:
        keep.append(e)
    else:
        bad.setdefault(str(e["sat"]), [0, 0])[0 if e["man"] else 1] += 1
D["events"] = keep; D["qc_removed_man_ctrl"] = bad
json.dump(D, open(sys.argv[2], "w"))
print(len(keep), sum(e["man"] for e in keep), "removed", bad)
