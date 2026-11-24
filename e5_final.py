import sys, json, numpy as np
import e5_tracklet as E
from numpy.polynomial.hermite_e import hermegauss
E.GH7x, E.GH7w = hermegauss(5); E.GH7w = E.GH7w / E.GH7w.sum()
sel = np.linspace(0, 63, 16).astype(int); E.MM = E.MM[sel]
N = int(sys.argv[1])
neg, _ = E.run2(False, 0.0, N)
res, allv = {}, {"neg": neg}
for dv in [0.0, 0.02, 0.05, 0.1, 0.2, 0.4]:
    pos, _ = E.run2(True, dv, N)
    allv[f"pos_{dv}"] = pos
    res[dv] = {k: round(float(E.auc(pos[:, i], neg[:, i])), 3) for i, k in enumerate(E.NAMES2)}
    print(dv, res[dv], flush=True)
json.dump(res, open("e5_summary.json", "w"), indent=1)
np.savez("e5_results.npz", **allv)
