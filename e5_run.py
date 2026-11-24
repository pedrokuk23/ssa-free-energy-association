import sys, json, numpy as np
from e5_tracklet import *
N = int(sys.argv[1])
_, chk = run2(True, 0.0, 20, check_mc=True)
d = chk[:, 1] - chk[:, 0]
print("checagem RB vs MC (F0, já descontada a constante):", np.round(np.percentile(d + np.log(2*np.pi*sth*sthd), [5, 50, 95]), 3))
neg, _ = run2(False, 0.0, N)
res, allv = {}, {"neg": neg}
for dv in [0.0, 0.02, 0.05, 0.1, 0.2, 0.4]:
    pos, _ = run2(True, dv, N)
    allv[f"pos_{dv}"] = pos
    res[dv] = {k: round(float(auc(pos[:, i], neg[:, i])), 3) for i, k in enumerate(NAMES2)}
    print(dv, res[dv], flush=True)
json.dump(res, open("e5_summary.json", "w"), indent=1)
np.savez("e5_results.npz", **allv)
