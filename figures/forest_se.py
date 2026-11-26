"""Per-satellite standard errors for the forest plots (Fig. 3 and Fig. S1): the pooled within-satellite variance
s2/n_i used by the meta-analysis (by_sat in e12K_analyze.py). Needs the per-event files of the Zenodo deposit.
Run from the package root:  python figures/forest_se.py   (writes figures/forest_se.json)"""
import json, os, sys, runpy, contextlib, io
import numpy as np
sys.path.insert(0, os.getcwd())
from e12K_analyze import load_K, per_event, by_sat, METH
out = {}; _M0 = list(METH)
with contextlib.redirect_stdout(io.StringIO()):                 # e17K_analyze runs at import; its output file is unchanged
    # audit 2026-11-26: with the angle fix, the E6c forest uses e27_circ_analyze.py (e17K3 rows) when present
    G = runpy.run_path("e27_circ_analyze.py" if os.path.exists("e27_circ_analyze.py") else "e17K_analyze.py")
METH[:] = _M0                                                   # e27 extends the shared method list
d = G["val"]("learn", "corr") - G["val"]("CTRLu", "corr"); y, v, n, us = by_sat(d, G["sats"])
out["e6c"] = {str(u): [round(float(yy), 4), int(nn), round(float(np.sqrt(vv)), 4)] for u, yy, nn, vv in zip(us, y, n, v)}
for meas in ("full", "tracklet"):
    rows = load_K(meas)
    thr = {m: np.quantile([dd["F"][m] for r in rows for dd in r["draws"]], 0.01) for m in METH}
    E = per_event(rows, thr); sats = np.array([r["sat"] for r in rows])
    y, v, n, us = by_sat(E["learn"]["corr"] - E["CTRLu"]["corr"], sats)
    out[f"e6s_{meas}"] = {str(u): [round(float(yy), 4), int(nn), round(float(np.sqrt(vv)), 4)] for u, yy, nn, vv in zip(us, y, n, v)}
json.dump(out, open("figures/forest_se.json", "w"), indent=0); print("forest_se ok")
