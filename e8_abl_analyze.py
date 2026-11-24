import json, numpy as np
D = json.load(open("e8_ablrows_s0.5.json")); p1 = D["p1"]; rows = D["rows"]
V = ["GM", "learn", "flat", "maxwell_0.3", "maxwell_0.03", "maxwell_3"]
def auc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg); r = np.concatenate([pos, neg]).argsort().argsort() + 1
    return 1 - (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))
def pH1(F0, Fm): return np.clip(1 - (1 - p1) * np.exp(-np.asarray(F0) + np.asarray(Fm)), 1e-9, 1 - 1e-9)
out = {}
for v in V[1:]:
    a0 = pH1([r["Z"]["GM"] for r in rows], [r["Z"][v] for r in rows])
    a1 = pH1([r["M"]["GM"] for r in rows], [r["M"][v] for r in rows])
    big = np.array([r["s"] >= 0.05 for r in rows])
    ll = -((1 - p1) * np.mean(np.log(1 - a0)) + p1 * np.mean(np.log(a1)))
    out[v] = dict(logloss=round(float(ll), 4),
                  AUC_det_dv05=round(float(auc(a0, a1[big]) * -1 + 1), 3),
                  AUC_viz_dv05=round(float(auc([r["M"][v] for r in rows if r["s"] >= 0.05], [r["N"][v] for r in rows])), 3),
                  AUC_viz_sem=round(float(auc([r["Z"][v] for r in rows], [r["N"][v] for r in rows])), 3))
out["GM"] = dict(AUC_viz_dv05=round(float(auc([r["M"]["GM"] for r in rows if r["s"] >= 0.05], [r["N"]["GM"] for r in rows])), 3),
                 AUC_viz_sem=round(float(auc([r["Z"]["GM"] for r in rows], [r["N"]["GM"] for r in rows])), 3))
out["ref_logloss_const"] = round(float(-(p1 * np.log(p1) + (1 - p1) * np.log(1 - p1))), 4)
print(json.dumps(out, indent=1)); json.dump(out, open("e8_abl_calib.json", "w"), indent=1)
