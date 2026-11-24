"""Recalibração de P(mnv) por regressão logística de 2 parâmetros no logit, ajustada deixando um satélite de fora
(LOSO). Avalia log-loss ponderada (pesos 1-p1 em observações sem manobra, p1 em observações com manobra) antes e depois.
Saída: e14_recal.json"""
import json, glob, numpy as np
from scipy.optimize import minimize

def lg(p):
    p = np.clip(np.asarray(p, float), 1e-9, 1 - 1e-9); return np.log(p / (1 - p))

def fit(x, y, w):
    def nll(t):
        z = t[0] + t[1] * x; return np.sum(w * (np.logaddexp(0, z) - y * z))
    return minimize(nll, [0.0, 1.0], method="Nelder-Mead", options=dict(xatol=1e-6, fatol=1e-9, maxiter=4000)).x

def logloss(z, y, w):
    return float(np.sum(w * (np.logaddexp(0, z) - y * z)) / np.sum(w))

def evaluate(x, y, w, sat):
    zr = x.copy()
    for s in np.unique(sat):
        tr = sat != s; a, b = fit(x[tr], y[tr], w[tr]); zr[~tr] = a + b * x[~tr]
    return dict(raw=round(logloss(x, y, w), 4), recal=round(logloss(zr, y, w), 4),
                base=round(logloss(np.full_like(x, np.log(np.sum(w * y) / np.sum(w * (1 - y)))), y, w), 4))

out = {}
for f in sorted(glob.glob("e11_rows_v3_*_seed5_K41.json")) + sorted(glob.glob("e11_rows_v3_*_seed6_K41_mis*.json")):
    D = json.load(open(f)); R = D["rows"]
    if not D["meta"]["complete"]:
        continue
    p1 = np.mean([r.get("p1", np.nan) for r in R]) if "p1" in R[0] else D["meta"].get("p1", 0.027)
    res = {}
    for m in [k for k in ["learn", "lin_learn", "iso_lin"] if "pH1_" + k in R[0]["M"]]:
        x = np.r_[lg([r["Z"]["pH1_" + m] for r in R]), lg([r["M"]["pH1_" + m] for r in R])]
        y = np.r_[np.zeros(len(R)), np.ones(len(R))]; w = np.r_[np.full(len(R), 1 - p1), np.full(len(R), p1)]
        sat = np.r_[[r["sat"] for r in R], [r["sat"] for r in R]]
        res[m] = evaluate(x, y, w, sat)
    out["E5_" + f.replace("e11_rows_v3_", "").replace(".json", "")] = dict(p1=round(float(p1), 4), **res)
for seed in (7, 11, "pool"):
    for meas in ["full", "tracklet"]:
        fn = f"e12_rows_{meas}_s0.05_seed{seed}.json"
        try:
            R = json.load(open(fn))["rows"]
        except FileNotFoundError:
            continue
        p1 = 0.027; res = {}
        for m in ["learn", "lin_learn"]:
            x = lg([(r["M"] if r["man"] else r["Z"])["pH1_" + m] for r in R]); y = np.array([float(r["man"]) for r in R])
            w = np.where(y == 1, p1 / max(y.mean(), 1e-9), (1 - p1) / max(1 - y.mean(), 1e-9))
            sat = np.array([r["sat"] for r in R]); res[m] = evaluate(x, y, w, sat)
            big = np.array([r["man"] and r["s"] >= 0.05 or not r["man"] for r in R])
            res[m + "_big"] = evaluate(x[big], y[big], w[big], sat[big])
        out[f"E6_{meas}_seed{seed}"] = res
json.dump(out, open("e14_recal.json", "w")); print(json.dumps(out))
