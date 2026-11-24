"""Exploratório pós-E6c (após o desbloqueio): (i) reprodutibilidade de e17K2 contra e17K; (ii) linhas de base CTRLp e
IMM na associação conjunta (mesma estatística do primário); (iii) aceitação a 1% com o limiar do campo original (F) e
do campo em escala do regime (F2), por regime e tipo de UCT. Saída: e17K2_summary.json"""
import json, glob, numpy as np
import e12K_analyze as A
A.METH = ["UKF", "GM", "CTRL", "CTRLu", "iso_lin", "lin_learn", "learn", "IMM", "CTRLp"]
from e12K_analyze import hksj, signflip, by_sat
from scipy import stats
METH = A.METH; GROUPS = ["meo", "qzss", "leo"]

def load(pfx, g, meas):
    rows = []
    for f in sorted(glob.glob(f"{pfx}_e17sel_{g}_{meas}_s0.05_seed17_c*of*.json")):
        D = json.load(open(f))
        if not D["meta"]["complete"]:
            raise SystemExit(f"incompleto: {f}")
        rows += [r for r in D["rows"] if len(r["draws"]) > 0]
    return rows

def per_event(rows, fld):
    out = {m: dict(corr=[], acc=[]) for m in METH}
    for m in METH:
        thr = np.quantile([d[fld][m] for r in rows for d in r["draws"]], 0.01)
        for r in rows:
            Ft = np.array([d["M"][m] for d in r["draws"]]); Fn = np.array([d["Mn"][m] for d in r["draws"]])
            out[m]["corr"].append(np.mean(Ft < Fn) + 0.5 * np.mean(Ft == Fn)); out[m]["acc"].append(np.mean(Ft <= thr))
    return {m: {k: np.array(v) for k, v in o.items()} for m, o in out.items()}

repro = []; E = {}
for g in GROUPS:
    for meas in ("full", "tracklet"):
        r2 = load("e17K2", g, meas); r1 = {r["eid"]: r for r in load("e17K", g, meas)}
        for r in r2:
            o = r1.get(r["eid"])
            if o and len(o["draws"]) == len(r["draws"]):
                repro.append(max(abs(a[c][m] - b[c][m]) / max(1.0, abs(b[c][m])) for a, b in zip(r["draws"], o["draws"])
                                 for c in ("M", "Mn", "F") for m in ("learn", "CTRLu", "GM")))
        pe = {f: per_event(r2, f) for f in ("F", "F2")}
        for i, r in enumerate(r2):
            k = (g, r["eid"]); E.setdefault(k, dict(sat=r["sat"], g=g, meas={}))
            E[k]["meas"][meas] = {f: {m: {q: float(pe[f][m][q][i]) for q in ("corr", "acc")} for m in METH} for f in pe}
keys = [k for k in E if len(E[k]["meas"]) == 2]
sats = np.array([E[k]["sat"] for k in keys]); grp = np.array([E[k]["g"] for k in keys])

def val(m, q, fld="F", meas=None):
    if meas: return np.array([E[k]["meas"][meas][fld][m][q] for k in keys])
    return np.array([np.mean([E[k]["meas"][x][fld][m][q] for x in ("full", "tracklet")]) for k in keys])

def test(a, b, q="corr", fld="F", sel=None, meas=None):
    d = val(a, q, fld, meas) - val(b, q, fld, meas); sl = np.ones(len(d), bool) if sel is None else sel
    y, v, n, us = by_sat(d[sl], sats[sl]); o = hksj(y, v) if len(y) > 2 else dict(k=len(y))
    o["signflip_p"] = signflip(y, 1 / np.maximum(v, 1e-12)) if len(y) > 1 else None
    o["mean_event"] = round(float(np.mean(d[sl])), 4)
    return o

res = dict(repro_max_rel_diff=float(max(repro)) if repro else None, n_repro=len(repro), n_events=len(keys))
res["rates_corr"] = {m: round(float(np.mean(val(m, "corr"))), 4) for m in METH}
res["rates_corr_meas"] = {mm: {m: round(float(np.mean(val(m, "corr", meas=mm))), 3) for m in METH} for mm in ("full", "tracklet")}
res["joint_vs"] = {b: test("learn", b) for b in ("CTRLp", "IMM", "CTRLu")}
res["joint_vs_meas"] = {mm: {b: test("learn", b, meas=mm) for b in ("CTRLp", "IMM")} for mm in ("full", "tracklet")}
res["acc"] = {f: {g: {mm: {m: round(float(np.mean(val(m, "acc", f, mm)[grp == g])), 3) for m in METH}
                      for mm in ("full", "tracklet")} for g in GROUPS} for f in ("F", "F2")}
res["acc_test_F2"] = {b: test("learn", b, q="acc", fld="F2") for b in ("CTRLu", "GM", "UKF", "IMM")}
res["acc_test_F2_meas"] = {mm: test("learn", "CTRLu", q="acc", fld="F2", meas=mm) for mm in ("full", "tracklet")}
json.dump(res, open("e17K2_summary.json", "w"), indent=1)
print("repro", res["repro_max_rel_diff"], res["n_repro"]); print("joint", {b: (o["mu"], o["ci"], o["signflip_p"]) for b, o in res["joint_vs"].items()})
