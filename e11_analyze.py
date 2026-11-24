"""Análise pré-registrada do E5 v3 (PREREG_E5v3.md). Saída: e11_summary.json e e11_compact.txt (linhas de 3000)."""
import json, glob, numpy as np
from scipy.stats import rankdata

FAM = {26997: "Jason", 33105: "Jason", 41240: "Jason", 46984: "Jason", 41335: "S3", 43437: "S3",
       20436: "SPOT", 25260: "SPOT", 27421: "SPOT", 27386: "Envisat", 36508: "CryoSat2", 37781: "HY2A", 39086: "SARAL"}
METH = ["UKF", "GM", "CTRL", "CTRLu", "iso_lin", "lin_learn", "learn", "flat", "maxwell_0.03", "maxwell_0.3", "maxwell_3"]
PROB = ["iso_lin", "lin_learn", "learn", "flat", "maxwell_0.03", "maxwell_0.3", "maxwell_3"]
QUAD = ("CTRL", "CTRLu")
B = 2000

def auc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    r = rankdata(np.concatenate([pos, neg]))
    return 1 - (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))

def tpr(pos, neg, f=0.01):
    pos = np.asarray(pos)
    return float(np.mean(pos < np.quantile(neg, f))) if len(pos) else np.nan

class Arr:
    def __init__(self, rows):
        self.sat = np.array([r["sat"] for r in rows]); self.s = np.array([r["s"] for r in rows])
        self.tau = np.array([r["tau"] for r in rows]); self.p1 = np.array([r["p1"] for r in rows])
        self.sc = {c: {m: np.array([r[c][m] for r in rows], float) for m in METH} for c in ("Z", "M", "D", "F", "Mn", "Dn")}
        self.ph = {c: {m: np.clip(np.array([r[c]["pH1_" + m] for r in rows]), 1e-9, 1 - 1e-9) for m in PROB} for c in ("Z", "M")}
        self.J = {}
        for m in METH:
            k = 2.0 if m in QUAD else 1.0
            self.J[m] = {"M": np.logaddexp(0, (self.sc["M"][m] - self.sc["Mn"][m]) / k),
                         "D": np.logaddexp(0, (self.sc["D"][m] - self.sc["Dn"][m]) / k)}

    def metric(self, m, key, idx):
        sc = {c: v[m][idx] for c, v in self.sc.items()}; big = self.s[idx] >= 0.05
        if key == "acc_big":
            return tpr(sc["M"][big], sc["F"])
        if key == "acc_all":
            return tpr(sc["M"], sc["F"])
        if key == "acc_cross":
            return tpr(sc["M"][self.tau[idx] == "cross-track"], sc["F"])
        if key == "auc_D_big":
            return auc(sc["M"][big], sc["D"])
        if key == "auc_D_nomnv":
            return auc(sc["Z"], sc["D"])
        if key == "aucJ_big":
            return auc(self.J[m]["M"][idx][big], self.J[m]["D"][idx])
        if key == "aucJ_all":
            return auc(self.J[m]["M"][idx], self.J[m]["D"][idx])
        if key == "det_big":
            if m in PROB:
                a0 = self.ph["Z"][m][idx]; a1 = self.ph["M"][m][idx][big]
                return auc(np.log(a0 / (1 - a0)), np.log(a1 / (1 - a1)))
            return auc(sc["Z"], sc["M"][big])
        if key == "logloss":
            p1 = self.p1[idx].mean(); a0 = self.ph["Z"][m][idx]; a1 = self.ph["M"][m][idx]
            return float(-((1 - p1) * np.mean(np.log(1 - a0)) + p1 * np.mean(np.log(a1))))
        raise KeyError(key)

def cboot(A, fn, B=B, seed=0):
    rng = np.random.default_rng(seed)
    groups = [np.where(A.sat == s)[0] for s in np.unique(A.sat)]; v = []
    for _ in range(B):
        idx = np.concatenate([groups[i][rng.integers(0, len(groups[i]), len(groups[i]))] for i in rng.integers(0, len(groups), len(groups))])
        v.append(fn(idx))
    v = np.array(v, float)
    return [round(float(np.nanpercentile(v, 2.5)), 4), round(float(np.nanpercentile(v, 97.5)), 4), round(float(np.mean((v <= 0) | np.isnan(v))), 4)]

def dfn(A, a, b, key):
    return lambda idx: A.metric(a, key, idx) - A.metric(b, key, idx)

def holm(ps, alpha=0.05):
    order = np.argsort(ps); m = len(ps); adj = [0.0] * m; run = 0
    for i, k in enumerate(order):
        run = max(run, min(1, (m - i) * ps[k])); adj[k] = round(run, 4)
    return adj, [a <= alpha for a in adj]

def char(rows):
    names = ["radial", "in-track", "cross-track"]
    sel = [r for r in rows if r["s"] >= 0.05 and r["M"]["pH1_learn"] > 0.5]
    if not sel:
        return {}
    s = np.array([r["s"] for r in sel]); dh = np.array([r["M"]["dv_hat"] for r in sel]); dr = np.array([r["M"]["dv_ref"] for r in sel])
    typ = np.array([r["tau"] for r in sel]); est = np.array([names[int(np.argmax(r["M"]["p_dir"]))] for r in sel])
    edges = np.logspace(-3, 1.1, 9); cen = np.sqrt(edges[:-1] * edges[1:])
    snap = cen[np.clip(np.digitize(s, edges) - 1, 0, 7)]
    fac = lambda a, b: round(float(10 ** np.median(np.abs(np.log10(a / b)))), 3)
    return {"n": len(sel), "type_acc": round(float(np.mean(typ == est)), 3),
            "type_acc_by": {t: round(float(np.mean(est[typ == t] == t)), 3) for t in set(typ)},
            "factor_bin": fac(dh, s), "factor_refined": fac(dr, s), "factor_snap_true_bin": fac(snap, s),
            "frac_within_20pct_refined": round(float(np.mean(np.abs(np.log(dr / s)) < np.log(1.2))), 3),
            "frac_within_20pct_bin": round(float(np.mean(np.abs(np.log(dh / s)) < np.log(1.2))), 3)}

KEYS = ["acc_big", "acc_all", "acc_cross", "auc_D_big", "auc_D_nomnv", "aucJ_big", "aucJ_all", "det_big"]

if __name__ == "__main__":
    out = {}; prim = []
    for f in sorted(glob.glob("e11_rows_v3_*_seed5_K41.json")) + sorted(glob.glob("e11_rows_v3_*_seed6_K41_mis*.json")):
        D = json.load(open(f)); rows = D["rows"]; key = f.replace("e11_rows_", "").replace(".json", "")
        if not D["meta"]["complete"]:
            continue
        A = Arr(rows); idx = np.arange(len(rows)); p1 = A.p1.mean()
        o = {"meta": D["meta"], "n_big": int((A.s >= 0.05).sum()),
             "base_logloss": round(float(-(p1 * np.log(p1) + (1 - p1) * np.log(1 - p1))), 4),
             "metrics": {m: {k: round(float(A.metric(m, k, idx)), 4) for k in KEYS + (["logloss"] if m in PROB else [])} for m in METH}}
        meas = D["meta"]["meas"]
        if not D["meta"]["mismatch"]:
            pr = {}
            tests = [("aucJ_big", "CTRLu"), ("aucJ_big", "GM")] + ([("acc_big", "lin_learn"), ("acc_big", "CTRLu")] if meas == "tracklet" else [])
            for k, b in tests:
                pr[f"{k}:learn-{b}"] = cboot(A, dfn(A, "learn", b, k)); prim.append((key, f"{k}:learn-{b}", pr[f"{k}:learn-{b}"][2]))
            o["primary"] = pr
            sec = {}
            for b in ["UKF", "GM", "CTRL", "CTRLu", "iso_lin", "lin_learn"]:
                for k in ["acc_big", "acc_cross", "auc_D_big", "aucJ_big", "aucJ_all"]:
                    if f"{k}:learn-{b}" not in pr:
                        sec[f"{k}:learn-{b}"] = cboot(A, dfn(A, "learn", b, k), B=1000)
            for k in ["acc_big", "acc_cross"]:
                sec[f"{k}:maxwell_0.3-iso_lin"] = cboot(A, dfn(A, "maxwell_0.3", "iso_lin", k), B=1000)
            sec["logloss:base-learn"] = cboot(A, lambda i: o["base_logloss"] - A.metric("learn", "logloss", i), B=1000)
            o["secondary"] = sec
            ci = {}
            for m in ["UKF", "GM", "CTRL", "CTRLu", "lin_learn", "learn"]:
                for k in ["acc_big", "acc_cross", "aucJ_big"]:
                    ci[f"{k}:{m}"] = cboot(A, lambda i, m=m, k=k: A.metric(m, k, i), B=1000)[:2]
            o["ci"] = ci
            o["char"] = char(rows)
            edges = [0, 0.005, 0.05, 0.5, 100]
            o["acc_bins"] = {m: [[int(((A.s >= a) & (A.s < b)).sum()),
                                  round(tpr(A.sc["M"][m][(A.s >= a) & (A.s < b)], A.sc["F"][m]), 3)] for a, b in zip(edges[:-1], edges[1:])]
                             for m in ["GM", "CTRLu", "lin_learn", "learn"]}
        out[key] = o
        print(key, "done", flush=True)
    # descasamento: pareado 0.2 vs 0.4
    mis = {k: v for k, v in out.items() if v["meta"]["mismatch"]}
    out["mismatch_pair"] = {k: {m: {kk: v["metrics"][m][kk] for kk in ("acc_big", "logloss")} for m in ["learn", "lin_learn", "flat", "maxwell_0.03"]}
                            for k, v in mis.items()}
    if prim:
        adj, rej = holm([p for _, _, p in prim])
        out["holm"] = [[k, t, p, a, bool(r)] for (k, t, p), a, r in zip(prim, adj, rej)]
    json.dump(out, open("e11_summary.json", "w"), separators=(",", ":"))
    js = json.dumps(out, separators=(",", ":"))
    open("e11_compact.txt", "w").write("\n".join(js[i:i + 3000] for i in range(0, len(js), 3000)) + "\n")
    print("ok")
