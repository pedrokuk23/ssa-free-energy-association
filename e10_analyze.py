"""Análise pré-registrada do E5 v2 (ver PREREG_E5v2.md).
Uso: python e10_analyze.py  (procura os arquivos e10_rows_v2_*_seed3_K41_J12.json, *_mis.json e kconv)
Saída: e10_summary.json (compacto)."""
import json, glob, numpy as np
from scipy.stats import rankdata

FAM = {26997: "Jason", 33105: "Jason", 41240: "Jason", 46984: "Jason", 41335: "S3", 43437: "S3",
       20436: "SPOT", 25260: "SPOT", 27421: "SPOT", 27386: "Envisat", 36508: "CryoSat2", 37781: "HY2A", 39086: "SARAL"}
METH = ["UKF", "GM", "CTRL", "CTRLu", "iso_lin", "lin_learn", "learn", "flat", "maxwell_0.03", "maxwell_0.3", "maxwell_3"]
PROB = ["iso_lin", "lin_learn", "learn", "flat", "maxwell_0.03", "maxwell_0.3", "maxwell_3"]
B = 2000

def auc(pos, neg):
    """P(pos < neg) com empates contados como 1/2 (menor escore = mesmo objeto)."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    r = rankdata(np.concatenate([pos, neg]))
    return 1 - (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))

def tpr(pos, neg, f=0.01):
    pos = np.asarray(pos)
    return float(np.mean(pos < np.quantile(neg, f))) if len(pos) else np.nan

def pH(r, c, m):
    return float(np.clip(r[c]["pH1_" + m], 1e-9, 1 - 1e-9))

def stats(rows, m):
    F = [r["F"][m] for r in rows]
    big = [r for r in rows if r["s"] >= 0.05]
    o = {"acc_big": tpr([r["M"][m] for r in big], F),
         "acc_all": tpr([r["M"][m] for r in rows], F),
         "acc_cross": tpr([r["M"][m] for r in rows if r["tau"] == "cross-track"], F),
         "acc_intrack_big": tpr([r["M"][m] for r in big if r["tau"] == "in-track"], F),
         "auc_D_big": auc([r["M"][m] for r in big], [r["D"][m] for r in rows]),
         "auc_N_big": auc([r["M"][m] for r in big], [r["N"][m] for r in rows]),
         "auc_D_nomnv": auc([r["Z"][m] for r in rows], [r["D"][m] for r in rows]),
         "auc_N_nomnv": auc([r["Z"][m] for r in rows], [r["N"][m] for r in rows]),
         "auc_D_within": float(np.mean([r["M"][m] < r["D"][m] for r in big])) if big else np.nan}
    if m in PROB:
        p1 = np.mean([r["p1"] for r in rows])
        a0 = np.array([pH(r, "Z", m) for r in rows]); a1 = np.array([pH(r, "M", m) for r in rows])
        o["logloss"] = float(-((1 - p1) * np.mean(np.log(1 - a0)) + p1 * np.mean(np.log(a1))))
        ab = np.array([pH(r, "M", m) for r in big])
        o["det_big"] = auc(np.log(a0 / (1 - a0)), np.log(ab / (1 - ab)))   # P(logit_Z < logit_M)
    elif m in ("GM", "CTRL", "CTRLu", "UKF"):
        o["det_big"] = auc([r["Z"][m] for r in rows], [r["M"][m] for r in big])
    return o

def acc_bins(rows, m):
    F = [r["F"][m] for r in rows]; edges = [0, 0.005, 0.05, 0.5, 100]; out = []
    for a, b in zip(edges[:-1], edges[1:]):
        P = [r["M"][m] for r in rows if a <= r["s"] < b]
        out.append([len(P), round(tpr(P, F), 3) if P else None])
    return out

def cluster_boot(rows, fn, B=B, seed=0):
    """bootstrap em dois estágios: satélites com reposição, depois linhas dentro de cada satélite."""
    rng = np.random.default_rng(seed)
    by = {}
    for r in rows:
        by.setdefault(r["sat"], []).append(r)
    sats = list(by); v = []
    for _ in range(B):
        rr = []
        for s in rng.choice(len(sats), len(sats)):
            g = by[sats[s]]; rr += [g[i] for i in rng.integers(0, len(g), len(g))]
        v.append(fn(rr))
    v = np.array(v, float)
    return [round(float(np.nanpercentile(v, 2.5)), 4), round(float(np.nanpercentile(v, 97.5)), 4),
            round(float(np.mean(v <= 0)), 4)]

def diff_fn(a, b, key):
    return lambda rr: stats(rr, a)[key] - stats(rr, b)[key]

def ll_diff(a, b):   # positivo = a pior (maior log-loss) que b
    return lambda rr: stats(rr, a)["logloss"] - stats(rr, b)["logloss"]


class Arr:
    """arrays pré-computados para bootstrap rápido por índices."""
    def __init__(self, rows):
        self.rows = rows
        self.sat = np.array([r["sat"] for r in rows]); self.s = np.array([r["s"] for r in rows])
        self.tau = np.array([r["tau"] for r in rows]); self.p1 = np.array([r["p1"] for r in rows])
        self.sc = {c: {m: np.array([r[c][m] for r in rows]) for m in METH} for c in "ZMNDF"}
        self.ph = {c: {m: np.clip(np.array([r[c]["pH1_" + m] for r in rows]), 1e-9, 1 - 1e-9) for m in PROB} for c in "ZM"}

    def metric(self, m, key, idx):
        sc = {c: v[m][idx] for c, v in self.sc.items()}; big = self.s[idx] >= 0.05
        if key == "acc_big":
            return tpr(sc["M"][big], sc["F"])
        if key == "acc_cross":
            return tpr(sc["M"][self.tau[idx] == "cross-track"], sc["F"])
        if key == "auc_D_big":
            return auc(sc["M"][big], sc["D"])
        if key == "auc_N_big":
            return auc(sc["M"][big], sc["N"])
        if key == "logloss":
            p1 = self.p1[idx].mean(); a0 = self.ph["Z"][m][idx]; a1 = self.ph["M"][m][idx]
            return float(-((1 - p1) * np.mean(np.log(1 - a0)) + p1 * np.mean(np.log(a1))))
        raise KeyError(key)

def cboot(A, fn, B=B, seed=0):
    rng = np.random.default_rng(seed)
    groups = [np.where(A.sat == s)[0] for s in np.unique(A.sat)]; v = []
    for _ in range(B):
        idx = np.concatenate([g[rng.integers(0, len(g), len(g))] for g in (groups[i] for i in rng.integers(0, len(groups), len(groups)))])
        v.append(fn(idx))
    v = np.array(v, float)
    return [round(float(np.nanpercentile(v, 2.5)), 4), round(float(np.nanpercentile(v, 97.5)), 4), round(float(np.mean(v <= 0)), 4)]

def dfn(A, a, b, key):
    return lambda idx: A.metric(a, key, idx) - A.metric(b, key, idx)

def holm(ps, alpha=0.05):
    order = np.argsort(ps); m = len(ps); rej = [False] * m; adj = [0.0] * m; run = 0
    for i, k in enumerate(order):
        run = max(run, min(1, (m - i) * ps[k])); adj[k] = round(run, 4)
    for k in range(m):
        rej[k] = adj[k] <= alpha
    return adj, rej

def char(rows):
    names = ["radial", "in-track", "cross-track"]
    sel = [r for r in rows if r["s"] >= 0.05 and r["M"]["pH1_learn"] > 0.5]
    if not sel:
        return {}
    s = np.array([r["s"] for r in sel]); dh = np.array([r["M"]["dv_hat"] for r in sel])
    typ = np.array([r["tau"] for r in sel]); est = np.array([names[int(np.argmax(r["M"]["p_dir"]))] for r in sel])
    edges = np.logspace(-3, 1.1, 9); cen = np.sqrt(edges[:-1] * edges[1:])
    snap = cen[np.clip(np.digitize(s, edges) - 1, 0, 7)]
    med = {t: float(np.median(s[typ == t])) for t in set(typ)}       # constante por tipo (oráculo de tipo)
    fac = lambda a, b: round(float(10 ** np.median(np.abs(np.log10(a / b)))), 3)
    return {"n": len(sel), "type_acc": round(float(np.mean(typ == est)), 3),
            "type_acc_by": {t: round(float(np.mean(est[typ == t] == t)), 3) for t in set(typ)},
            "factor": fac(dh, s), "factor_snap_true_bin": fac(snap, s),
            "factor_const_per_type": fac(np.array([med[t] for t in typ]), s),
            "frac_top_bin": round(float(np.mean(dh > cen[-1] * 0.99)), 3)}

def rnd(d):
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in d.items()}

if __name__ == "__main__":
    out = {}; prim = []
    for f in sorted(glob.glob("e10_rows_v2_*_seed3_K41_J12*.json")):
        D = json.load(open(f)); rows = D["rows"]; key = f.replace("e10_rows_", "").replace(".json", "")
        assert D["meta"]["complete"], f
        p1 = np.mean([r["p1"] for r in rows])
        o = {"meta": {k: D["meta"][k] for k in ("N", "s_vI", "seed", "K1", "J", "mismatch", "n_done")},
             "n_big": sum(r["s"] >= 0.05 for r in rows), "n_cross": sum(r["tau"] == "cross-track" for r in rows),
             "base_logloss": round(float(-(p1 * np.log(p1) + (1 - p1) * np.log(1 - p1))), 4), "p1": round(float(p1), 4),
             "metrics": {m: rnd(stats(rows, m)) for m in METH},
             "acc_bins": {m: acc_bins(rows, m) for m in ["GM", "CTRLu", "lin_learn", "learn"]}}
        A = Arr(rows)
        if not D["meta"]["mismatch"]:
            pr = {}
            for b in ["lin_learn", "CTRLu"]:
                for k in ["acc_big", "auc_D_big"]:
                    pr[f"{k}:learn-{b}"] = cboot(A, dfn(A, "learn", b, k))
                    prim.append((key, f"{k}:learn-{b}", pr[f"{k}:learn-{b}"][2]))
            o["primary"] = pr
            sec = {}
            for b in ["GM", "CTRL", "iso_lin", "UKF"]:
                for k in ["acc_big", "auc_D_big", "auc_N_big", "acc_cross", "logloss"][:4]:
                    sec[f"{k}:learn-{b}"] = cboot(A, dfn(A, "learn", b, k))
            for k in ["acc_big", "auc_D_big", "acc_cross"]:
                sec[f"{k}:maxwell_0.3-iso_lin"] = cboot(A, dfn(A, "maxwell_0.3", "iso_lin", k))
                sec[f"{k}:learn-lin_learn"] = pr.get(f"{k}:learn-lin_learn") or cboot(A, dfn(A, "learn", "lin_learn", k))
            for b in ["flat", "maxwell_0.03", "maxwell_0.3", "maxwell_3", "iso_lin", "lin_learn"]:
                sec[f"logloss:{b}-learn"] = cboot(A, dfn(A, b, "learn", "logloss"))
            sec["logloss:base-learn"] = cboot(A, lambda idx: o["base_logloss"] - A.metric("learn", "logloss", idx))
            o["secondary"] = sec
            fam = {}
            for fa in sorted(set(FAM.values())):
                rr = [r for r in rows if FAM.get(r["sat"]) == fa]
                if len(rr) >= 30:
                    fam[fa] = {"n": len(rr), "acc_big_learn-lin": round(stats(rr, "learn")["acc_big"] - stats(rr, "lin_learn")["acc_big"], 3),
                               "acc_big_learn-CTRLu": round(stats(rr, "learn")["acc_big"] - stats(rr, "CTRLu")["acc_big"], 3)}
            o["by_family"] = fam
            o["char"] = char(rows)
            names = ["radial", "in-track", "cross-track"]
            o["char_list"] = [[round(r["s"], 4), round(r["M"]["dv_hat"], 4), r["tau"][0], names[int(np.argmax(r["M"]["p_dir"]))][0]]
                              for r in rows if r["s"] >= 0.05 and r["M"]["pH1_learn"] > 0.5]
            ci = {}
            for m in ["UKF", "GM", "CTRL", "CTRLu", "iso_lin", "lin_learn", "learn"]:
                for k in ["acc_big", "acc_cross", "auc_D_big"]:
                    ci[f"{k}:{m}"] = cboot(A, lambda idx, m=m, k=k: A.metric(m, k, idx), B=1000)[:2]
            for m in PROB:
                ci[f"logloss:{m}"] = cboot(A, lambda idx, m=m: A.metric(m, "logloss", idx), B=1000)[:2]
                ci[f"auc_D_big:{m}"] = cboot(A, lambda idx, m=m: A.metric(m, "auc_D_big", idx), B=1000)[:2]
            o["ci"] = ci
        out[key] = o
        print(key, "done", flush=True)
    if prim:
        adj, rej = holm([p for _, _, p in prim])
        out["holm"] = [[k, t, p, a, bool(r)] for (k, t, p), a, r in zip(prim, adj, rej)]
    # convergência em K1 (pareado: mesma semente e mesmos objetos)
    kc = {}
    files = sorted(glob.glob("e10_rows_kconv_s0.5_seed4_K*_J12.json"), key=lambda f: int(f.split("_K")[1].split("_")[0]))
    R = {int(f.split("_K")[1].split("_")[0]): json.load(open(f))["rows"] for f in files}
    if 81 in R:
        ref = R[81]
        lo = lambda r, c: np.log(pH(r, c, "learn") / (1 - pH(r, c, "learn")))
        for K, rows in R.items():
            dl = [abs(lo(a, c) - lo(b, c)) for a, b in zip(rows, ref) for c in "ZMD"]
            kc[K] = {"n": len(rows), "median_abs_dlogit_vs81": round(float(np.median(dl)), 3),
                     "p90_abs_dlogit_vs81": round(float(np.percentile(dl, 90)), 3),
                     **{k: round(v, 3) for k, v in stats(rows, "learn").items() if k in ("acc_big", "auc_D_big", "logloss")}}
    out["kconv"] = kc
    json.dump(out, open("e10_summary.json", "w"), separators=(",", ":"))
    print("ok")
    js = json.dumps(out, separators=(",", ":"))
    open("e10_compact.txt", "w").write("\n".join(js[i:i + 3000] for i in range(0, len(js), 3000)) + "\n")
