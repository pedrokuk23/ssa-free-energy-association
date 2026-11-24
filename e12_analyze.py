"""Análise pré-registrada do E6 (PREREG_E6.md). Saída: e12_summary.json e e12_compact.txt."""
import json, glob, os, numpy as np
SEED = os.environ.get("SEED", "7")
from e11_analyze import auc, tpr, holm

METH = ["UKF", "GM", "CTRL", "CTRLu", "iso_lin", "lin_learn", "learn"]
PROB = ["iso_lin", "lin_learn", "learn"]
QUAD = ("CTRL", "CTRLu")
B = 2000

class Arr:
    def __init__(self, rows):
        self.r = rows; n = len(rows)
        self.sat = np.array([r["sat"] for r in rows]); self.man = np.array([r["man"] for r in rows])
        self.s = np.array([r["s"] for r in rows]); self.tau = np.array([r["tau"] for r in rows])
        g = lambda c, m: np.array([r[c][m] if c in r else np.nan for r in rows], float)
        self.obs = {m: np.where(self.man, g("M", m), g("Z", m)) for m in METH}
        self.obsn = {m: np.where(self.man, g("Mn", m), g("Zn", m)) for m in METH}
        self.D = {m: g("D", m) for m in METH}; self.Dn = {m: g("Dn", m) for m in METH}; self.F = {m: g("F", m) for m in METH}
        ph = lambda m: np.clip(np.where(self.man, g("M", "pH1_" + m), g("Z", "pH1_" + m)), 1e-9, 1 - 1e-9)
        self.lg = {m: np.log(ph(m) / (1 - ph(m))) for m in PROB}

    def jsc(self, a, b, m):
        k = 2.0 if m in QUAD else 1.0
        return np.logaddexp(0, (a - b) / k)

    def metric(self, m, key, idx):
        man = self.man[idx]; big = man & (self.s[idx] >= 0.05)
        if key == "acc_big":
            return tpr(self.obs[m][idx][big], self.F[m][idx])
        if key == "acc_ctrl":
            return tpr(self.obs[m][idx][~man], self.F[m][idx])
        if key == "acc_cross":
            return tpr(self.obs[m][idx][man & (self.tau[idx] == "cross-track")], self.F[m][idx])
        if key == "det_big":
            st = self.lg[m][idx] if m in PROB else self.obs[m][idx]
            return auc(st[~man], st[big])
        if key == "aucJ_big":
            jm = self.jsc(self.obs[m][idx], self.obsn[m][idx], m)[big]
            jd = self.jsc(self.D[m][idx], self.Dn[m][idx], m)
            return auc(jm, jd)
        if key == "auc_D_big":
            return auc(self.obs[m][idx][big], self.D[m][idx])
        raise KeyError(key)

def cboot(A, fn, B=B, seed=0):
    rng = np.random.default_rng(seed); groups = [np.where(A.sat == s)[0] for s in np.unique(A.sat)]; v = []
    for _ in range(B):
        idx = np.concatenate([groups[i][rng.integers(0, len(groups[i]), len(groups[i]))] for i in rng.integers(0, len(groups), len(groups))])
        v.append(fn(idx))
    v = np.array(v, float)
    return [round(float(np.nanpercentile(v, 2.5)), 4), round(float(np.nanpercentile(v, 97.5)), 4),
            round(float(np.mean((v <= 0) | np.isnan(v))), 4)]

KEYS = ["acc_big", "acc_ctrl", "acc_cross", "det_big", "aucJ_big", "auc_D_big"]

if __name__ == "__main__":
    out = {}; prim = []
    for f in sorted(glob.glob(f"e12_rows_*_seed{SEED}.json")):
        D = json.load(open(f)); rows = D["rows"]; key = f.replace("e12_rows_", "").replace(".json", "")
        if not D["meta"]["complete"]:
            continue
        A = Arr(rows); idx = np.arange(len(rows))
        o = {"meta": D["meta"], "n_man": int(A.man.sum()), "n_big": int((A.man & (A.s >= 0.05)).sum()), "n_ctrl": int((~A.man).sum()),
             "by_sat": {str(s): [int(((A.sat == s) & A.man).sum()), int(((A.sat == s) & ~A.man).sum())] for s in np.unique(A.sat)},
             "metrics": {m: {k: round(float(A.metric(m, k, idx)), 4) for k in KEYS} for m in METH}}
        pr = {}
        for k in ["acc_big", "det_big", "aucJ_big"]:
            pr[f"{k}:learn-CTRLu"] = cboot(A, lambda i, k=k: A.metric("learn", k, i) - A.metric("CTRLu", k, i))
            prim.append((key, f"{k}:learn-CTRLu", pr[f"{k}:learn-CTRLu"][2]))
        o["primary"] = pr
        sec = {}
        for b in ["GM", "lin_learn", "UKF"]:
            for k in ["acc_big", "det_big", "aucJ_big", "acc_ctrl"]:
                sec[f"{k}:learn-{b}"] = cboot(A, lambda i, k=k, b=b: A.metric("learn", k, i) - A.metric(b, k, i), B=1000)
        o["secondary"] = sec
        o["ci"] = {f"{k}:{m}": cboot(A, lambda i, k=k, m=m: A.metric(m, k, i), B=1000)[:2]
                   for m in ["GM", "CTRLu", "lin_learn", "learn"] for k in ["acc_big", "acc_ctrl", "det_big", "aucJ_big"]}
        names = ["radial", "in-track", "cross-track"]
        sel = [r for r in rows if r["man"] and r["s"] >= 0.05 and r["M"]["pH1_learn"] > 0.5]
        if sel:
            s = np.array([r["s"] for r in sel]); dr = np.array([r["M"]["dv_ref"] for r in sel]); dh = np.array([r["M"]["dv_hat"] for r in sel])
            typ = np.array([r["tau"] for r in sel]); est = np.array([names[int(np.argmax(r["M"]["p_dir"]))] for r in sel])
            fac = lambda a: round(float(10 ** np.median(np.abs(np.log10(a / s)))), 3)
            o["char"] = {"n": len(sel), "type_acc": round(float(np.mean(typ == est)), 3),
                         "type_acc_by": {t: [int((typ == t).sum()), round(float(np.mean(est[typ == t] == t)), 3)] for t in set(typ)},
                         "factor_refined": fac(dr), "factor_bin": fac(dh),
                         "w20_ref": round(float(np.mean(np.abs(np.log(dr / s)) < np.log(1.2))), 3)}
        ctrl = [r for r in rows if not r["man"]]
        o["model_err"] = {str(sat): [round(float(np.median([r["model_err_pos"] for r in ctrl if r["sat"] == sat])), 1),
                                     round(float(np.median([r["model_err_vel"] for r in ctrl if r["sat"] == sat])), 4)]
                          for sat in sorted({r["sat"] for r in ctrl})}
        out[key] = o; print(key, "done", flush=True)
    if prim:
        adj, rej = holm([p for _, _, p in prim])
        out["holm"] = [[k, t, p, a, bool(r)] for (k, t, p), a, r in zip(prim, adj, rej)]
    js = json.dumps(out, separators=(",", ":")); json.dump(out, open(f"e12_summary_seed{SEED}.json", "w"), separators=(",", ":"))
    open(f"e12_compact_seed{SEED}.txt", "w").write("\n".join(js[i:i + 3000] for i in range(0, len(js), 3000)) + "\n")
    print("ok")
