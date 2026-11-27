"""Análise do E6s (PREREG_E6s.md) e meta-análise por satélite dos 6 desfechos originais de E6a/E6b.
Saída: e12K_summary.json, e12K_compact.txt"""
import json, glob, itertools, numpy as np
from scipy import stats

METH = ["UKF", "GM", "CTRL", "CTRLu", "iso_lin", "lin_learn", "learn"]
QUAD = ("CTRL", "CTRLu")

def _pfmt(p):
    """p-value kept to 4 decimals, or to 3 significant digits below 1e-3 (audit revision 3, R08: never print 0.0000)."""
    p = float(p); return round(p, 4) if p >= 1e-3 else float(f"{p:.3g}")

def hksj(y, v):
    """Efeitos aleatórios (DerSimonian-Laird) com correção HKSJ. y, v: efeitos e variâncias por satélite."""
    y = np.asarray(y, float); v = np.maximum(np.asarray(v, float), 1e-12); k = len(y)
    w = 1 / v; mu0 = np.sum(w * y) / w.sum(); Q = np.sum(w * (y - mu0) ** 2)
    tau2 = max(0.0, (Q - (k - 1)) / (w.sum() - np.sum(w ** 2) / w.sum()))
    ws = 1 / (v + tau2); mu = np.sum(ws * y) / ws.sum()
    q = np.sum(ws * (y - mu) ** 2) / (k - 1); se = np.sqrt(max(q, 1e-30) / ws.sum())
    t = stats.t.ppf(0.975, k - 1)
    pi = stats.t.ppf(0.975, k - 2) * np.sqrt(tau2 + se ** 2) if k > 2 else np.nan
    return dict(k=k, mu=round(float(mu), 4), ci=[round(float(mu - t * se), 4), round(float(mu + t * se), 4)],
                p_one=_pfmt(stats.t.sf(mu / se, k - 1)), tau=round(float(np.sqrt(tau2)), 4),
                pred=[round(float(mu - pi), 4), round(float(mu + pi), 4)], I2=round(float(max(0, (Q - (k - 1)) / Q)) if Q > 0 else 0.0, 3))

def signflip(y, w):
    """Teste exato de troca de sinal no nível de satélite (média ponderada), unilateral."""
    y = np.asarray(y, float); w = np.asarray(w, float); obs = np.sum(w * y) / w.sum(); k = len(y)
    if k > 20:
        rng = np.random.default_rng(0); S = rng.choice([-1, 1], size=(200000, k))
    else:
        S = np.array(list(itertools.product([-1, 1], repeat=k)))
    null = (S * y) @ w / w.sum()
    b = int(np.sum(null >= obs - 1e-15))
    # exact enumeration: b/2^k; Monte Carlo (k > 20): (b+1)/(N+1), so the estimate is never 0 (audit revision 3, R08)
    return _pfmt(b / len(null) if k <= 20 else (b + 1) / (len(null) + 1))

def jsc(a, b, m):
    return np.logaddexp(0, (a - b) / (2.0 if m in QUAD else 1.0))

def load_K(meas):
    rows = []
    for f in sorted(glob.glob(f"e12K_{meas}_s0.05_seed13_c*of*.json")):
        D = json.load(open(f))
        if not D["meta"]["complete"]:
            raise SystemExit(f"incompleto: {f}")
        rows += D["rows"]
    return rows

def per_event(rows, thr):
    """Por evento: taxa de associação correta (UCT manobrada -> próprio track), taxa correta da UCT do vizinho,
    taxa de aceitação a 1% de falsa aceitação de campo."""
    out = {m: dict(corr=[], corrN=[], acc=[]) for m in METH}
    for r in rows:
        for m in METH:
            Ft = np.array([d["M"][m] for d in r["draws"]]); Fn = np.array([d["Mn"][m] for d in r["draws"]])
            Dt = np.array([d["D"][m] for d in r["draws"]]); Dn = np.array([d["Dn"][m] for d in r["draws"]])
            out[m]["corr"].append(np.mean(Ft < Fn) + 0.5 * np.mean(Ft == Fn))
            out[m]["corrN"].append(np.mean(Dn < Dt) + 0.5 * np.mean(Dn == Dt))
            out[m]["acc"].append(np.mean(Ft <= thr[m]))
    return {m: {k: np.array(v) for k, v in o.items()} for m, o in out.items()}

def by_sat(vals, sats, minn=1):
    """Médias por satélite; variância dentro do satélite agrupada (variância comum s2/n_i), porque estimativas por
    satélite com poucos eventos (ou diferenças todas nulas) são instáveis e zeram pesos."""
    us = [s for s in sorted(set(sats)) if np.sum(sats == s) >= minn]
    res = np.concatenate([vals[sats == s] - vals[sats == s].mean() for s in us]); N = len(res)
    s2 = float(np.sum(res ** 2) / max(N - len(us), 1))
    if s2 <= 0:
        s2 = float(np.var(vals, ddof=1)) if len(vals) > 1 else 1e-6
    y = np.array([vals[sats == s].mean() for s in us]); n = np.array([np.sum(sats == s) for s in us])
    return y, s2 / n, n, us

def analyze_K(meas):
    rows = load_K(meas)
    thr = {m: np.quantile([d["F"][m] for r in rows for d in r["draws"]], 0.01) for m in METH}
    E = per_event(rows, thr)
    sats = np.array([r["sat"] for r in rows]); s = np.array([r["s"] for r in rows]); study = np.array([r["study"] for r in rows])
    res = {"n_events": len(rows), "K": len(rows[0]["draws"]), "n_sat": len(set(sats)),
           "rates": {m: {k: round(float(np.mean(v)), 4) for k, v in E[m].items()} for m in METH}}
    def test(a, b, key, sel=None):
        d = E[a][key] - E[b][key]; sl = np.ones(len(d), bool) if sel is None else sel
        y, v, n, us = by_sat(d[sl], sats[sl])
        o = hksj(y, v); o["signflip_p"] = signflip(y, 1 / np.maximum(v, 1e-12)) if len(y) > 1 else np.nan
        o["mean_event"] = round(float(np.mean(d[sl])), 4); o["n"] = int(sl.sum())
        o["forest"] = {str(u): [round(float(yy), 4), int(nn)] for u, yy, nn in zip(us, y, n)}
        return o
    res["primary"] = test("learn", "CTRLu", "corr")
    res["secondary"] = {f"corr:learn-{b}": test("learn", b, "corr") for b in ["GM", "UKF", "lin_learn", "CTRL"]}
    res["secondary"].update({f"acc:learn-{b}": test("learn", b, "acc") for b in ["CTRLu", "GM", "UKF", "lin_learn"]})
    res["secondary"]["corrN:learn-CTRLu"] = test("learn", "CTRLu", "corrN")
    res["exploratory_ge30"] = {f"corr:learn-{b}": test("learn", b, "corr", s >= 0.30) for b in ["CTRLu", "GM", "UKF"]}
    res["by_study"] = {st: test("learn", "CTRLu", "corr", study == st) for st in ["a", "b"]}
    for o in res["by_study"].values():
        o.pop("forest")
    return res

# ---------- meta-análise por satélite dos 6 desfechos originais (rows de E6a e E6b) ----------
def analyze_orig():
    from e12_analyze import Arr
    from e11_analyze import auc, tpr
    out = {}
    for seed in (7, 11):
        for meas in ["full", "tracklet"]:
            A = Arr(json.load(open(f"e12_rows_{meas}_s0.05_seed{seed}.json"))["rows"])
            big = A.man & (A.s >= 0.05); o = {}
            for key in ["acc_big", "det_big", "aucJ_big"]:
                y, v, n, us = [], [], [], []
                rng = np.random.default_rng(1)
                for sat in sorted(set(A.sat)):
                    ib = np.where(big & (A.sat == sat))[0]; ic = np.where(~A.man & (A.sat == sat))[0]
                    if len(ib) < 2:
                        continue
                    def diff(ibb, icc):
                        f = {}
                        for m in ("learn", "CTRLu"):
                            if key == "acc_big":
                                f[m] = tpr(A.obs[m][ibb], A.F[m])
                            elif key == "det_big":
                                st = A.lg[m] if m == "learn" else A.obs[m]
                                f[m] = auc(st[icc], st[ibb])
                            else:
                                f[m] = auc(A.jsc(A.obs[m], A.obsn[m], m)[ibb], A.jsc(A.D[m], A.Dn[m], m))
                        return f["learn"] - f["CTRLu"]
                    y.append(diff(ib, ic))
                    bs = [diff(rng.choice(ib, len(ib)), rng.choice(ic, len(ic))) for _ in range(200)]
                    v.append(np.nanvar(bs)); n.append(len(ib)); us.append(int(sat))
                y, v = np.array(y), np.array(v)
                pos = v[v > 0]; v = np.maximum(v, np.median(pos) if len(pos) else 1e-6)
                r = hksj(y, v); r["signflip_p"] = signflip(y, 1 / np.maximum(v, 1e-6))
                r["forest"] = {str(u): [round(float(a), 4), int(b)] for u, a, b in zip(us, y, n)}
                o[key] = r
            out[f"{meas}_seed{seed}"] = o
    return out

if __name__ == "__main__":
    out = {"orig_meta": analyze_orig()}
    for meas in ["full", "tracklet"]:
        try:
            out[meas] = analyze_K(meas)
        except SystemExit as e:
            out[meas] = str(e)
    pk = [out[m]["primary"]["p_one"] for m in ["full", "tracklet"] if isinstance(out[m], dict)]
    if len(pk) == 2:
        o = np.argsort(pk); adj = np.empty(2); adj[o[0]] = min(1, 2 * pk[o[0]]); adj[o[1]] = max(adj[o[0]], pk[o[1]])
        out["holm_primary"] = {"full": round(float(adj[0]), 4), "tracklet": round(float(adj[1]), 4)}
    js = json.dumps(out, separators=(",", ":")); open("e12K_summary.json", "w").write(js)
    open("e12K_compact.txt", "w").write("\n".join(js[i:i + 3000] for i in range(0, len(js), 3000)) + "\n")
    print("ok")
