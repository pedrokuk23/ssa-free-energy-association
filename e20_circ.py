"""[Auditoria 2026-11-26: cópia de e20_p1.py sobre as linhas e17K3 com a correção de ângulos.]
Exploratório pós-E6c (revisão): sensibilidade da meta-análise, decomposição da variância, efeito do deslocamento do
vizinho (reconstruído pelas sementes), calibração com o mapa de E6a/E6b, limiares de dv, dv estimado vs operador (QZSS),
empates e efeito do teto de 8 eventos. Lê e17K2_* (mesmos sorteios do E6c). Saída: e20_circ.json e c121.txt"""
import json, glob, numpy as np
from scipy import stats, optimize
from e12K_analyze import hksj, signflip, by_sat
G = ["meo", "qzss", "leo"]; MS = ["full", "tracklet"]; KD = 20; SEED = 17

def load(g, m):
    rows = []
    for f in sorted(glob.glob(f"e17K3_e17sel_{g}_{m}_s0.05_seed17_c*of*.json")):
        rows += [r for r in json.load(open(f))["rows"] if r["draws"]]
    return rows

def cind(d, m):  # acerto da associação conjunta num sorteio
    return float(d["M"][m] < d["Mn"][m]) + 0.5 * float(d["M"][m] == d["Mn"][m])

def ugap(eid, k, meas):  # deslocamento do vizinho (em sigma), reproduzindo a sequência de sorteios do e17_scoreK
    rng = np.random.default_rng(np.random.SeedSequence([SEED, int(eid), k, meas == "tracklet", 17]))
    rng.standard_normal(6); rng.standard_normal(6); return float(rng.uniform(1.5, 3.0))

E = {}
for g in G:
    for m in MS:
        for r in load(g, m):
            bad = {f["k"] for f in r.get("failed", [])}; ks = [k for k in range(KD) if k not in bad][:len(r["draws"])]
            e = E.setdefault((g, r["eid"]), dict(sat=r["sat"], g=g, s=r["s"], tau=r["tau"], dr={}))
            e["dr"][m] = [dict(k=k, u=ugap(r["eid"], k, m), d=d) for k, d in zip(ks, r["draws"])]
keys = [k for k in E if len(E[k]["dr"]) == 2]
sats = np.array([E[k]["sat"] for k in keys]); grp = np.array([E[k]["g"] for k in keys])
sv = np.array([E[k]["s"] for k in keys]); tau = np.array([E[k]["tau"] for k in keys])

def dev(a, b, sel_draw=None):
    out = []
    for k in keys:
        v = [cind(x["d"], a) - cind(x["d"], b) for m in MS for x in E[k]["dr"][m] if sel_draw is None or sel_draw(x)]
        out.append(np.mean(v) if v else np.nan)
    return np.array(out)

def base(d, sl=None):
    sl = np.isfinite(d) if sl is None else (sl & np.isfinite(d)); y, v, n, us = by_sat(d[sl], sats[sl]); o = hksj(y, v)
    o["signflip_p"] = signflip(y, 1 / np.maximum(v, 1e-12)); o["n_ev"] = int(sl.sum()); return o

def tau2_pm(y, v):
    k = len(y); f = lambda t: np.sum((y - np.sum(y / (v + t)) / np.sum(1 / (v + t))) ** 2 / (v + t)) - (k - 1)
    return 0.0 if f(0) <= 0 else optimize.brentq(f, 0, 10 * np.var(y) + 1)

def tau2_reml(y, v):
    t = max(tau2_pm(y, v), 1e-8)
    for _ in range(200):
        w = 1 / (v + t); mu = np.sum(w * y) / w.sum()
        t2 = max(0.0, np.sum(w ** 2 * ((y - mu) ** 2 - v)) / np.sum(w ** 2) + 1 / w.sum())
        if abs(t2 - t) < 1e-12: break
        t = t2
    return t

def re_ci(y, v, t2, trunc=False):
    w = 1 / (v + t2); mu = np.sum(w * y) / w.sum(); k = len(y); q = np.sum(w * (y - mu) ** 2) / (k - 1)
    se = np.sqrt((max(q, 1.0) if trunc else q) / w.sum()); tq = stats.t.ppf(0.975, k - 1)
    return dict(mu=round(float(mu), 4), ci=[round(float(mu - tq * se), 4), round(float(mu + tq * se), 4)], p_one=round(float(stats.t.sf(mu / se, k - 1)), 4), tau=round(float(np.sqrt(t2)), 4))

def sens(d):
    o = {"pooled_DL_HKSJ": base(d)}; y, v, n, us = by_sat(d, sats)
    o["pooled_HKSJ_trunc"] = re_ci(y, v, tau2_pm(y, v) * 0 + max(0, hksj(y, v)["tau"]) ** 2, trunc=True)
    o["pooled_PM"] = re_ci(y, v, tau2_pm(y, v)); o["pooled_REML"] = re_ci(y, v, tau2_reml(y, v))
    s2p = v * n  # variância agrupada
    vu = np.array([np.var(d[sats == s], ddof=1) / np.sum(sats == s) if np.sum(sats == s) > 1 else s2p[0] / 1 for s in us])
    vu = np.maximum(vu, 1e-6); o["unpooled_PM"] = re_ci(y, vu, tau2_pm(y, vu)); o["unpooled_PM_trunc"] = re_ci(y, vu, tau2_pm(y, vu), True)
    o["sat_mean_t"] = dict(mu=round(float(y.mean()), 4), p_one=round(float(stats.ttest_1samp(y, 0, alternative="greater").pvalue), 4))
    N = len(d); mu = d.mean(); a = 1 / N; meat = 0.0  # CR2, média por evento agrupada por satélite
    for s in us:
        eg = d[sats == s] - mu; ng = len(eg); adj = 1 / np.sqrt(1 - a * ng) - 1
        meat += (eg.sum() * (1 + adj)) ** 2
    se = np.sqrt(meat) / N; k = len(us)
    o["event_CR2"] = dict(mu=round(float(mu), 4), ci=[round(float(mu - stats.t.ppf(0.975, k - 1) * se), 4), round(float(mu + stats.t.ppf(0.975, k - 1) * se), 4)],
                          p_one=round(float(stats.t.sf(mu / se, k - 1)), 4))
    o["leave_out"] = {g: base(d, grp != g) for g in G}
    return o

res = {"n_events": len(keys), "n_sats": len(set(sats))}
d = dev("learn", "CTRLu"); res["sens_learn_CTRLu"] = sens(d)
res["sens_learn_CTRLp"] = sens(dev("learn", "CTRLp")); res["sens_learn_IMM"] = sens(dev("learn", "IMM"))
# decomposição da variância (diferença por sorteio, média das duas UCTs no mesmo índice k)
dk = {k: np.array([np.mean([cind(E[k]["dr"][m][i]["d"], "learn") - cind(E[k]["dr"][m][i]["d"], "CTRLu") for m in MS])
                   for i in range(min(len(E[k]["dr"][m]) for m in MS))]) for k in keys}
within = np.mean([np.var(x, ddof=1) for x in dk.values()]); ev_mean = np.array([dk[k].mean() for k in keys])
satm = np.array([ev_mean[sats == s].mean() for s in sorted(set(sats))])
bw = np.mean([np.var(ev_mean[sats == s], ddof=1) for s in set(sats) if np.sum(sats == s) > 1])
res["variance"] = dict(draw_within_event=round(float(within), 4), event_within_sat=round(float(bw), 5), between_sat_means=round(float(np.var(satm, ddof=1)), 5),
                       implied_event_component=round(float(bw - within / 40), 5))
# deslocamento do vizinho
bins = [(1.5, 2.0), (2.0, 2.5), (2.5, 3.0)]
res["offset"] = {}
for lo, hi in bins:
    f = lambda x, lo=lo, hi=hi: lo <= x["u"] < hi
    rates = {m: round(float(np.nanmean([np.mean([cind(x["d"], m) for mm in MS for x in E[k]["dr"][mm] if f(x)] or [np.nan]) for k in keys])), 4)
             for m in ["learn", "CTRLu", "CTRLp", "IMM", "lin_learn"]}
    dd = dev("learn", "CTRLu", f); o = base(dd); res["offset"][f"{lo}-{hi}"] = dict(rates=rates, learn_CTRLu=dict(mu=o["mu"], ci=o["ci"], p_one=o["p_one"], n_ev=o["n_ev"]))
# empates
res["ties"] = {m: round(float(np.mean([d_["M"][m] == d_["Mn"][m] for k in keys for mm in MS for d_ in [x["d"] for x in E[k]["dr"][mm]]])), 4)
               for m in ["learn", "CTRLu", "CTRL", "CTRLp", "IMM", "GM", "UKF"]}
# limiares de dv e tipo
res["dv"] = {f">={t}": {kk: o[kk] for kk in ("mu", "ci", "p_one", "n_ev", "k")} for t in (0.05, 0.10, 0.20, 0.30) for o in [base(d, sv >= t)]}
res["type"] = {t: {kk: o[kk] for kk in ("mu", "ci", "p_one", "n_ev", "k")} for t in ("in-track", "cross-track") if np.sum(tau == t) > 3 for o in [base(d, tau == t)]}
# calibração: mapa logístico ajustado em E6a/E6b (frozen: refeito deterministicamente), avaliado no E6c
def lg(p): p = np.clip(np.asarray(p, float), 1e-9, 1 - 1e-9); return np.log(p / (1 - p))
def ll(z, y, w): return float(np.sum(w * (np.logaddexp(0, z) - y * z)) / np.sum(w))
def fit(x, y, w):
    return optimize.minimize(lambda t: np.sum(w * (np.logaddexp(0, t[0] + t[1] * x) - y * (t[0] + t[1] * x))), [0, 1], method="Nelder-Mead",
                             options=dict(xatol=1e-7, fatol=1e-10, maxiter=4000)).x
P1 = 0.027; res["calib"] = {}
for m in MS:
    Rr = []
    for sd in (7, 11):
        try: Rr += json.load(open(f"e12_rows_{m}_s0.05_seed{sd}.json"))["rows"]
        except FileNotFoundError: pass
    for meth in ["learn", "lin_learn"]:
        out = {}
        if Rr:
            x = lg([(r["M"] if r["man"] else r["Z"])["pH1_" + meth] for r in Rr]); y = np.array([float(r["man"]) for r in Rr])
            w = np.where(y == 1, P1 / y.mean(), (1 - P1) / (1 - y.mean())); a, b = fit(x, y, w); out["map"] = [round(float(a), 3), round(float(b), 3)]
        else:
            a, b = 0.0, 1.0; out["map"] = "missing"
        for g in G + ["all"]:
            ks = [k for k in keys if g == "all" or E[k]["g"] == g]
            if any("Dn" not in x["d"] for k in ks for x in E[k]["dr"][m]):  # e17K3 não guarda a UCT do vizinho
                out[g] = "not available (no neighbour UCT in e17K3 rows)"; continue
            xp = lg([x["d"]["M"]["pH1_" + meth] for k in ks for x in E[k]["dr"][m]]); xn = lg([x["d"]["Dn"]["pH1_" + meth] for k in ks for x in E[k]["dr"][m]])
            x = np.r_[xp, xn]; y = np.r_[np.ones(len(xp)), np.zeros(len(xn))]; w = np.r_[np.full(len(xp), P1 / len(xp)), np.full(len(xn), (1 - P1) / len(xn))]
            out[g] = dict(raw=round(ll(x, y, w), 4), recal=round(ll(a + b * x, y, w), 4), base=round(ll(np.full_like(x, np.log(P1 / (1 - P1))), y, w), 4),
                          auc=round(float(stats.mannwhitneyu(xp, xn).statistic / (len(xp) * len(xn))), 4))
        res["calib"][f"{m}_{meth}"] = out
# dv estimado vs operador (QZSS) e teto de 8 eventos
def evs(fs, bad=()):
    ev = []
    for f in fs:
        try: ev += [e for e in json.load(open(f))["events"] if not (f == "e17_events_qzss.json" and e["sat"] in bad)]
        except FileNotFoundError: pass
    return ev
q = [e for e in evs(["e17_events_qzss.json", "e17_events_qzss2.json"], {49336, 62876}) if e["man"] and e.get("s_label") and e.get("est")]
if q:
    lab = np.array([e["s_label"] for e in q]); est = np.array([np.hypot(e["est"]["dv_it"], e["est"]["dv_ct"]) for e in q]); r = np.abs(np.log(est / lab))
    res["qzss_dv"] = dict(n=len(q), median_factor=round(float(np.exp(np.median(r))), 3), p90_factor=round(float(np.exp(np.quantile(r, 0.9))), 3),
                          within_1p5=round(float(np.mean(r < np.log(1.5))), 3), spearman=round(float(stats.spearmanr(est, lab)[0]), 3),
                          same_side_5cm=round(float(np.mean((est >= 0.05) == (lab >= 0.05))), 3))
cap = {}
for g, fs in {"meo": ["e17_events_gnss.json"], "qzss": ["e17_events_qzss.json", "e17_events_qzss2.json"], "leo": ["e17_events_leo.json"]}.items():
    el = [e for e in evs(fs, {49336, 62876}) if e["man"] and e["s"] >= 0.05]; sel = json.load(open(f"e17sel_{g}.json"))["events"]
    cap[g] = dict(eligible=len(el), selected=len(sel))
res["cap"] = cap
json.dump(res, open("e20_circ.json", "w"), indent=1)
L = []
def P(*a): L.append(" ".join(str(x) for x in a))
for nm in ["sens_learn_CTRLu", "sens_learn_CTRLp", "sens_learn_IMM"]:
    P("==", nm)
    for kk, o in res[nm].items():
        if kk == "leave_out":
            for g, oo in o.items(): P(" -", g, oo["mu"], oo["ci"], oo["p_one"], "k", oo["k"])
        else: P(kk, o.get("mu"), o.get("ci"), o.get("p_one"), "tau", o.get("tau"))
P("== variance", res["variance"])
for b, o in res["offset"].items(): P("== off", b, o["rates"], o["learn_CTRLu"])
P("== ties", res["ties"])
for t, o in res["dv"].items(): P("== dv", t, o)
for t, o in res["type"].items(): P("== type", t, o)
for c, o in res["calib"].items():
    P("== calib", c, "map", o["map"])
    for g in G + ["all"]: P(" ", g, o[g])
P("== qzss_dv", res.get("qzss_dv")); P("== cap", res["cap"])
open("c121_circ.txt", "w").write("\n".join(L) + "\n"); print("\n".join(L))
