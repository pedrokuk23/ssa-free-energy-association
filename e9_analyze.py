import json, sys, numpy as np

def auc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    r = np.concatenate([pos, neg]).argsort().argsort() + 1
    return 1 - (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))

def tpr(pos, neg, f=0.01):
    return float(np.mean(np.asarray(pos) < np.quantile(neg, f)))

def boot(fn, rows, B=300, seed=0):
    """IC 95% por bootstrap sobre objetos (linhas)."""
    rng = np.random.default_rng(seed); n = len(rows); v = []
    for _ in range(B):
        idx = rng.integers(0, n, n); v.append(fn([rows[i] for i in idx]))
    return [round(float(np.percentile(v, 2.5)), 3), round(float(np.percentile(v, 97.5)), 3)]

M = ["UKF", "GM", "CTRL", "iso_lin", "learn", "flat", "maxwell_0.03", "maxwell_0.3", "maxwell_3"]
big = lambda r: r["s"] >= 0.05

def metrics(rows):
    out = {}
    for m in M:
        F = [r["F"][m] for r in rows]
        out[m] = {
            "acc1_big": tpr([r["M"][m] for r in rows if big(r)], F),
            "acc1_cross": tpr([r["M"][m] for r in rows if r["tau"] == "cross-track"], F),
            "auc_nb_big": auc([r["M"][m] for r in rows if big(r)], [r["N"][m] for r in rows]),
            "auc_nb_nomnv": auc([r["Z"][m] for r in rows], [r["N"][m] for r in rows]),
        }
        if m in ["GM", "CTRL"]:
            out[m]["det_big"] = auc([r["Z"][m] for r in rows], [r["M"][m] for r in rows if big(r)])
        if m not in ["UKF", "GM", "CTRL"]:
            k = "pH1_" + m
            a0 = np.clip([r["Z"][k] for r in rows], 1e-9, 1 - 1e-9); a1 = np.clip([r["M"][k] for r in rows], 1e-9, 1 - 1e-9)
            p1 = np.mean([r["p1"] for r in rows])
            out[m]["logloss"] = float(-((1 - p1) * np.mean(np.log(1 - a0)) + p1 * np.mean(np.log(a1))))
            out[m]["det_big"] = auc(a0, [r["M"][k] for r in rows if big(r)])
    return out

if __name__ == "__main__":
    res = {}
    for f in sys.argv[1:]:
        rows = json.load(open(f)); key = f.split("rows_")[1].split("_seed")[0]
        base = metrics(rows)
        p1 = np.mean([r["p1"] for r in rows])
        R = {"n": len(rows), "n_big": sum(big(r) for r in rows), "n_cross": sum(r["tau"] == "cross-track" for r in rows),
             "base_rate_logloss": float(-(p1 * np.log(p1) + (1 - p1) * np.log(1 - p1))), "metrics": base}
        # IC para as métricas-chave
        ci = {}
        for m in ["GM", "CTRL", "iso_lin", "learn", "flat"]:
            ci[m] = {"acc1_big": boot(lambda rr: tpr([r["M"][m] for r in rr if big(r)], [r["F"][m] for r in rr]), rows),
                     "auc_nb_big": boot(lambda rr: auc([r["M"][m] for r in rr if big(r)], [r["N"][m] for r in rr]), rows)}
        for m in ["learn", "flat", "iso_lin", "maxwell_0.3", "maxwell_3", "maxwell_0.03"]:
            def ll(rr, m=m):
                a0 = np.clip([r["Z"]["pH1_" + m] for r in rr], 1e-9, 1 - 1e-9); a1 = np.clip([r["M"]["pH1_" + m] for r in rr], 1e-9, 1 - 1e-9)
                return -((1 - p1) * np.mean(np.log(1 - a0)) + p1 * np.mean(np.log(a1)))
            ci.setdefault(m, {})["logloss"] = boot(ll, rows)
        R["ci95"] = ci
        # caracterização (priori aprendida)
        sel = [r for r in rows if big(r) and r["M"]["pH1_learn"] > 0.5]
        names = ["radial", "in-track", "cross-track"]
        acc = np.mean([names[int(np.argmax(r["M"]["p_dir"]))] == r["tau"] for r in sel]) if sel else np.nan
        err = [abs(np.log10(r["M"]["dv_hat"] / r["s"])) for r in sel]
        R["characterization"] = dict(n=len(sel), type_acc=float(acc), median_factor=float(10 ** np.median(err)) if err else None)
        # por satélite (fold): aceitação @1% para |Δv|≥5 cm/s
        per = {}
        for s in sorted(set(r["sat"] for r in rows)):
            P = [r for r in rows if r["sat"] == s and big(r)]
            if len(P) >= 5:
                per[s] = {m: round(tpr([r["M"][m] for r in P], [r["F"][m] for r in rows]), 2) for m in ["GM", "CTRL", "learn"]} | {"n": len(P)}
        R["per_satellite"] = per
        res[key] = R
    json.dump(res, open("e9_summary.json", "w"), indent=1, default=float)
    for k, R in res.items():
        print("=====", k, "n", R["n"], "n_big", R["n_big"], "n_cross", R["n_cross"], "base logloss %.4f" % R["base_rate_logloss"])
        for m, v in R["metrics"].items():
            print("  %-13s" % m, "  ".join(f"{a}={b:.3f}" for a, b in v.items()))
        print("  IC95:", json.dumps(R["ci95"]))
        print("  caracterização:", R["characterization"])
        print("  por satélite:", R["per_satellite"])

def paired_diff(rows, a, b, fn, B=1000, seed=1):
    rng = np.random.default_rng(seed); n = len(rows); v = []
    for _ in range(B):
        idx = rng.integers(0, n, n); rr = [rows[i] for i in idx]; v.append(fn(rr, a) - fn(rr, b))
    v = np.array(v); return [round(float(np.percentile(v, 2.5)), 3), round(float(np.percentile(v, 97.5)), 3), round(float(np.mean(v <= 0)), 4)]
