"""E6c, exploratory (audit 2026-11): posterior weight of the dominant hypothesis term and agreement of CTRLp and Lin joint decisions,
on the e17K3 rescoring (full-state rows identical to the confirmatory runs; tracklet rows corrected for the angle defect).
Usage: python e28_dominant.py [package_dir]  (default: current directory)
F_lin = lin_learn = -log(soma dos termos ponderados pela priori); CTRLp = -log(maior termo) (mesmos termos; ver e11_numeric.py).
Peso do termo dominante = exp(F_lin - F_CTRLp) em (0,1]."""
import json, glob, sys, numpy as np
P = sys.argv[1] if len(sys.argv) > 1 else "."
out = {}
for f in sorted(glob.glob(P + "/e17K3_e17sel_*_s0.05_seed17_c*.json")):
    meas = "full" if "_full_" in f else "tracklet"
    for r in json.load(open(f))["rows"]:
        for d in r["draws"]:
            for key in d:
                v = d[key]
                if isinstance(v, dict) and "CTRLp" in v and "lin_learn" in v:
                    w = np.exp(min(0.0, v["lin_learn"] - v["CTRLp"]))
                    out.setdefault((meas, key), []).append(w)
for k, w in sorted(out.items()):
    w = np.array(w)
    print(k, "n=", len(w), " peso dominante: mediana=%.3f  q10=%.3f  q25=%.3f  fracao>0.9=%.2f  fracao>0.5=%.2f" %
          (np.median(w), np.quantile(w, .1), np.quantile(w, .25), np.mean(w > .9), np.mean(w > .5)))

# diferenca, por sorteio, do deslocamento log(soma/max) entre a trilha-alvo (M) e a trilha do vizinho (Mn)
for meas in ("full", "tracklet"):
    dd = []; agree = []
    for f in sorted(glob.glob(P + f"/e17K3_e17sel_*_{meas}_s0.05_seed17_c*.json")):
        for r in json.load(open(f))["rows"]:
            for d in r["draws"]:
                a = d["M"]["CTRLp"] - d["M"]["lin_learn"]; b = d["Mn"]["CTRLp"] - d["Mn"]["lin_learn"]
                dd.append(a - b)
                agree.append((d["M"]["CTRLp"] < d["Mn"]["CTRLp"]) == (d["M"]["lin_learn"] < d["Mn"]["lin_learn"]))
    dd = np.array(dd)
    print(meas, "offset_M - offset_Mn [nats]: mediana=%.3f  IQR=[%.3f, %.3f]  |.|>1 nat: %.3f ; decisao conjunta igual CTRLp x Lin: %.3f"
          % (np.median(dd), *np.quantile(dd, [.25, .75]), np.mean(abs(dd) > 1), np.mean(agree)))
