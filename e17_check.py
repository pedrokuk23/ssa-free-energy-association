import json, sys, numpy as np, collections
import numprop as N
ev = json.load(open(sys.argv[1]))["events"]
M = [e for e in ev if e["man"]]; C = [e for e in ev if not e["man"]]
print("eventos", len(ev), "manobras", len(M), "controles", len(C), "satélites c/ manobra", len({e["sat"] for e in M}))
s = np.array([e["s"] for e in M]); print("dv quantis (m/s) 10/25/50/75/90:", np.percentile(s, [10, 25, 50, 75, 90]).round(3))
big = [e for e in M if e["s"] >= 0.05]; print(">=5 cm/s:", len(big), "satélites:", len({e["sat"] for e in big}))
print("por satélite (>=5cm/s):", sorted(collections.Counter(e["sat"] for e in big).values(), reverse=True))
print("tipos:", collections.Counter(e["tau"] for e in big))
lab = [(e["s_label"], np.hypot(e["est"]["dv_it"], e["est"]["dv_ct"])) for e in M if e.get("s_label")]
if lab:
    r = np.array([b / a for a, b in lab]); print("QZSS estimado/rotulo: n", len(r), "mediana", np.median(r).round(3), "fator |log| mediano", np.exp(np.median(np.abs(np.log(r)))).round(3))
CRAM = {"G": 0.02, "E": 0.015, "J": 0.012}
err = []; dvc = []
for e in C[:60]:
    N.set_forces(1, e["epoch_j2000"], CRAM[e["prn"][0]])
    x0 = np.array(e["x0"]); yT = np.array(e["yT"])
    y = N.integrate(x0[None], 0.0, 0.0, e["Tgap"])[0]
    err.append(np.linalg.norm(y[:3] - yT[:3])); dvc.append(np.hypot(e["est"]["dv_it"], e["est"]["dv_ct"]))
err = np.array(err); print("erro do modelo 2 dias (controles) km: mediana %.2f p90 %.2f max %.2f" % (np.median(err) / 1e3, np.percentile(err, 90) / 1e3, err.max() / 1e3))
print("dv aparente nos controles (m/s): mediana %.4f p90 %.4f max %.4f" % (np.median(dvc), np.percentile(dvc, 90), np.max(dvc)))
