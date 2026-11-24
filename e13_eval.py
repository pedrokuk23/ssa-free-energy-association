import json, sys, numpy as np
f = sys.argv[1]; R = json.load(open(f))
M = [r for r in R if r["kind"] == "man" and r["det"]]; C = [r for r in R if r["kind"] == "ctrl" and r["det"]]
print("n man", len(M), "n ctrl", len(C), "sats", sorted({r["sat"] for r in R}))
zm = np.array([r["det"]["z"] for r in M]); zc = np.array([r["det"]["z"] for r in C])
dt = np.array([abs(r["dt_hours"]) for r in M]); iso = np.array([r["n_near"] == 0 for r in M])
rat = np.array([r["det"]["dv"] / r["dv"] for r in M])
print("z man pct 5/25/50:", np.percentile(zm, [5, 25, 50]).round(1), " z ctrl pct 50/90/max:", np.percentile(zc, [50, 90, 100]).round(1))
for thr in [5, 8, 10, 15, 20, 30, 50]:
    for tw in [3, 6, 12]:
        tp = ((zm > thr) & (dt <= tw)); rec = tp[iso].mean(); fp = (zc > thr).sum()
        prec = tp.sum() / max(tp.sum() + fp, 1)
        print(f"thr {thr:3d} win {tw:2d}h  recall_iso {rec:.3f}  prec {prec:.3f}  fp {fp}/{len(C)}")
ok = iso & (dt <= 6)
lr = np.abs(np.log(rat[ok])); print("magnitude factor median", round(float(np.exp(np.median(lr))), 3), "p75", round(float(np.exp(np.percentile(lr, 75))), 3), "median ratio", round(float(np.median(rat[ok])), 3))
print("dt hours median", round(float(np.median(dt[iso])), 2))
for s in sorted({r["sat"] for r in M}):
    mm = [r for r in M if r["sat"] == s and r["n_near"] == 0]
    print(s, len(mm), "medratio", round(float(np.median([r["det"]["dv"] / r["dv"] for r in mm])), 2) if mm else "-", "types", {t: sum(1 for r in mm if r["typ"] == t) for t in {r["typ"] for r in mm}}, "det types", {t: sum(1 for r in mm if r["det"]["typ"] == t) for t in {r["det"]["typ"] for r in mm}})
