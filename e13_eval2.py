import json, glob, os, numpy as np
R = []
for t in "abc":
    f = f"e13_val_21_v2{t}.json"
    f = f if os.path.exists(f) else f"e13_val_21_v2{t}.partial.json"
    if os.path.exists(f): R += json.load(open(f))
M = [r for r in R if r["kind"] == "man" and r["det"]]; C = [r for r in R if r["kind"] == "ctrl" and r["det"]]
sats = sorted({r["sat"] for r in R}); print("sats", sats, "man", len(M), "ctrl", len(C))
med = {}
for s in sats:
    cc = [r["det"] for r in C if r["sat"] == s]
    med[s] = (np.median([d["z_it"] for d in cc]), np.median([d["z_ct"] for d in cc])) if cc else (np.nan, np.nan)
    print(s, "nctrl", len(cc), "med z_it %.1f z_ct %.1f" % med[s], "nman_iso", sum(1 for r in M if r["sat"] == s and r["n_near"] == 0))
def zn(r):
    a, b = med[r["sat"]]; d = r["det"]; return max(d["z_it"] / a, d["z_ct"] / b)
def dvsel(r):
    a, b = med[r["sat"]]; d = r["det"]; return d["dv_it"] if d["z_it"] / a >= d["z_ct"] / b else d["dv_ct"]
iso = [r for r in M if r["n_near"] == 0]
zm = np.array([zn(r) for r in iso]); zc = np.array([zn(r) for r in C]); dt = np.array([abs(r["dt_hours"]) for r in iso])
zr = np.array([r["det"]["z"] for r in iso]); zrc = np.array([r["det"]["z"] for r in C])
print("raw z: man p10/50 %s ctrl p50/90/max %s" % (np.percentile(zr, [10, 50]).round(1), np.percentile(zrc, [50, 90, 100]).round(1)))
print("norm z: man p10/50 %s ctrl p50/90/max %s" % (np.percentile(zm, [10, 50]).round(2), np.percentile(zc, [50, 90, 100]).round(2)))
for thr in [1.5, 2, 2.5, 3, 4, 5]:
    tp = (zm > thr) & (dt <= 6); fp = (zc > thr).sum()
    print(f"norm thr {thr}: recall {tp.mean():.3f} prec {tp.sum()/max(tp.sum()+fp,1):.3f} fp {fp}/{len(C)}")
for thr in [12, 15, 20]:
    tp = (zr > thr) & (dt <= 6); fp = (zrc > thr).sum()
    print(f"raw thr {thr}: recall {tp.mean():.3f} prec {tp.sum()/max(tp.sum()+fp,1):.3f} fp {fp}/{len(C)}")
ok = dt <= 6
for name, f in [("v2 dv", lambda r: r["det"]["dv"]), ("raw dv", lambda r: r["det"]["dv_raw"]), ("sel dv", dvsel)]:
    rat = np.array([f(r) / r["dv"] for r in iso])[ok]; lr = np.abs(np.log(rat))
    print(name, "factor med %.3f p75 %.3f medratio %.3f" % (np.exp(np.median(lr)), np.exp(np.percentile(lr, 75)), np.median(rat)))
for s in sats:
    ii = [r for r in iso if r["sat"] == s]
    if ii: print(s, "medratio sel %.2f" % np.median([dvsel(r) / r["dv"] for r in ii]), "type ok %d/%d" % (sum(1 for r in ii if (r["det"]["z_it"]/med[s][0] >= r["det"]["z_ct"]/med[s][1]) == (r["typ"] == "in-track")), len(ii)))
