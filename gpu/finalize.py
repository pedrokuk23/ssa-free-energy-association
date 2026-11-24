"""Gera um resumo compacto (para leitura remota) a partir dos arquivos de linhas do E9."""
import json, sys, numpy as np
import e9_analyze as A
files = sys.argv[1:]; out = {}
acc_big = lambda rr, m: A.tpr([r["M"][m] for r in rr if r["s"] >= 0.05], [r["F"][m] for r in rr])
acc_x = lambda rr, m: A.tpr([r["M"][m] for r in rr if r["tau"] == "cross-track"], [r["F"][m] for r in rr])
aucnb = lambda rr, m: A.auc([r["M"][m] for r in rr if r["s"] >= 0.05], [r["N"][m] for r in rr])
def ll(rr, m):
    p1 = np.mean([r["p1"] for r in rr]); a0 = np.clip([r["Z"]["pH1_"+m] for r in rr],1e-9,1-1e-9); a1 = np.clip([r["M"]["pH1_"+m] for r in rr],1e-9,1-1e-9)
    return -((1-p1)*np.mean(np.log(1-a0)) + p1*np.mean(np.log(a1)))
for f in files:
    rows = json.load(open(f)); key = f.replace("e9_rows_", "").replace(".json", "")
    met = A.metrics(rows)
    o = {"n": len(rows), "metrics": {m: {k: round(v, 3) for k, v in d.items()} for m, d in met.items()}}
    o["paired"] = {}
    for nm, fn in [("acc_big", acc_big), ("acc_cross", acc_x), ("auc_nb_big", aucnb)]:
        for b in ["GM", "CTRL", "iso_lin"]:
            o["paired"][f"{nm}:learn-{b}"] = A.paired_diff(rows, "learn", b, fn, B=500)
    for b in ["iso_lin", "maxwell_0.3", "maxwell_3", "flat", "maxwell_0.03"]:
        o["paired"][f"logloss:{b}-learn"] = A.paired_diff(rows, b, "learn", ll, B=500)
    o["ci_acc_big"] = {m: A.boot(lambda rr, m=m: acc_big(rr, m), rows, B=300) for m in ["GM", "CTRL", "iso_lin", "learn"]}
    o["ci_auc_nb"] = {m: A.boot(lambda rr, m=m: aucnb(rr, m), rows, B=300) for m in ["GM", "CTRL", "iso_lin", "learn"]}
    o["ci_logloss"] = {m: A.boot(lambda rr, m=m: ll(rr, m), rows, B=300) for m in ["learn", "flat", "iso_lin", "maxwell_0.3", "maxwell_3", "maxwell_0.03"]}
    names = ["radial", "in-track", "cross-track"]
    sel = [r for r in rows if r["s"] >= 0.05 and r["M"]["pH1_learn"] > 0.5]
    o["char"] = [[round(r["s"], 4), round(r["M"]["dv_hat"], 4), r["tau"][0], names[int(np.argmax(r["M"]["p_dir"]))][0]] for r in sel]
    out[key] = o
json.dump(out, open("e9_gpu_summary.json", "w"), separators=(",", ":"))
for k, o in out.items():
    print("==", k, "n", o["n"])
    for m in ["GM", "CTRL", "iso_lin", "learn"]:
        print("  ", m, o["metrics"][m])
    print("   paired", o["paired"])
