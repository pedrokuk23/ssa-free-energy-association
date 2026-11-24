import json, numpy as np
d = json.load(open("e9_gpu_summary_all.json"))
out = []
for k, o in d.items():
    c = o.pop("char")
    o["char_stats"] = {"n": len(c), "type_acc": round(float(np.mean([a[2] == a[3] for a in c])), 3),
                       "median_factor": round(float(10 ** np.median([abs(np.log10(a[1] / a[0])) for a in c])), 3)}
    out.append(k + " " + json.dumps(o, separators=(",", ":")))
    s = ";".join(f"{a[0]:.3g},{a[1]:.3g},{a[2]}{a[3]}" for a in c)
    for i in range(0, len(s), 4000):
        out.append(f"CHAR {k} " + s[i:i + 4000])
open("compact.txt", "w").write("\n".join(out) + "\n")
print([len(x) for x in out])
