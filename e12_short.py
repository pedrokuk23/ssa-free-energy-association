import json, sys
d = json.load(open(sys.argv[1])); out = {}
for k, o in d.items():
    if k == "holm":
        out[k] = d[k]; continue
    out[k] = {x: o[x] for x in ("n_man", "n_big", "n_ctrl", "by_sat", "metrics", "primary", "secondary", "char") if x in o}
    out[k]["metrics"] = {m: [o["metrics"][m][q] for q in ("acc_big", "acc_ctrl", "acc_cross", "det_big", "aucJ_big")] for m in o["metrics"]}
print(json.dumps(out))
