import json, sys
for fn in sys.argv[1:]:
    D = json.load(open(fn))
    for k, o in D.items():
        print("##", k, o.get("od", ""), o.get("rule7", ""))
        print(" ".join("%s:%.2f/%.2f/%.2f%s" % (m, v["acc_big"], v["aucJ_big"], v["det_big"], ("/ll%.2f" % v["logloss"]) if "logloss" in v else "") for m, v in o["metrics"].items()))
        print(" ".join("%s:%s=%+.3f[%.3f,%.3f]" % (d.split(":")[0][:4], d.split("learn-")[1], v[0], v[1], v[2]) for d, v in o["diff"].items()))
