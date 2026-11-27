"""Worst-case bound on the corrected E6c primary from rounding-level sensitivity of the tracklet maneuver scores
(audit revision 4, R4-03). e33 showed that, with optical tracklets, the 'learn' and 'lin_learn' scores of a draw can
change by up to about 0.7 and 2.7 nat under a 1e-15 relative change of the predicted covariance (hard gate of the
iterated posterior linearization); full-state scores and the CTRLu baseline do not change. Here every tracklet draw
whose 'learn' margin |F_n - F_t| is below DELTA is counted as a wrong joint association (adversarial to the primary),
and the primary analysis of e27_circ_analyze.py is recomputed. Usage: python e34_rounding_bound.py [DELTA,...]"""
import sys, json, re
src = open("e27_circ_analyze.py").read()
src = src.replace('json.dump(res, open("e27_circ_summary.json", "w"), indent=1)', '')
out = {}
for delta in [float(x) for x in (sys.argv[1] if len(sys.argv) > 1 else "0.5,1,2,3").split(",")]:
    n = dict(changed=0, total=0)
    def patch(rows, meas, delta=delta, n=n):
        if meas != "tracklet":
            return rows
        for r in rows:
            for d in r["draws"]:
                n["total"] += 1
                if abs(d["Mn"]["learn"] - d["M"]["learn"]) < delta:
                    if d["M"]["learn"] < d["Mn"]["learn"]:
                        n["changed"] += 1
                    d["M"]["learn"] = d["Mn"]["learn"] + 1.0       # adversarial: target loses
        return rows
    g = {"__name__": "e34", "PATCH": patch}
    s2 = src.replace("rows = load(g, meas)", "rows = PATCH(load(g, meas), meas)").replace('print("E6c', '0 and print("E6c').replace('print("por regime', '0 and print("por regime')
    exec(compile(s2, "e27_circ_analyze.py", "exec"), g)
    p = g["res"]["primary"]
    out[delta] = dict(tracklet_draws=n["total"], correct_decisions_turned_wrong=n["changed"],
                      mu=p["mu"], ci=p["ci"], lower_one_sided_95=p["lower_one_sided_95"], p_one=p["p_one"], success=p["success"])
    print(f"DELTA {delta} nat:", json.dumps(out[delta]), flush=True)
# tracklet-agnostic bound: the tracklet half of the primary contributes nothing (learn and CTRLu tie in every tracklet
# event), i.e. the primary equals half the full-state difference; full-state scores are reproduced exactly (e33)
g = {"__name__": "e34"}; s3 = src.replace('print("E6c', '0 and print("E6c').replace('print("por regime', '0 and print("por regime')
s3 = s3.replace('res["primary"] = test("learn", "CTRLu")', 'res["primary"] = test("learn", "CTRLu", meas="full")')
exec(compile(s3, "e27_circ_analyze.py", "exec"), g); p = g["res"]["primary"]
out["tracklet_half_zero"] = dict(mu=round(p["mu"] / 2, 4), ci=[round(c / 2, 4) for c in p["ci"]], lower_one_sided_95=round(p["lower_one_sided_95"] / 2, 4),
                                 p_one=p["p_one"], note="HKSJ is scale-equivariant: half the full-state test (mu, CI, bound halved; p unchanged)")
print("tracklet half set to zero:", json.dumps(out["tracklet_half_zero"]))
json.dump(out, open("e34_rounding_bound.json", "w"), indent=1)
