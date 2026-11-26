# E6c, exploratory (audit 2026-11): sensitivity of the primary to pi1 from navigation satellites only, with the
# corrected tracklet scores (e17K3 rows via e27_circ_analyze.py). Usage: python e29_p1nav.py [package_dir] > e29_p1nav.txt
# Sensibilidade do primário do E6c a p1 calculado só com satélites de navegação (exclui SELF-GEO e NOAA-GOES).
# Recupera L0 e L1 de (F, pH1) guardados: e^{-F}=(1-p1)L0+p1L1, pH1=p1L1 e^{-F}... ; recalcula F com p1'.
import sys, os, json, glob, numpy as np, pandas as pd
P=sys.argv[1] if len(sys.argv)>1 else '.'; sys.path.insert(0,P); os.chdir(P)
os.environ.update(LABELS="data/labels_gnss.json", PRIOR_CLASS="MEO,GEO,IGSO", T_DAYS="2", A_REF="26.56e6")
import e10_numeric as E
L=E.load_labels(); Lnav=L[~L.source.isin(["SELF-GEO","NOAA-GOES"])]
import e27_circ_analyze as A   # executa a análise original (carrega E, keys etc.)
from e12K_analyze import hksj, signflip, by_sat
def newF(F,ph,p1,p1n):
    ph=np.clip(ph,1e-300,1); lt=-F                       # log e^{-F}
    with np.errstate(divide="ignore"):
        l1=lt+np.log(ph)-np.log(p1); l0=lt+np.log1p(-np.minimum(ph,1-1e-16))-np.log1p(-p1)
        l0=np.where(ph>=1.0, -np.inf, l0)
    return -np.logaddexp(np.log1p(-p1n)+l0, np.log(p1n)+l1)
rows_out={}; info=[]
for g in ["meo","qzss"]:
    for meas in ["full","tracklet"]:
        R=A.load(g,meas)
        for r in R:
            p1=E.fold_prior(L,r["sat"])["p1"]; p1n=E.fold_prior(Lnav,r["sat"])["p1"]; info.append((g,p1,p1n))
            c=[]
            for d in r["draws"]:
                Ft=newF(d["M"]["learn"],d["M"]["pH1_learn"],p1,p1n)
                # Mn: trilha do vizinho, mesmo satélite -> mesmo p1
                Fn=newF(d["Mn"]["learn"],d["Mn"]["pH1_learn"],p1,p1n)
                c.append(float(Ft<Fn)+0.5*float(Ft==Fn))
            rows_out[(g,r["eid"],meas)]=np.mean(c)
# checagem: com p1n=p1 a recomputação reproduz corr original
info=np.array([(x[1],x[2]) for x in info]); print("p1 original (mediana) %.4f  p1 navegação (mediana) %.4f"%tuple(np.median(info,0)))
d=[]; sats=[]
for k in A.keys:
    g,eid=k
    if g=="leo":
        new=np.mean([A.E[k]["meas"][m]["learn"]["corr"] for m in ("full","tracklet")])
    else:
        new=np.mean([rows_out[(g,eid,m)] for m in ("full","tracklet")])
    ctrl=np.mean([A.E[k]["meas"][m]["CTRLu"]["corr"] for m in ("full","tracklet")])
    d.append(new-ctrl); sats.append(A.E[k]["sat"])
d=np.array(d); sats=np.array(sats); y,v,n,us=by_sat(d,sats); o=hksj(y,v)
print("primário com p1 de navegação: mu %.4f IC %s p_one %.4f signflip %.4f (original +0.0264 [0.0118,0.0410])"%(o["mu"],o["ci"],o["p_one"],signflip(y,1/np.maximum(v,1e-12))))
# sanidade: recomputar com p1n=p1
chk=[]
for g in ["meo","qzss"]:
    for meas in ["full","tracklet"]:
        for r in A.load(g,meas)[:5]:
            p1=E.fold_prior(L,r["sat"])["p1"]
            for dd in r["draws"]: chk.append(abs(newF(dd["M"]["learn"],dd["M"]["pH1_learn"],p1,p1)-dd["M"]["learn"])/max(1,abs(dd["M"]["learn"])))
print("sanidade (p1'=p1) max rel dif:",max(chk))
