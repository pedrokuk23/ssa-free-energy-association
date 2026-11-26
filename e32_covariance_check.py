"""Covariance check of loggauss (Supplementary Section S6): every matrix passed to loggauss during full scoring
(all methods) of LEO objects of the E5 Study B setting, tracklet and full-state UCTs, with and without an in-track
0.5 m/s maneuver, sigma_vI = 0.5 and 0.05 m/s. Usage: python e32_covariance_check.py . x 3,4,5,6,7,8 > e32_covariance_check.txt"""
# Audit C02: are the covariance matrices passed to loggauss positive definite? Full scoring (all methods) of
# LEO objects of the E5 Study B setting, tracklet and full-state UCTs, with and without a maneuver.
import sys, os, time, numpy as np
P=sys.argv[1]; sys.path.insert(0,P); os.chdir(P); os.environ["MEAS"]="tracklet"; os.environ["EXTRA"]="1"; os.environ["EXTRA2"]="1"
import e11_numeric as X, e10_numeric as E
from orbit import ref_state
from e6_nonlinear import prior_cov
stat=dict(n=0, nonpd=0, nonfinite=0, cond_gt_1e12=0, minrel=np.inf, corr_cond_gt_1e8=0, corr_minrel=np.inf)
def wrap(f):
    def g(r,S):
        S=np.asarray(S); lam=np.linalg.eigvalsh(0.5*(S+np.swapaxes(S,-1,-2)))
        rel=lam[...,0]/np.maximum(np.abs(lam[...,-1]),1e-300)
        stat["n"]+=rel.size; stat["nonpd"]+=int(np.sum(lam[...,0]<=0)); stat["nonfinite"]+=int(np.sum(~np.isfinite(S).all((-1,-2))))
        stat["cond_gt_1e12"]+=int(np.sum((rel>0)&(rel<1e-12))); stat["minrel"]=min(stat["minrel"], float(np.min(rel)))
        d=np.sqrt(np.abs(np.einsum("...ii->...i",S))); C=S/(d[...,:,None]*d[...,None,:]); lc=np.linalg.eigvalsh(0.5*(C+np.swapaxes(C,-1,-2))); rc=lc[...,0]/lc[...,-1]
        stat["corr_cond_gt_1e8"]+=int(np.sum(rc<1e-8)); stat["corr_minrel"]=min(stat["corr_minrel"], float(np.min(rc)))
        return f(r,S)
    return g
X.loggauss=wrap(X.loggauss); E.loggauss=wrap(E.loggauss)
L=X.load_labels(); pr=X.make_priors(X.fold_prior(L, sorted(L[L.orbit_class=="LEO"].norad_id.unique())[0]))
for seed in map(int, sys.argv[3].split(",")):
  for s_vI in (0.5,0.05):
    rng=np.random.default_rng(seed); Xref=ref_state(M=rng.uniform(0,6.28), raan=rng.uniform(0,6.28)); P0,e=prior_cov(Xref,s_vI)
    x0=Xref+np.linalg.cholesky(P0)@rng.standard_normal(6); t=time.time(); trk=X.Track3(Xref,P0,e)
    for mnv in (None, 0.5):
        y=X.truth(x0,0.0,None,rng) if mnv is None else X.truth(x0,0.0,("in-track",mnv),rng,tm=rng.uniform(0.1,0.9)*86400,sgn=1.0)
        st=X.station_for(y,rng); zt=X.h_tracklet(y,st)
        Rt=np.diag([(0.7*X.AS/max(np.cos(zt[1]),.2))**2,(0.7*X.AS)**2,(0.04*X.AS/max(np.cos(zt[1]),.2))**2,(0.04*X.AS)**2])
        o1=trk.scores(zt+np.sqrt(np.diag(Rt))*rng.standard_normal(4), Rt, X.Meas("tracklet",st), pr)
        Rf=np.diag([30.,100,30,.05,.03,.05])**2
        o2=trk.scores(y+np.sqrt(np.diag(Rf))*rng.standard_normal(6), Rf, X.Meas("full"), pr)
        fin=all(bool(np.all(np.isfinite(np.asarray(v,float)))) for v in list(o1.values())+list(o2.values()))
        print("seed %d s %.2f mnv %s  finite scores %s  methods %d  (%.0fs)"%(seed,s_vI,mnv,fin,len(o1),time.time()-t), flush=True)
print("STAT", stat)
