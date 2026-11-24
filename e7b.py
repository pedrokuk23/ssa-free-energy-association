"""E7b — (i) objeto manobrado vs objetos não relacionados no campo (±50 km em RIC em torno da previsão);
(ii) detecção de manobra: P(H1|z) [proposto] vs Δv mínimo [CTRL] vs −log p0 [GM, estilo Montilla]."""
import numpy as np, json, sys, time
from scipy.special import logsumexp
from e7_maneuver import *
from orbit import ric_frame
N = int(sys.argv[1]); s_vI = float(sys.argv[2]); rng = np.random.default_rng(33)

def scores2(z, m0, P0, e):
    R = SIG_M**2*np.eye(3); out = {}
    for name, Kc in [("UKF",1),("GM",K)]:
        w,M,my,Pp = gm_predict(m0,P0,e,Kc); r = z[None]-my
        l0 = logsumexp(loggauss(r,Pp+R)+np.log(w)); Phi = impulse_maps(M)
        S1 = Pp[:,None]+R+SIGMA_MAN**2*np.einsum("kjab,kjcb->kjac",Phi,Phi)
        l1 = logsumexp(loggauss(r[:,None],S1)+np.log(w)[:,None])-np.log(len(TJ))
        out[name] = -l0; out[name+"+M"] = -np.logaddexp(np.log(1-P_MAN)+l0, np.log(P_MAN)+l1)
        out[name+"_pH1"] = 1/(1+np.exp(np.log(1-P_MAN)+l0-np.log(P_MAN)-l1))
        if name=="GM":
            dv = np.linalg.solve(Phi, np.broadcast_to(r[:,None,:,None], Phi.shape[:2]+(3,1)))[...,0]
            out["CTRL"] = np.linalg.norm(dv,axis=-1).min()
    return out

A = ["UKF","UKF+M","GM","CTRL","GM+M"]
neg = []
for _ in range(N):
    m0,P0,e,yT = make_case(rng,s_vI); mT = propagate(m0,T); Rm = ric_frame(mT)
    off = Rm.T @ (rng.uniform(-50e3,50e3,3)*np.array([0.1,1,0.3]))   # radial ±5, along ±50, cross ±15 km
    neg.append([scores2(mT[:3]+off,m0,P0,e)[k] for k in A])
neg = np.array(neg); res = {}; pos0 = None
for dv in [0.0,0.1,0.3,1.0]:
    P = []
    for _ in range(N):
        m0,P0,e,yT = truth_with_maneuver(rng,s_vI,dv); z = yT[:3]+SIG_M*rng.standard_normal(3)
        P.append(scores2(z,m0,P0,e))
    pos = np.array([[p[k] for k in A] for p in P])
    det = {"GM+M: P(H1|z)": np.array([-p["GM_pH1"] for p in P]), "CTRL: Δv mín": np.array([-p["CTRL"] for p in P]),
           "GM: −log p0": np.array([-p["GM"] for p in P]), "UKF: −log p0": np.array([-p["UKF"] for p in P])}
    if dv == 0: pos0 = det
    r = {"auc_campo": {k: round(float(auc(pos[:,i],neg[:,i])),3) for i,k in enumerate(A)},
         "tpr1_campo": {k: round(float(np.mean(pos[:,i] < np.quantile(neg[:,i],0.01))),3) for i,k in enumerate(A)}}
    if dv > 0:   # detecção: maior estatística = manobra; auc(pos=sem manobra (menor), neg=com manobra)
        r["auc_deteccao"] = {k: round(float(auc(det[k], pos0[k])),3) for k in det}
    res[dv] = r; print(dv, json.dumps(r, ensure_ascii=False), flush=True)
json.dump(res, open(f"e7b_s{s_vI}.json","w"), indent=1, ensure_ascii=False)
