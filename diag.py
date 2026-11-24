import numpy as np
from scipy.special import logsumexp
import e9_numeric as E
from orbit import ref_state
from e6_nonlinear import prior_cov
rng=np.random.default_rng(3)
for i in range(8):
    Xref=ref_state(M=rng.uniform(0,2*np.pi),raan=rng.uniform(0,2*np.pi)); P0,e=prior_cov(Xref,0.5)
    x0=Xref+np.linalg.cholesky(P0)@rng.standard_normal(6); b=E.SIG_BETA*rng.standard_normal()
    t=E.Track(Xref,P0,e); y=E.truth(x0,b,None,rng); z,S=E.uct(rng,y)
    l0=logsumexp(E.loggauss(z-t.m0,t.P0+S)+np.log(t.w0))
    l0lin=logsumexp(E.loggauss(z-t.H0lin_m,t.H0lin_P+S)+np.log(t.w2))
    lb=t.logl_blocks(z,S)   # (K2,J,3,2,B)
    l_tiny=logsumexp(lb[...,0]+np.log(t.w2)[:,None,None,None])-np.log(lb[...,0].size/len(t.w2))
    print(f"l0(K0 UT)={l0:9.2f}  l0(K2 lin)={l0lin:9.2f}  l1(|Δv|=1.4mm/s)={l_tiny:9.2f}")
