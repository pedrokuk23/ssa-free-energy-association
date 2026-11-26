"""Scalar benchmark of the evidence formula of e11_numeric.ev() (Supplementary Section S6): y = x^2 + eta,
x ~ N(0,1) split into K Gaussian components, eta ~ N(0, 0.2^2); two IPL iterations, zero centre weight, 7-sigma rule.
Usage: python e31_ipl_scalar.py > e31_ipl_scalar.txt"""
# Audit 2, item M06: scalar benchmark of the evidence formula of e11_numeric.ev() (iterated posterior linearization,
# two iterations, unscented points with zero center weight, 7-sigma rule). y = x^2 + eta, x ~ N(0,1), eta ~ N(0, 0.2^2).
import numpy as np
from scipy import integrate, stats
s=0.2; R=s*s
def exact(z): return integrate.quad(lambda x: stats.norm.pdf(x)*stats.norm.pdf(z, x*x, s), -12, 12, limit=400, points=[-np.sqrt(abs(z)), np.sqrt(abs(z))])[0]
def ipl(z, it=2, mT=0.0, PT=1.0):
    n=1; m, P = mT, PT; out=None; act=True
    for _ in range(it):
        S=np.sqrt(n*P); pts=np.array([m, m+S, m-S]); W=np.array([0, .5, .5]); Hy=pts**2; my=W@Hy
        Pyy=W@((Hy-my)**2); Cxy=W@((pts-m)*(Hy-my)); A=Cxy/P; b=my-A*m; Om=max(Pyy-A*A*P, 0.0)
        Sy=A*A*PT+Om+R; yp=A*mT+b; l=stats.norm.pdf(z, yp, np.sqrt(Sy))
        if out is None or act: out=l
        K=PT*A/Sy; mn=mT+K*(z-yp); Pn=PT-K*A*PT
        ok=act and (mn-mT)**2/PT < 49
        if ok: m, P = mn, Pn
        act=ok
    return out

def mix(K):
    if K==1: return np.array([0.]),np.array([1.]),np.array([1.])
    mu=np.linspace(-4,4,K); h=mu[1]-mu[0]; w=stats.norm.pdf(mu); w=w/w.sum()
    v0=1-np.sum(w*mu**2); var=np.full(K,max(v0,(0.3*h)**2))
    return mu,var,w
def ipl_mix(z,K):
    mu,var,w=mix(K); return sum(wi*ipl(z,2,m,v) for m,v,wi in zip(mu,var,w))
def exact_mix(z,K):
    mu,var,w=mix(K); return sum(wi*integrate.quad(lambda x: stats.norm.pdf(x,m,np.sqrt(v))*stats.norm.pdf(z,x*x,s),m-10*np.sqrt(v),m+10*np.sqrt(v),limit=200)[0] for m,v,wi in zip(mu,var,w))
for K in (1,11,41,81):
    vals=[ipl_mix(z,K) for z in (0.0,1.0,4.0)]; ex=[exact_mix(z,K) for z in (0.0,1.0,4.0)]
    zs=np.linspace(-2,30,1601); I=np.trapezoid([ipl_mix(z,K) for z in zs],zs)
    print("K=%2d  IPL p(0,1,4)= %.4f %.4f %.5f | exact (same mixture) %.4f %.4f %.5f | log-ratio %+.3f %+.3f %+.3f | integral %.4f"%(K,*vals,*ex,*[np.log(v/e) for v,e in zip(vals,ex)],I))
