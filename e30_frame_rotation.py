"""Frame-rotation test of the tracklet evidence (Supplementary Section S6): the whole LEO problem of E5 Study B
(track, station, truth) is rotated about the polar axis by 30 random angles; log-evidences are recomputed with
e11_numeric.ev() (CIRC_MEAN=1, EQ_SQRT=1). Usage: python e30_frame_rotation.py . <seed> 0.5   (seeds 3, 4, 5 in the paper)"""
# (core) Invariância rotacional (eixo z) dos escores completos de Track3 (tracklet), problema LEO de E5 (J2-J4 + arrasto
# com atmosfera girante: simétrico por rotação em torno de z). Gira atributos da trilha em vez de re-propagar.
import sys, os, time, json, numpy as np
sys.path.insert(0, sys.argv[1]); os.chdir(sys.argv[1]); os.environ["MEAS"]="tracklet"
os.environ.setdefault("K1","11")
import e11_numeric as X
from orbit import ref_state
from e6_nonlinear import prior_cov
def Rz(a):
    c,s=np.cos(a),np.sin(a); R=np.array([[c,-s,0],[s,c,0],[0,0,1.]]); Z=np.zeros((6,6)); Z[:3,:3]=R; Z[3:,3:]=R; return Z,R
rng=np.random.default_rng(int(sys.argv[2]) if len(sys.argv)>2 else 3)
s_vI=float(sys.argv[3]) if len(sys.argv)>3 else 0.5
Xref=ref_state(M=rng.uniform(0,6.28), raan=rng.uniform(0,6.28)); P0,e=prior_cov(Xref,s_vI)
x0=Xref+np.linalg.cholesky(P0)@rng.standard_normal(6)
t=time.time(); trk=X.Track3(Xref,P0,e); print("track init %.0fs"%(time.time()-t), flush=True)
y=X.truth(x0,0.0,None,rng)
L=X.load_labels(); pr=X.make_priors(X.fold_prior(L, sorted(L[L.orbit_class=="LEO"].norad_id.unique())[0]))
pr={k:pr[k] for k in ("learn","iso_lin")}
st=X.station_for(y,rng); zz0=X.h_tracklet(y,st)
Rm=np.diag([(0.7*X.AS/max(np.cos(zz0[1]),.2))**2,(0.7*X.AS)**2,(0.04*X.AS/max(np.cos(zz0[1]),.2))**2,(0.04*X.AS)**2])
vec=["m0","Bm","H0_m","Lm"]; cov=["P0","BP","H0_P","iso_P","Wg","LP"]; rows=["G","g"]
def rotated(Z):
    T=X.Track3.__new__(X.Track3); T.__dict__.update(trk.__dict__)
    for k in vec+rows: setattr(T,k,np.einsum("ij,...j->...i",Z,getattr(trk,k)))
    for k in cov: setattr(T,k,np.einsum("ij,...jk,lk->...il",Z,getattr(trk,k),Z))
    return T
# dispersão em alfa das médias das componentes H0
al=X.h_tracklet(trk.m0,st)[:,0]; print("spread alpha das medias H0 (rad): %.3f"%np.ptp(np.unwrap(al)))

# Frame-rotation test (audit 2, item M01): rotations about the polar axis of the whole problem (track, station, truth),
# with the corrected ev() of the package (CIRC_MEAN=1, EQ_SQRT=1). Reports log-evidences for 30 random angles.
import sys, os, numpy as np
from scipy.special import logsumexp
a0=zz0[0]; rr=np.random.default_rng(123); angs=rr.uniform(-np.pi,np.pi,30)
res=[]
for ang in angs:
    Z,R=Rz(ang); s2=(R@st[0],R@st[1]); m=X.Meas("tracklet",s2); z=X.h_tracklet(Z@y,s2); T=rotated(Z)
    lh=X.ev(m,z,Rm,T.H0_m,T.H0_P,2)[0]; l0b=logsumexp(lh+np.log(T.w2))
    l0=logsumexp(X.ev(m,z,Rm,T.m0,T.P0,2)[0]+np.log(T.w0))
    mu=T.w0@T.m0; Pu=np.einsum("k,kde->de",T.w0,T.P0+np.einsum("kd,ke->kde",T.m0-mu,T.m0-mu)); lu=float(X.ev(m,z,Rm,mu,Pu,2)[0])
    res.append((l0,l0b,lu))
res=np.array(res)
print("seed",sys.argv[2],"GM41 mean %.3f range %.2e | H0_K11 mean %.3f range %.2e | UKF mean %.3f range %.2e"%(res[:,0].mean(),np.ptp(res[:,0]),res[:,1].mean(),np.ptp(res[:,1]),res[:,2].mean(),np.ptp(res[:,2])))
