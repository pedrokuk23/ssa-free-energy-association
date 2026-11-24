"""E3 v2: condições iniciais CW-limitadas (ẏ0 = −2n x0 + ruído pequeno), AR dimensionada pelo cenário
(cobertura conferida), termo μσφ(μ/σ) na integral de alcance, verificação RB vs MC e EKF salva."""
import sys, json, numpy as np
from scipy.stats import norm
from scipy.special import logsumexp
import e5_tracklet as E
from numpy.polynomial.hermite_e import hermegauss
E.GH7x, E.GH7w = hermegauss(5); E.GH7w = E.GH7w / E.GH7w.sum()
sel = np.linspace(0, 63, 16).astype(int); E.MM = E.MM[sel]
n = E.n

def make_case(same, dv):
    rng = E.rng
    rho = rng.uniform(1000, 4000); ang = rng.uniform(0, 2*np.pi)
    x, y = rho*np.cos(ang), rho*np.sin(ang)
    x0 = np.r_[x, y, 0.3*rng.standard_normal(), -2*n*x + 0.01*rng.standard_normal()]
    S0 = E.rand_cov(); m0 = x0 + E.sample(S0)
    xa = x0 if same else x0 + E.sample(E.OFF @ E.OFF)
    x1 = E.Phi @ xa + E.sample(E.G0)
    if dv > 0:
        t = rng.uniform(0.2*E.T, 0.8*E.T); a = rng.uniform(0, 2*np.pi)
        x1 = x1 + E.expm(E.A*(E.T-t)) @ E.B @ (dv*np.r_[np.cos(a), np.sin(a)])
    z = E.h(x1)[0] + np.r_[E.sth, E.sthd]*rng.standard_normal(2)
    return m0, S0, z, x1
E.make_case = lambda same, dv: make_case(same, dv)[:3]

def _log_g(theta, thd, m, S):
    c, s = np.cos(theta), np.sin(theta)
    J = np.array([[c, 0], [s, 0], [-thd*s, c], [thd*c, s]])
    Si = np.linalg.inv(S); P = J.T@Si@J; b = J.T@Si@m
    Pi = np.linalg.inv(P); r = Pi@b; cq = m@Si@m - b@r
    _, ldS = np.linalg.slogdet(S); _, ldP = np.linalg.slogdet(P)
    sr = np.sqrt(Pi[0, 0]); mu = r[0]
    m2 = (mu**2 + sr**2)*norm.cdf(mu/sr) + mu*sr*norm.pdf(mu/sr)   # E[ρ² 1{ρ>0}]
    return -0.5*cq - 0.5*ldS - 0.5*ldP - np.log(2*np.pi) + np.log(m2 + 1e-300)
E._log_g = _log_g

if __name__ == "__main__":
    N = int(sys.argv[1])
    # cobertura da AR e escolha dos limites a partir do cenário
    fin = np.array([make_case(True, dv)[3] for dv in [0.0]*400 + [0.4]*400])
    rr = np.hypot(fin[:, 0], fin[:, 1]); rd = (fin[:, 0]*fin[:, 2] + fin[:, 1]*fin[:, 3])/rr
    print("range pct", np.percentile(rr, [0, 5, 50, 95, 100]).round(0), "rdot pct", np.percentile(np.abs(rd), [50, 95, 100]).round(2))
    E.RHO = (100.0, float(np.ceil(rr.max()*1.2/1000)*1000)); E.RHOD = float(np.ceil(np.abs(rd).max()*1.2))
    cov = np.mean((rr >= E.RHO[0]) & (rr <= E.RHO[1]) & (np.abs(rd) <= E.RHOD))
    print("AR", E.RHO, E.RHOD, "coverage", cov, flush=True)
    # verificação RB vs MC (prior sampling, NP=20000) e vs EKF
    chk = []
    for i in range(60):
        m0, S0, z, _ = make_case(i % 2 == 0, 0.0)
        f0 = E.rb_free_energies(m0, S0, z)[0]
        mp0 = E.Phi@m0; Pp0 = E.Phi@S0@E.Phi.T + E.G0
        Xs = mp0 + E.rng.standard_normal((400000, 4))@np.linalg.cholesky(Pp0).T
        ll = E.loglik(Xs, z) - np.log(2*np.pi*E.sth*E.sthd)
        mc = -(logsumexp(ll) - np.log(len(Xs))); ess = np.exp(2*logsumexp(ll) - logsumexp(2*ll))
        mp = E.Phi@m0; Pp = E.Phi@S0@E.Phi.T + E.G0; Hj = E.hjac(mp)
        nu = z - E.h(mp)[0]; nu[0] = np.angle(np.exp(1j*nu[0]))
        Sz = Hj@Pp@Hj.T + np.diag([E.sth**2, E.sthd**2])
        ekf = -E.logN0(nu, Sz)
        chk.append([f0, mc, ekf, ess])
    chk = np.array(chk); ok = chk[:, 3] > 200; d_mc = chk[ok, 1]-chk[ok, 0]; d_ekf = chk[:, 2]-chk[:, 0]
    print("RB-MC pct", np.percentile(d_mc, [5, 50, 95]).round(3), "EKF-RB pct", np.percentile(d_ekf, [5, 50, 95]).round(3), flush=True)
    neg, _ = E.run2(False, 0.0, N)
    res = {"AR": {"rho": E.RHO, "rhod": E.RHOD, "coverage": float(cov)},
           "check": {"rb_minus_mc_p5_50_95": np.percentile(-d_mc, [5, 50, 95]).tolist(),
                     "ekf_minus_rb_p5_50_95": np.percentile(d_ekf, [5, 50, 95]).tolist(),
                     "max_abs_rb_mc": float(np.abs(d_mc).max()), "max_abs_ekf_rb": float(np.abs(d_ekf).max()), "n_ess_ok": int(ok.sum()), "n": len(chk)}}
    allv = {"neg": neg}
    for dv in [0.0, 0.02, 0.05, 0.1, 0.2, 0.4]:
        pos, _ = E.run2(True, dv, N); allv[f"pos_{dv}"] = pos
        res[dv] = {k: round(float(E.auc(pos[:, i], neg[:, i])), 3) for i, k in enumerate(E.NAMES2)}
        print(dv, res[dv], flush=True)
    json.dump(res, open("e3v2_summary.json", "w"), indent=1)
    np.savez("e3v2_results.npz", **allv)
