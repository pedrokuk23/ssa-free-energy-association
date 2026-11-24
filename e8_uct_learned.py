"""
E8 — Correlação UCT–catálogo após lacuna de 1 dia (2 corpos + J2) com manobras REAIS.

Trilha nova (UCT): estado completo (pos+vel) com covariância típica de OD de passagem curta.
Manobras verdadeiras: sorteadas dos rótulos DORIS/IDS (operadores) de satélites de TESTE.
Priori aprendida: ajustada só nos satélites de TREINO (tipo, |Δv| em bins log, taxa/dia).

Estatísticas (menor = mesmo objeto, exceto as de detecção):
  UKF     : −log p(z|H0), gaussiana única
  GM      : −log p(z|H0), mistura K=41                       [Montilla et al. 2025]
  CTRL    : energia mínima de controle contínuo (gramiano)   [Holzinger et al. 2012, L2]
  GM+iso  : energia livre com priori de manobra isotrópica N(0, 0.3² I)
  GM+learn: energia livre com priori aprendida dos dados     [proposto]
"""
import numpy as np, json, sys, time
import pandas as pd
from scipy.special import logsumexp
from orbit import propagate, ref_state, ric_frame
from e6_nonlinear import prior_cov, split, ut_points

T = 86400.0
K0, K2 = 41, 21
TJ = np.linspace(0.06, 0.94, 8) * T
W_UT = np.r_[0.0, np.full(12, 1 / 12)]
UCT_SIG = np.array([30, 100, 30, 0.05, 0.03, 0.05])        # RIC: pos [m], vel [m/s]
SIG_ISO = 0.3

# ------------------------------------------------------------- priori aprendida
def learn_prior(seed=0, n_test=4):
    L = pd.DataFrame(json.load(open("data/maneuver_labels_v0.3.json")))
    L["epoch"] = pd.to_datetime(L.epoch, utc=True, format="mixed")
    d = L[(L.orbit_class == "LEO") & L.delta_v.notna() & (L.delta_v > 1e-4)
          & L.maneuver_type.isin(["in-track", "cross-track"])]
    sats = np.array(sorted(d.norad_id.unique()))
    rng = np.random.default_rng(seed); rng.shuffle(sats)
    test, train = set(sats[:n_test]), set(sats[n_test:])
    tr, te = d[d.norad_id.isin(train)], d[d.norad_id.isin(test)]
    edges = np.logspace(-3, 1.1, 15)                         # 14 bins, razão ≈ 2
    centers = np.sqrt(edges[:-1] * edges[1:])
    prior = {}
    for tau in ["in-track", "cross-track"]:
        h, _ = np.histogram(tr[tr.maneuver_type == tau].delta_v.clip(edges[0], edges[-1] * 0.999), edges)
        prior[tau] = (h + 0.5) / (h + 0.5).sum()             # suavização de Laplace
    ptype = tr.maneuver_type.value_counts(normalize=True)
    # taxa de manobra por dia (treino)
    days = sum((g.epoch.max() - g.epoch.min()).total_seconds() / 86400 for _, g in L[L.norad_id.isin(train)].groupby("norad_id"))
    rate = len(L[L.norad_id.isin(train)]) / days
    return dict(centers=centers, q=prior, ptype={k: float(ptype[k]) for k in prior},
                rate=rate, test=te[["maneuver_type", "delta_v", "norad_id"]].to_numpy(),
                train_sats=sorted(map(int, train)), test_sats=sorted(map(int, test)))

# ------------------------------------------------------------- utilidades
def ric_dirs(X):
    r, v = X[..., :3], X[..., 3:]
    R = r / np.linalg.norm(r, axis=-1, keepdims=True)
    C = np.cross(r, v); C /= np.linalg.norm(C, axis=-1, keepdims=True)
    I = np.cross(C, R)
    return R, I, C

def ric_cov(X, sig):
    R, I, C = ric_dirs(X); Q = np.stack([R, I, C], -1)
    Z = np.zeros((6, 6)); Z[:3, :3] = Q; Z[3:, 3:] = Q
    return Z @ np.diag(sig**2) @ Z.T

def ut_stats(Y):
    m = np.einsum("j,...jd->...d", W_UT, Y)
    D = Y - m[..., None, :]
    return m, np.einsum("j,...jd,...je->...de", W_UT, D, D)

def loggauss(r, S):
    sol = np.linalg.solve(S, r[..., None])[..., 0]
    _, ld = np.linalg.slogdet(S)
    return -0.5 * (np.sum(r * sol, -1) + ld + 6 * np.log(2 * np.pi))

# ------------------------------------------------------------- predição por hipótese
class Track:
    """Pré-computa, para um objeto de catálogo, as misturas preditas sob H0, H_iso e H_learn."""
    def __init__(self, m0, P0, e, PR):
        self.PR = PR
        w, M, P = split(m0, P0, e, K0)
        pts, _ = ut_points(M, P)
        self.w0 = w
        self.m0, self.P0 = ut_stats(propagate(pts, T))                         # (K0,6),(K0,6,6)
        w2, M2, P2 = split(m0, P0, e, K2)
        pts2, _ = ut_points(M2, P2)                                            # (K2,13,6)
        Xj = propagate(pts2[:, None], TJ[None, :, None])                       # (K2,J,13,6)
        mj = np.einsum("j,kajd->kad", W_UT, Xj)                                # (K2,J,6)
        R, I, C = ric_dirs(mj)
        dirs = np.stack([R, I, C], 2)                                          # (K2,J,3,3)
        h = 0.01
        base = propagate(mj, (T - TJ)[None, :])                                # (K2,J,6)
        Xd = mj[:, :, None, :].repeat(3, 2).copy(); Xd[..., 3:] += h * dirs
        G = (propagate(Xd, (T - TJ)[None, :, None]) - base[:, :, None]) / h    # (K2,J,3dir,6)
        self.G = G
        mH0, PH0 = ut_stats(propagate(pts2, T))
        self.w2, self.mH0_2, self.PH0_2 = w2, mH0, PH0
        # isotrópica (linearizada): cov += σ² Σ_dir g gᵀ
        self.iso_P = PH0[:, None] + SIG_ISO**2 * np.einsum("kjad,kjae->kjde", G, G)
        # gramiano (controle contínuo, L2)
        self.Wg = np.einsum("kjad,kjae->kde", G, G) * (T / len(TJ))
        # aprendida: propagação exata dos pontos-sigma com Δv = ±s_b·e_τ
        cen = PR["centers"]; B = len(cen)
        means, covs, logw = [], [], []
        for ti, tau in enumerate(["in-track", "cross-track"]):
            e_t = dirs[:, :, 1 if tau == "in-track" else 2]                    # (K2,J,3)
            for sg in (+1, -1):
                dv = sg * cen[None, None, :, None] * e_t[:, :, None, :]        # (K2,J,B,3)
                X = Xj[:, :, None].repeat(B, 2).copy()                         # (K2,J,B,13,6)
                X[..., 3:] += dv[:, :, :, None, :]
                Y = propagate(X, (T - TJ)[None, :, None, None])
                m, P = ut_stats(Y)                                             # (K2,J,B,6)
                # espalhamento dentro do bin (largura relativa ~0.35)
                Xs = mj[:, :, None].repeat(B, 2).copy(); Xs[..., 3:] += 1.1 * dv
                Ys = propagate(Xs, (T - TJ)[None, :, None])
                Xc = mj[:, :, None].repeat(B, 2).copy(); Xc[..., 3:] += dv
                Yc = propagate(Xc, (T - TJ)[None, :, None])
                g = (Ys - Yc) / 0.1                                            # dY/dlog s ≈ s·dY/ds
                P = P + 0.35**2 * np.einsum("...d,...e->...de", g, g)
                means.append(m); covs.append(P)
                lw = (np.log(w2)[:, None, None] - np.log(len(TJ)) + np.log(PR["ptype"][tau])
                      + np.log(0.5) + np.log(PR["q"][tau])[None, None, :])
                logw.append(np.broadcast_to(lw, m.shape[:-1]))
        self.L_m = np.stack(means).reshape(-1, 6)
        self.L_P = np.stack(covs).reshape(-1, 6, 6)
        self.L_lw = np.stack(logw).reshape(-1)
        self.L_type = np.repeat([0, 0, 1, 1], len(self.L_lw) // 4)             # 0 in-track, 1 cross
        self.L_mag = np.tile(np.repeat(cen[None], K2 * len(TJ), 0).reshape(-1), 4)

    def scores(self, z, S):
        p1 = np.log(self.PR["p1"]); p0 = np.log1p(-self.PR["p1"])
        # UKF (gaussiana única = momento da mistura K0)
        mu = self.w0 @ self.m0
        Pu = np.einsum("k,kde->de", self.w0, self.P0 + np.einsum("kd,ke->kde", self.m0 - mu, self.m0 - mu))
        l_ukf = loggauss(z - mu, Pu + S)
        l0 = logsumexp(loggauss(z - self.m0, self.P0 + S) + np.log(self.w0))
        r2 = z - self.mH0_2
        li = logsumexp(loggauss(r2[:, None], self.iso_P + S) + np.log(self.w2)[:, None]) - np.log(len(TJ))
        ll = loggauss(z - self.L_m, self.L_P + S) + self.L_lw
        lL = logsumexp(ll)
        post = np.exp(ll - lL)
        ctrl = min(r2[k] @ np.linalg.solve(self.Wg[k], r2[k]) for k in range(K2))
        out = dict(UKF=-l_ukf, GM=-l0, CTRL=ctrl,
                   **{"GM+iso": -np.logaddexp(p0 + l0, p1 + li), "GM+learn": -np.logaddexp(p0 + l0, p1 + lL)})
        out["pH1_iso"] = 1 / (1 + np.exp(p0 + l0 - p1 - li))
        out["pH1_learn"] = 1 / (1 + np.exp(p0 + l0 - p1 - lL))
        out["p_cross"] = float(post[self.L_type == 1].sum())
        out["dv_hat"] = float(np.exp(post @ np.log(self.L_mag)))
        return out

# ------------------------------------------------------------- cenários
def make_object(rng, s_vI):
    Xref = ref_state(M=rng.uniform(0, 2 * np.pi), raan=rng.uniform(0, 2 * np.pi))
    P0, e = prior_cov(Xref, s_vI)
    x0 = Xref + np.linalg.cholesky(P0) @ rng.standard_normal(6)
    return Xref, P0, e, x0

def truth_T(x0, man=None, rng=None):
    if man is None:
        return propagate(x0, T)
    tau, s = man
    tm = rng.uniform(0.05, 0.95) * T
    xm = propagate(x0, tm).copy()
    R, I, C = ric_dirs(xm)
    xm[3:] += rng.choice([-1, 1]) * s * (I if tau == "in-track" else C)
    return propagate(xm, T - tm)

def uct(rng, yT):
    S = ric_cov(yT, UCT_SIG)
    return yT + np.linalg.cholesky(S) @ rng.standard_normal(6), S

def auc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    r = np.concatenate([pos, neg]).argsort().argsort() + 1
    return 1 - (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))

if __name__ == "__main__":
    N = int(sys.argv[1]); s_vI = float(sys.argv[2])
    PR = learn_prior(); PR["p1"] = float(min(0.5, PR["rate"] * T / 86400))
    print("priori: taxa/dia %.3f  p1=%.3f  tipo %s  treino %s  teste %s" %
          (PR["rate"], PR["p1"], PR["ptype"], PR["train_sats"], PR["test_sats"]), flush=True)
    rng = np.random.default_rng(int(1000 * s_vI) + 5)
    test = PR["test"]; t0 = time.time()
    rows = []
    for it in range(N):
        Xref, P0, e, x0 = make_object(rng, s_vI)
        trk = Track(Xref, P0, e, PR)
        tau, s, _ = test[rng.integers(len(test))]
        y0 = truth_T(x0); yM = truth_T(x0, (tau, float(s)), rng)
        z0, S0 = uct(rng, y0); zM, SM = uct(rng, yM)
        # vizinho próximo (sem manobra): 1,5–3σ na métrica 6D do UKF
        mu = trk.w0 @ trk.m0
        Pu = np.einsum("k,kde->de", trk.w0, trk.P0 + np.einsum("kd,ke->kde", trk.m0 - mu, trk.m0 - mu))
        d6 = rng.standard_normal(6); d6 /= np.linalg.norm(d6)
        yN = y0 + rng.uniform(1.5, 3.0) * np.linalg.cholesky(Pu) @ d6
        zN, SN = uct(rng, yN)
        # objeto do campo: caixa RIC ±(5,50,15) km, ±(1,1,10) m/s
        R, I, C = ric_dirs(y0); Qm = np.stack([R, I, C], -1)
        off = np.r_[Qm @ (rng.uniform(-1, 1, 3) * [5e3, 50e3, 15e3]), Qm @ (rng.uniform(-1, 1, 3) * [1, 1, 10])]
        zF, SF = uct(rng, y0 + off)
        rows.append(dict(tau=tau, s=float(s), s0=trk.scores(z0, S0), sM=trk.scores(zM, SM),
                         sN=trk.scores(zN, SN), sF=trk.scores(zF, SF)))
        if (it + 1) % 50 == 0:
            print(f"  {it+1}/{N} ({time.time()-t0:.0f}s)", flush=True)
    json.dump(dict(prior={k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in PR.items() if k != "test"},
                   rows=rows), open(f"e8_rows_s{s_vI}.json", "w"),
              default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o))
    print("ok", time.time() - t0)
