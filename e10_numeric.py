"""
E9 — Correlação UCT–catálogo com propagação NUMÉRICA (J2–J4 + arrasto, densidade incerta),
manobras reais e validação cruzada deixando um satélite de fora (13 folds).

Estado aumentado 7-D: [r, v, β], β = log do fator de densidade ~ N(0, 0.2²) (incerto no prior e na verdade).
H0: mistura K0=41 (divisão ao longo de v_I), cada componente propagada pelo UT (15 pontos).
Hipóteses de manobra: K2=11 componentes × J=6 épocas × 3 direções (R, I, C) × 2 sinais × 8 bins log de |Δv|;
cada bloco propagado pela média + diferenças finitas (colunas de Cholesky) — EKF por componente.
"""
import numpy as np, json, sys, time
import pandas as pd
from scipy.special import logsumexp
import os
if os.environ.get("USE_GPU") == "1":
    from numprop_gpu import integrate
else:
    from numprop import integrate
from orbit import ref_state
from e6_nonlinear import prior_cov

T = float(os.environ.get("T_DAYS", "1")) * 86400.0; DT = 30.0
A_REF = float(os.environ.get("A_REF", str(6378.137e3 + 550e3)))   # semieixo para espalhar as fases das épocas
K0 = 41
K2 = int(os.environ.get("K1", "11"))
NJ = int(os.environ.get("NJ", "12"))
def _epochs(J):
    """J épocas espalhadas no dia E com fase n(T−t) uniforme em [0, 2π) (cobre R/C em qualquer argumento de latitude)."""
    n = np.sqrt(3.986004418e14 / A_REF**3); Pn = 2 * np.pi / n
    base = np.linspace(0.06, 0.96, J) * T; perm = (np.arange(J) * 5) % J      # fases embaralhadas ao longo do dia
    t = []
    for j in range(J):
        phi = 2 * np.pi * perm[j] / J
        dt = ((phi - n * (T - base[j])) % (2 * np.pi)) / n                  # recua até a fase alvo (< 1 período)
        t.append(base[j] - dt if base[j] - dt > 0.02 * T else base[j] - dt + Pn)
    return np.sort(np.round(np.array(t) / DT) * DT)
TJ = _epochs(NJ)
SIG_BETA = 0.2
BETA0 = 0.0; P7X = None   # padrão: β centrado em zero e independente (comportamento original)
UCT_SIG = np.array([30, 100, 30, 0.05, 0.03, 0.05])
EDGES = np.logspace(-3, 1.1, 9); CEN = np.sqrt(EDGES[:-1] * EDGES[1:]); NB = len(CEN)
SIG_LOG_BIN = np.log(EDGES[1] / EDGES[0]) / np.sqrt(12)
P1_ISO_SIG = 0.3

# ------------------------------------------------------------------ priori por fold
def load_labels():
    L = pd.DataFrame(json.load(open(os.environ.get("LABELS", "data/maneuver_labels_v0.3.json"))))
    L["epoch"] = pd.to_datetime(L.epoch, utc=True, format="mixed")
    return L

def fold_prior(L, held):
    CL = os.environ.get("PRIOR_CLASS", "LEO").split(",")
    d = L[L.orbit_class.isin(CL) & L.delta_v.notna() & (L.delta_v > 1e-4)]
    tr = d[d.norad_id != held]
    q, cnt = {}, {}
    for tau in ["radial", "in-track", "cross-track"]:
        x = tr[tr.maneuver_type == tau].delta_v.clip(EDGES[0], EDGES[-1] * 0.999)
        h, _ = np.histogram(x, EDGES); q[tau] = (h + 0.5) / (h + 0.5).sum(); cnt[tau] = len(x) + 0.5
    tot = sum(cnt.values()); pdir = {k: v / tot for k, v in cnt.items()}
    allL = L[L.orbit_class.isin(CL) & (L.norad_id != held)]
    days = sum((g.epoch.max() - g.epoch.min()).total_seconds() / 86400 for _, g in allL.groupby("norad_id"))
    return dict(q=q, pdir=pdir, p1=float(len(allL) / days * T / 86400))

# ------------------------------------------------------------------ utilidades
def ric(X):
    r, v = X[..., :3], X[..., 3:6]
    R = r / np.linalg.norm(r, axis=-1, keepdims=True)
    C = np.cross(r, v); C /= np.linalg.norm(C, axis=-1, keepdims=True)
    return R, np.cross(C, R), C

def ric_cov(X, sig):
    R, I, C = ric(X); Q = np.stack([R, I, C], -1); Z = np.zeros((6, 6)); Z[:3, :3] = Q; Z[3:, 3:] = Q
    return Z @ np.diag(sig**2) @ Z.T

def split7(m, P, u6, K):
    if K == 1:
        return np.ones(1), m[None].copy(), P.copy()
    L = np.linalg.cholesky(P); u = L.T @ np.r_[u6, 0]; u /= np.linalg.norm(u)
    Lb = 4.0 if K > 9 else 2.5
    mu = np.linspace(-Lb, Lb, K); h = mu[1] - mu[0]; st = min(0.85, 0.75 * h)
    w = np.exp(-0.5 * mu**2 / (1 - st**2)); w /= w.sum()
    M = m[None] + (mu[:, None] * u[None]) @ L.T
    Pk = L @ (np.eye(7) - (1 - st**2) * np.outer(u, u)) @ L.T
    return w, M, Pk

def prop7(X7, t0, t1):
    """Propaga estados aumentados (...,7) — β é parâmetro."""
    Y = integrate(X7[..., :6], X7[..., 6], t0, t1, DT)
    return np.concatenate([Y, X7[..., 6:7]], -1)

# loggauss: batched Gaussian log-density with per-matrix validation (audit revision 3, R01-R02); see gauss_logpdf.py.
from gauss_logpdf import loggauss

# ------------------------------------------------------------------ trilha de catálogo
class Track:
    def __init__(self, m6, P6, e6):
        m = np.r_[m6, BETA0]; P = np.zeros((7, 7)); P[:6, :6] = P6; P[6, 6] = SIG_BETA**2
        if P7X is not None: P = P7X   # revisão (e22_od): β estimado e covariância 7x7 do ajuste
        # H0 por UT
        w, M, Pk = split7(m, P, e6, K0)
        S = np.linalg.cholesky(7 * Pk)
        pts = np.concatenate([M[:, None], M[:, None] + S.T[None], M[:, None] - S.T[None]], 1)  # (K0,15,7)
        Wt = np.r_[0.0, np.full(14, 1 / 14)]
        Y = prop7(pts, 0.0, T)[..., :6]
        self.w0 = w; self.m0 = np.einsum("j,kjd->kd", Wt, Y)
        D = Y - self.m0[:, None]; self.P0 = np.einsum("j,kjd,kje->kde", Wt, D, D)
        # hipóteses de manobra: componentes K2 propagadas pelo UT até t_j, impulso, UT até T
        w2, M2, Pk2 = split7(m, P, e6, K2)
        S2 = np.linalg.cholesky(7 * Pk2)
        pts2 = np.concatenate([M2[:, None], M2[:, None] + S2.T[None], M2[:, None] - S2.T[None]], 1)  # (K2,15,7)
        Wt = np.r_[0.0, np.full(14, 1 / 14)]
        Xj = prop7(pts2[:, None].repeat(len(TJ), 1), 0.0, TJ[None, :, None])    # (K2,J,15,7)
        mj = np.einsum("p,kjpd->kjd", Wt, Xj)
        R, I, C = ric(mj)
        dirs = np.stack([R, I, C], 2)                                           # (K2,J,3,3)
        sg = np.array([1.0, -1.0])
        dv = (sg[None, None, None, :, None, None] * CEN[None, None, None, None, :, None]
              * dirs[:, :, :, None, None, :])                                   # (K2,J,3,2,B,3)
        X = Xj[:, :, None, None, None].repeat(3, 2).repeat(2, 3).repeat(NB, 4).copy()  # (K2,J,3,2,B,15,7)
        X[..., 3:6] += dv[..., None, :]
        Xs = mj[:, :, None, None, None, None, :].repeat(3, 2).repeat(2, 3).repeat(NB, 4).copy()
        Xs[..., 3:6] += 1.05 * dv[..., None, :]                                 # média com |Δv|·1.05
        Xa = np.concatenate([X, Xs], -2)                                        # (...,16,7)
        tj = TJ[None, :, None, None, None, None]
        Yb = prop7(Xa, tj, T)[..., :6]
        Yu = Yb[..., :15, :]
        mean = np.einsum("p,...pd->...d", Wt, Yu)
        D = Yu - mean[..., None, :]
        Pb = np.einsum("p,...pd,...pe->...de", Wt, D, D)
        Xc = mj[:, :, None, None, None, :].repeat(3, 2).repeat(2, 3).repeat(NB, 4).copy(); Xc[..., 3:6] += dv
        Yc_ = prop7(Xc, TJ[None, :, None, None, None], T)[..., :6]
        g = (Yb[..., 15, :] - Yc_) / np.log(1.05)                               # dY/dlog|Δv|
        Pb = Pb + SIG_LOG_BIN**2 * np.einsum("...d,...e->...de", g, g)
        self.Bm, self.BP = mean, Pb
        self.w2 = w2
        # H0 na MESMA aproximação (K2, UT) — para a razão H1/H0 consistente
        Y0 = prop7(pts2, 0.0, T)[..., :6]
        self.H0_m = np.einsum("p,kpd->kd", Wt, Y0); D0 = Y0 - self.H0_m[:, None]
        self.H0_P = np.einsum("p,kpd,kpe->kde", Wt, D0, D0)
        # linearizado isotrópico + gramiano: G = dY/dΔv na média de cada (k,j)
        h = 0.01
        Xg = mj[:, :, None].repeat(4, 2).copy(); Xg[:, :, 1:, 3:6] += h * dirs
        Yg = prop7(Xg, TJ[None, :, None], T)[..., :6]
        G = (Yg[:, :, 1:] - Yg[:, :, :1]) / h                                   # (K2,J,3,6)
        self.iso_P = self.H0_P[:, None] + P1_ISO_SIG**2 * np.einsum("kjad,kjae->kjde", G, G)
        self.Wg = np.einsum("kjad,kjae->kde", G, G) * (T / len(TJ))
        self.G = G
        sg = np.array([1.0, -1.0])
        shift = (sg[None, None, None, :, None, None] * CEN[None, None, None, None, :, None] * G[:, :, :, None, None, :])  # (K2,J,3,2,B,6)
        self.Lm = self.H0_m[:, None, None, None, None, :] + shift
        gg = np.einsum("kjad,kjae->kjade", G, G)                                # (K2,J,3,6,6)
        self.LP = (self.H0_P[:, None, None, None, None] + (SIG_LOG_BIN * CEN)[None, None, None, None, :, None, None]**2
                   * gg[:, :, :, None, None])
        self.SC = np.r_[np.full(3, 1e-3), np.ones(3)]                            # escala km, m/s

    def logl_blocks(self, z, S):
        return loggauss(z - self.Bm, self.BP + S)                              # (K2,J,3,2,B)

    def scores(self, z, S, priors):
        """priors: dict nome -> (pdir[3], q[3,B], p1)."""
        l0 = logsumexp(loggauss(z - self.m0, self.P0 + S) + np.log(self.w0))
        mu = self.w0 @ self.m0
        Pu = np.einsum("k,kde->de", self.w0, self.P0 + np.einsum("kd,ke->kde", self.m0 - mu, self.m0 - mu))
        out = {"UKF": -loggauss(z - mu, Pu + S), "GM": -l0}
        r = z - self.H0_m
        D = self.SC
        out["CTRL"] = min((D * r[k]) @ np.linalg.solve(self.Wg[k] * np.outer(D, D), D * r[k]) for k in range(K2))
        Cu = self.H0_P[:, None] + S + P1_ISO_SIG**2 * np.einsum("kjad,kjae->kjde", self.G, self.G)
        out["CTRLu"] = float(np.min(np.einsum("kjd,kjd->kj", (D * r)[:, None] * np.ones((1, len(TJ), 1)),
                          np.linalg.solve(Cu * np.outer(D, D), ((D * r)[:, None] * np.ones((1, len(TJ), 1)))[..., None])[..., 0])))
        llin = loggauss(z - self.Lm, self.LP + S)                                 # (K2,J,3,2,B)
        l0b = logsumexp(loggauss(r, self.H0_P + S) + np.log(self.w2))          # H0 na aproximação K2
        lb = self.logl_blocks(z, S)
        base = np.log(self.w2)[:, None, None, None, None] - np.log(len(TJ)) + np.log(0.5)
        for name, (pdir, q, p1) in priors.items():
            if name == "iso_lin":
                l1 = logsumexp(loggauss(r[:, None], self.iso_P + S) + np.log(self.w2)[:, None]) - np.log(len(TJ))
            else:
                lw = base + np.log(pdir)[None, None, :, None, None] + np.log(q)[None, None, :, None, :]
                tot = lb + lw
                l1 = logsumexp(tot)
                if name == "learn":
                    l1l = logsumexp(llin + lw); lrl = l1l - l0b
                    out["lin_learn"] = -(l0 + np.logaddexp(np.log1p(-p1), np.log(p1) + lrl))
                    out["pH1_lin_learn"] = float(1 / (1 + np.exp(np.clip(np.log1p(-p1) - np.log(p1) - lrl, -700, 700))))
                    post = np.exp(tot - l1)
                    out["p_dir"] = post.sum(axis=(0, 1, 3, 4)).tolist()
                    out["dv_hat"] = float(np.exp(np.sum(post.sum(axis=(0, 1, 2, 3)) * np.log(CEN))))
            lr = l1 - l0b                                   # log-razão H1/H0 na mesma aproximação
            out[name] = -(l0 + np.logaddexp(np.log1p(-p1), np.log(p1) + lr))
            out["pH1_" + name] = float(1 / (1 + np.exp(np.clip(np.log1p(-p1) - np.log(p1) - lr, -700, 700))))
        return out

# ------------------------------------------------------------------ verdade e medidas
MISMATCH = os.environ.get("MISMATCH", "0") == "1"
def _seg_prop(x6, betas, t0, t1, seg=10800.0):
    """propaga com β por segmentos de 3 h (descasamento de densidade)."""
    t = t0; X = x6.copy()
    while t < t1 - 1e-6:
        tn = min(t1, (np.floor(t / seg + 1e-9) + 1) * seg)
        X = integrate(X, betas[int(t // seg)], t, tn, DT); t = tn
    return X

def truth(x0, beta, man, rng, tm=None, sgn=None, betas=None):
    if betas is None:
        betas = np.full(int(np.ceil(T / 10800.0)) + 1, beta)
    if man is None:
        return _seg_prop(x0, betas, 0.0, T)
    tau, s = man
    Xm = _seg_prop(x0, betas, 0.0, tm)
    R, I, C = ric(Xm[:6])
    d = {"radial": R, "in-track": I, "cross-track": C}[tau]
    Xm[3:6] += sgn * s * d
    return _seg_prop(Xm, betas, tm, T)

def uct(rng, y):
    S = ric_cov(y, UCT_SIG); return y + np.linalg.cholesky(S) @ rng.standard_normal(6), S

def make_priors(fp):
    uni = np.full(NB, 1 / NB)
    from scipy.stats import maxwell
    def mx(sig):
        qb = np.diff(maxwell.cdf(EDGES, scale=sig)) + 1e-12; return qb / qb.sum()
    order = ["radial", "in-track", "cross-track"]
    P = {"learn": (np.array([fp["pdir"][k] for k in order]), np.stack([fp["q"][k] for k in order]), fp["p1"]),
         "flat": (np.full(3, 1 / 3), np.stack([uni] * 3), fp["p1"]),
         "maxwell_0.3": (np.full(3, 1 / 3), np.stack([mx(0.3)] * 3), fp["p1"]),
         "maxwell_3": (np.full(3, 1 / 3), np.stack([mx(3.0)] * 3), fp["p1"]),
         "maxwell_0.03": (np.full(3, 1 / 3), np.stack([mx(0.03)] * 3), fp["p1"]),
         "iso_lin": (None, None, fp["p1"])}
    return P

def field_offset(rng, y0):
    """objeto de campo fisicamente coerente: posição relativa em RIC (±5, ±50, ±15 km) e velocidade relativa
    no referencial rotativo com deriva limitada (ẏ ≈ −2n x), convertida para inercial (v_rel = ρ̇ + ω×ρ)."""
    n = np.linalg.norm(np.cross(y0[:3], y0[3:])) / np.dot(y0[:3], y0[:3])
    x, yy, zc = rng.uniform(-1, 1, 3) * [5e3, 50e3, 15e3]
    xd, yd_, zd = rng.uniform(-1, 1, 3) * [1.0, 1.0, 10.0]
    yd = -2 * n * x + yd_
    R, I, C = ric(y0); Qm = np.stack([R, I, C], -1)
    return np.r_[Qm @ [x, yy, zc], Qm @ [xd - n * yy, yd + n * x, zd]]

if __name__ == "__main__":
    N = int(sys.argv[1]); s_vI = float(sys.argv[2]); seed = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    tag = os.environ.get("TAG", "v2")
    L = load_labels()
    lab = L[(L.orbit_class == "LEO") & L.delta_v.notna() & (L.delta_v > 1e-4)
            & L.maneuver_type.isin(["in-track", "cross-track", "radial"])]
    sats = sorted(lab.norad_id.unique())
    priors = {s: make_priors(fold_prior(L, s)) for s in sats}
    rng = np.random.default_rng(np.random.SeedSequence([seed, int(round(1000 * s_vI)), int(MISMATCH)]))
    fn = f"e10_rows_{tag}_s{s_vI}_seed{seed}_K{K2}_J{NJ}{'_mis' if MISMATCH else ''}.json"
    meta = dict(N=N, s_vI=s_vI, seed=seed, K0=K0, K1=K2, J=NJ, TJ=TJ.tolist(), mismatch=MISMATCH, tag=tag)
    rows = []; t0 = time.time()
    for it in range(N):
        Xref = ref_state(M=rng.uniform(0, 2 * np.pi), raan=rng.uniform(0, 2 * np.pi))
        P0, e = prior_cov(Xref, s_vI)
        Lp0 = np.linalg.cholesky(P0)
        scale = 1.5 if MISMATCH else 1.0
        x0 = Xref + scale * Lp0 @ rng.standard_normal(6); beta = SIG_BETA * rng.standard_normal()
        betas = None
        if MISMATCH:   # densidade verdadeira AR(1) a cada 3 h, σ marginal 0.4, ρ=0.7
            b = [0.4 * rng.standard_normal()]
            for _ in range(7):
                b.append(0.7 * b[-1] + 0.4 * np.sqrt(1 - 0.49) * rng.standard_normal())
            betas = np.array(b)
        trk = Track(Xref, P0, e)
        held = sats[rng.integers(len(sats))]
        g = lab[lab.norad_id == held].iloc[rng.integers((lab.norad_id == held).sum())]
        man = (g.maneuver_type, float(g.delta_v))
        tm = np.round(rng.uniform(0.05, 0.95) * T / DT) * DT; sgn = rng.choice([-1, 1])
        y0 = truth(x0, beta, None, rng, betas=betas); yM = truth(x0, beta, man, rng, tm, sgn, betas)
        z0, S0 = uct(rng, y0); zM, SM = uct(rng, yM)
        # vizinho A (antigo): 1.5–3σ na métrica gaussiana única no instante T
        mu = trk.w0 @ trk.m0
        Pu = np.einsum("k,kde->de", trk.w0, trk.P0 + np.einsum("kd,ke->kde", trk.m0 - mu, trk.m0 - mu))
        d6 = rng.standard_normal(6); d6 /= np.linalg.norm(d6)
        zN, SN = uct(rng, y0 + rng.uniform(1.5, 3.0) * np.linalg.cholesky(Pu) @ d6)
        # vizinho B (novo): 1.5–3σ no prior em t0, transportado pela dinâmica (fica sobre a densidade curva)
        d6b = rng.standard_normal(6); d6b /= np.linalg.norm(d6b)
        xN = x0 + rng.uniform(1.5, 3.0) * Lp0 @ d6b
        zD, SD = uct(rng, truth(xN, beta, None, rng, betas=betas))
        zF, SF = uct(rng, y0 + field_offset(rng, y0))
        pr = priors[held]
        rows.append(dict(sat=int(held), tau=man[0], s=man[1], tm=float(tm), sgn=int(sgn), p1=pr["learn"][2],
                         Z=trk.scores(z0, S0, pr), M=trk.scores(zM, SM, pr), N=trk.scores(zN, SN, pr),
                         D=trk.scores(zD, SD, pr), F=trk.scores(zF, SF, pr)))
        if (it + 1) % 50 == 0:
            print(f"{it+1}/{N} ({time.time()-t0:.0f}s)", flush=True)
            json.dump(dict(meta=dict(meta, complete=False, n_done=len(rows)), rows=rows), open(fn + ".partial", "w"))
    json.dump(dict(meta=dict(meta, complete=True, n_done=len(rows), seconds=time.time() - t0), rows=rows), open(fn, "w"))
    if os.path.exists(fn + ".partial"):
        os.remove(fn + ".partial")
    print("ok", fn, time.time() - t0)
