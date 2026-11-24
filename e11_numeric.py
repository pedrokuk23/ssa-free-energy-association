"""
E5 v3 — (a) UCT de tracklet só de ângulos (atribuível α, δ, α̇, δ̇ de estação óptica em solo) ou estado completo;
(b) associação conjunta: cada UCT é avaliada contra a trilha-alvo E contra a trilha de catálogo do próprio vizinho;
(c) refinamento contínuo de |Δv| dentro do bin (atualização linear-gaussiana de log|Δv|);
(d) incerteza de densidade do estimador configurável (SIG_BETA_EST) para o caso de descasamento.
A propagação (blocos no espaço de estados em T) é a mesma de e10_numeric; a medida é projetada por EKF em cada
bloco (componentes estreitas), com resíduo angular em [−π, π).
Env: USE_GPU, K1, NJ, MEAS=full|tracklet|tracklet2, MISMATCH, SIG_BETA_EST, TAG.
Fase D / pós-E6c (exploratório): EXTRA=1 acrescenta CTRLp (MAP linearizado com a priori aprendida) e IMM (2 modos,
varredura única); SIGNED=1 (in-track sempre posígrado na verdade + priori "learn_sgn" com P(+)=PSGN); COV_SCALE;
tracklet2 = dois atribuíveis simultâneos de estações distintas; contagem da regra 7σ do IPLF em CNT7.
"""
import os, sys, json, time
import numpy as np
from scipy.special import logsumexp
import e10_numeric as E
from e10_numeric import (T, DT, K0, K2, TJ, CEN, NB, SIG_LOG_BIN, P1_ISO_SIG, UCT_SIG, split7, prop7, ric, ric_cov,
                         truth, load_labels, fold_prior, make_priors, field_offset)
from orbit import ref_state
from e6_nonlinear import prior_cov

MEAS = os.environ.get("MEAS", "full")
SIG_BETA_TRUE = 0.2
SIG_BETA_EST = float(os.environ.get("SIG_BETA_EST", "0.2"))
E.SIG_BETA = SIG_BETA_EST                     # usado por Track (prior do estimador)
MISMATCH = E.MISMATCH
RE = 6378.137e3; OMEGA = 7.292115e-5
AS = np.pi / 180 / 3600
SIG_ANG = 0.7 * AS; SIG_RATE = 0.04 * AS      # atribuível de 8 obs. a 2" em 60 s

# ------------------------------------------------------------------ medida
def h_tracklet(x, st):
    """x (...,6) inercial; st = (r_s, v_s). Retorna (...,4) = α, δ, α̇, δ̇."""
    rs, vs = st
    p = x[..., :3] - rs; pd = x[..., 3:6] - vs
    rxy2 = p[..., 0]**2 + p[..., 1]**2; rho = np.sqrt(rxy2 + p[..., 2]**2); rxy = np.sqrt(rxy2)
    a = np.arctan2(p[..., 1], p[..., 0]); d = np.arcsin(p[..., 2] / rho)
    ad = (p[..., 0] * pd[..., 1] - p[..., 1] * pd[..., 0]) / rxy2
    rhod = np.sum(p * pd, -1) / rho
    dd = (pd[..., 2] * rho - p[..., 2] * rhod) / (rho * rxy)
    return np.stack([a, d, ad, dd], -1)

def wrap(r):
    r = r.copy(); r[..., 0] = (r[..., 0] + np.pi) % (2 * np.pi) - np.pi
    if r.shape[-1] == 8:
        r[..., 4] = (r[..., 4] + np.pi) % (2 * np.pi) - np.pi
    return r

CIRC_MEAN = os.environ.get("CIRC_MEAN", "1") == "1"
EQ_SQRT = os.environ.get("EQ_SQRT", "1") == "1"
EXTRA = os.environ.get("EXTRA", "0") == "1"
EXTRA2 = os.environ.get("EXTRA2", "0") == "1"   # revisão: IMM com a priori aprendida (IMMl: 2 modos; IMMd: 1+3 modos por direção)
SIGNED = os.environ.get("SIGNED", "0") == "1"; PSGN = float(os.environ.get("PSGN", "0.9"))
COV_SCALE = float(os.environ.get("COV_SCALE", "1"))
CNT7 = np.zeros(5)   # [ativas, recusas 7σ, chamadas ev, ativas relevantes (≤10 nats da melhor), recusas relevantes]

class Meas:
    """Projeta (m, P) de estado para o espaço de medida: ŷ, H (Jacobiana) — identidade para estado completo."""
    def __init__(self, kind, st=None):
        self.kind, self.st = kind, st
        sc4 = np.r_[1 / AS, 1 / AS, 10 / AS, 10 / AS]
        self.SC = (np.r_[np.full(3, 1e-3), np.ones(3)] if kind == "full" else
                   np.r_[sc4, sc4] if kind == "tracklet2" else sc4)

    def h(self, x):
        if self.kind == "full":
            return x[..., :6]
        if self.kind == "tracklet2":
            return np.concatenate([h_tracklet(x, self.st[0]), h_tracklet(x, self.st[1])], -1)
        return h_tracklet(x, self.st)

    def jac(self, m):
        if self.kind == "full":
            return np.broadcast_to(np.eye(6), m.shape[:-1] + (6, 6))
        st = np.r_[np.full(3, 1.0), np.full(3, 1e-3)]
        cols = []
        for i in range(6):
            dp = m.copy(); dm = m.copy(); dp[..., i] += st[i]; dm[..., i] -= st[i]
            cols.append(wrap(self.h(dp) - self.h(dm)) / (2 * st[i]))
        return np.stack(cols, -1)                                               # (...,4,6)

    def res(self, z, y):
        r = z - y
        return r if self.kind == "full" else wrap(r)

def _reg(M):
    d = np.einsum("...ii->...i", M); eps = 1e-10 * np.maximum(np.abs(d).mean(-1), 1e-30)
    return M + eps[..., None, None] * np.eye(M.shape[-1])

def _inv(M):
    """inversa; só em matriz singular, regularização mínima (E6c, emenda 3)."""
    try:
        return np.linalg.inv(M)
    except np.linalg.LinAlgError:
        return np.linalg.inv(_reg(M))

def _solve(M, b):
    try:
        return np.linalg.solve(M, b)
    except np.linalg.LinAlgError:
        return np.linalg.solve(_reg(M), b)

def loggauss(r, S):
    try:
        sol = np.linalg.solve(S, r[..., None])[..., 0]
    except np.linalg.LinAlgError:
        # E6c emenda 3: só quando alguma matriz do lote é singular (geometria degenerada rara de tracklet);
        # regularização mínima relativa à escala da diagonal; matrizes não singulares não mudam na prática
        d = np.einsum("...ii->...i", S); eps = 1e-10 * np.maximum(np.abs(d).mean(-1), 1e-30)
        S = S + eps[..., None, None] * np.eye(S.shape[-1])
        sol = np.linalg.solve(S, r[..., None])[..., 0]
    _, ld = np.linalg.slogdet(S)
    return -0.5 * (np.sum(r * sol, -1) + ld + r.shape[-1] * np.log(2 * np.pi))


def msq(P):
    if EQ_SQRT:     # symmetric (rotation-equivariant) square root (audit fix, 2026-11-24; EQ_SQRT=0 reproduces the original runs)
        lam, V = np.linalg.eigh(0.5 * (P + np.swapaxes(P, -1, -2))); lam = np.clip(lam, 0.0, None)
        return V * np.sqrt(lam)[..., None, :]
    d = np.sqrt(np.clip(np.einsum("...ii->...i", P), 1e-30, None)); C = P / (d[..., :, None] * d[..., None, :])
    lam, V = np.linalg.eigh(C); lam = np.clip(lam, 1e-12, None)
    return V * np.sqrt(lam)[..., None, :] * d[..., :, None]

def ev(meas, z, R, mT, PT, it=2):
    """log N(z; ·) por componente com linearização estatística ITERADA da posterior (IPLF) da medida em T.
    Retorna (l, A, Sy, r): A = Jacobiana estatística final, Sy = cov. preditiva, r = resíduo."""
    if meas.kind == "full":
        A = np.broadcast_to(np.eye(6), mT.shape[:-1] + (6, 6)); Sy = PT + R; r = z - mT
        return loggauss(r, Sy), A, Sy, r
    n = mT.shape[-1]; Wt = np.r_[0.0, np.full(2 * n, 1 / (2 * n))]
    m = mT.copy(); P = PT.copy(); Pi = _inv(PT); act = np.ones(mT.shape[:-1], bool); out = None; CNT7[2] += 1
    for _ in range(it):
        S = msq(n * P); St = np.swapaxes(S, -1, -2)
        pts = np.concatenate([m[..., None, :], m[..., None, :] + St, m[..., None, :] - St], -2)
        Hy = meas.h(pts); my = np.einsum("p,...pd->...d", Wt, Hy)
        if CIRC_MEAN:   # right ascension(s) averaged on the circle (audit fix, 2026-11-24; CIRC_MEAN=0 reproduces the original runs)
            for ia in ((0, 4) if Hy.shape[-1] == 8 else (0,)):
                my[..., ia] = np.arctan2(np.einsum("p,...p->...", Wt, np.sin(Hy[..., ia])), np.einsum("p,...p->...", Wt, np.cos(Hy[..., ia])))
        Dy = wrap(Hy - my[..., None, :])
        Pyy = np.einsum("p,...pd,...pe->...de", Wt, Dy, Dy); DX = pts - m[..., None, :]
        Cxy = np.einsum("p,...pd,...pe->...de", Wt, DX, Dy)
        A = np.einsum("...de,...df->...ef", Cxy, _inv(P)); b = my - np.einsum("...ij,...j->...i", A, m)
        Om = Pyy - np.einsum("...ij,...jl,...ml->...im", A, P, A); Om = 0.5 * (Om + np.swapaxes(Om, -1, -2))
        yp = np.einsum("...ij,...j->...i", A, mT) + b
        Sy = np.einsum("...ij,...jl,...ml->...im", A, PT, A) + Om + R; r = meas.res(z, yp)
        l = loggauss(r, Sy)
        if out is None:
            out = [l, A, Sy, r]
        else:
            k = act
            out[0] = np.where(k, l, out[0]); out[1] = np.where(k[..., None, None], A, out[1])
            out[2] = np.where(k[..., None, None], Sy, out[2]); out[3] = np.where(k[..., None], r, out[3])
        Kg = np.einsum("...ij,...mj,...ml->...il", PT, A, _inv(Sy)); mn = mT + np.einsum("...ij,...j->...i", Kg, r)
        Pn = PT - np.einsum("...ij,...jl,...lm->...im", Kg, A, PT); Pn = 0.5 * (Pn + np.swapaxes(Pn, -1, -2))
        d2 = np.einsum("...i,...ij,...j->...", mn - mT, Pi, mn - mT); ok = act & np.isfinite(d2) & (d2 < 49)
        CNT7[0] += act.sum(); CNT7[1] += (act & ~ok).sum()
        rel = out[0] >= np.max(out[0]) - 10.0                                   # componentes a menos de 10 nats da melhor
        CNT7[3] += (act & rel).sum(); CNT7[4] += (act & rel & ~ok).sum()
        m = np.where(ok[..., None], mn, m); P = np.where(ok[..., None, None], Pn, P); act = ok
    return tuple(out)

def proj(meas, m, P, H=None):
    if H is None:
        H = meas.jac(m)
    return meas.h(m), np.einsum("...ij,...jk,...lk->...il", H, P, H), H

class Track3(E.Track):
    """Mesma propagação de e10; guarda g = dY/dlog|Δv| para o refinamento contínuo."""
    def __init__(self, m6, P6, e6):
        super().__init__(m6, P6, e6)

    def scores(self, z, R, meas, priors, refine=True):
        it = int(os.environ.get("IPLF_IT", "2"))
        lk0, _, _, _ = ev(meas, z, R, self.m0, self.P0, it)
        l0 = logsumexp(lk0 + np.log(self.w0))
        mu = self.w0 @ self.m0
        Pu = np.einsum("k,kde->de", self.w0, self.P0 + np.einsum("kd,ke->kde", self.m0 - mu, self.m0 - mu))
        out = {"UKF": -float(ev(meas, z, R, mu, Pu, it)[0]), "GM": -l0}
        lh, Ah, Sh, r = ev(meas, z, R, self.H0_m, self.H0_P, it)                 # (K2,·) mesma linearização p/ controle
        D = meas.SC
        Gy = np.einsum("kij,kpaj->kpai", Ah, self.G)                            # (K2,J,3,dim)
        Wy = np.einsum("kpai,kpaj->kij", Gy, Gy) * (T / len(TJ))
        out["CTRL"] = float(min((D * r[k]) @ np.linalg.lstsq(Wy[k] * np.outer(D, D), D * r[k], rcond=1e-12)[0] for k in range(K2)))
        Cu = Sh[:, None] + P1_ISO_SIG**2 * np.einsum("kpai,kpaj->kpij", Gy, Gy)
        rr = (D * r)[:, None] * np.ones((1, len(TJ), 1))
        out["CTRLu"] = float(np.min(np.einsum("kjd,kjd->kj", rr, _solve(Cu * np.outer(D, D), rr[..., None])[..., 0])))
        llin = ev(meas, z, R, self.Lm, self.LP, it)[0]
        l0b = logsumexp(lh + np.log(self.w2))
        lb, Ab, Sb, rb = ev(meas, z, R, self.Bm, self.BP, it)
        base = np.log(self.w2)[:, None, None, None, None] - np.log(len(TJ)) + np.log(0.5)
        if EXTRA:
            p1e = priors["learn"][2]
            Qm = P1_ISO_SIG**2 * np.einsum("k,kjad,kjae->de", self.w2, self.G, self.G) / len(TJ)
            lu0 = ev(meas, z, R, mu, Pu, it)[0]; lu1 = ev(meas, z, R, mu, Pu + Qm, it)[0]
            out["IMM"] = -float(np.logaddexp(np.log1p(-p1e) + lu0, np.log(p1e) + lu1))
            if EXTRA2:
                pdl, ql, _ = priors["learn"]; ma = (ql * CEN ** 2).sum(-1)
                Qa = np.einsum("k,kjad,kjae->ade", self.w2, self.G, self.G) / len(TJ)
                luL = ev(meas, z, R, mu, Pu + np.einsum("a,a,ade->de", pdl, ma, Qa), it)[0]
                out["IMMl"] = -float(np.logaddexp(np.log1p(-p1e) + lu0, np.log(p1e) + luL))
                lud = [np.log(p1e * pdl[a]) + ev(meas, z, R, mu, Pu + ma[a] * Qa[a], it)[0] for a in range(3)]
                out["IMMd"] = -float(logsumexp(np.array([np.log1p(-p1e) + lu0] + lud, float)))
        for name, (pdir, q, p1) in priors.items():
            if name == "iso_lin":
                Pi = Sh[:, None] + P1_ISO_SIG**2 * np.einsum("kpai,kpaj->kpij", Gy, Gy)
                l1 = logsumexp(loggauss(r[:, None], Pi) + np.log(self.w2)[:, None]) - np.log(len(TJ))
            else:
                lw = base + np.log(pdir)[None, None, :, None, None] + np.log(q)[None, None, :, None, :]
                if name == "learn_sgn":   # P(+) = PSGN só para in-track (índice 1); R e C simétricos
                    W = np.full((3, 2), 0.5); W[1] = [PSGN, 1 - PSGN]
                    lw = lw - np.log(0.5) + np.log(W)[None, None, :, :, None]
                tot = lb + lw; l1 = logsumexp(tot)
                if name == "learn" and EXTRA:
                    out["CTRLp"] = -float(max(np.max(lh + np.log(self.w2)) + np.log1p(-p1), np.max(llin + lw) + np.log(p1)))
                if name == "learn":
                    lrl = logsumexp(llin + lw) - l0b
                    out["lin_learn"] = -(l0 + np.logaddexp(np.log1p(-p1), np.log(p1) + lrl))
                    out["pH1_lin_learn"] = float(1 / (1 + np.exp(np.clip(np.log1p(-p1) - np.log(p1) - lrl, -700, 700))))
                    post = np.exp(tot - l1)
                    out["p_dir"] = post.sum(axis=(0, 1, 3, 4)).tolist()
                    out["dv_hat"] = float(np.exp(np.sum(post.sum(axis=(0, 1, 2, 3)) * np.log(CEN))))
                    if refine:   # δ = log|Δv| − log c_b ~ N(0, σ_b²) dentro do bin; posterior linear-gaussiana
                        idx = np.nonzero(post > 1e-4 * post.max())
                        gy = np.einsum("nij,nj->ni", Ab[idx], self.g[idx])
                        sol = _solve(Sb[idx], rb[idx][..., None])[..., 0]
                        dlt = np.clip(SIG_LOG_BIN**2 * np.sum(gy * sol, -1), -2 * SIG_LOG_BIN, 2 * SIG_LOG_BIN)
                        pw = post[idx] / post[idx].sum()
                        out["dv_ref"] = float(np.exp(np.sum(pw * (np.log(CEN)[idx[4]] + dlt))))
            lr = l1 - l0b
            out[name] = -(l0 + np.logaddexp(np.log1p(-p1), np.log(p1) + lr))
            out["pH1_" + name] = float(1 / (1 + np.exp(np.clip(np.log1p(-p1) - np.log(p1) - lr, -700, 700))))
        return out

# g = dY/dlog|Δv| precisa ser guardado: recalcula a partir de Bm/BP não é possível; injeta no __init__ do pai
_orig_init = E.Track.__init__
def _init_keep_g(self, m6, P6, e6):
    _orig_init(self, m6, P6, e6)
_src = E.Track.__init__.__code__
def _patch():
    import inspect, textwrap
    src = textwrap.dedent(inspect.getsource(E.Track.__init__))
    src = src.replace("self.Bm, self.BP = mean, Pb", "self.Bm, self.BP = mean, Pb; self.g = g")
    ns = {}
    exec(compile(src, "<track_init>", "exec"), E.__dict__, ns)
    E.Track.__init__ = ns["__init__"]
_patch()

# ------------------------------------------------------------------ UCT
def station_for(y, rng):
    """estação em solo a 0–600 km do ponto subsatélite (alcance ≈ 550–820 km, elevação > 40°)."""
    u = y[:3] / np.linalg.norm(y[:3])
    a = rng.standard_normal(3); a -= a @ u * u; a /= np.linalg.norm(a)
    ang = rng.uniform(0, 600e3) / RE
    us = np.cos(ang) * u + np.sin(ang) * a
    rs = RE * us; vs = np.cross([0, 0, OMEGA], rs)
    return rs, vs

def make_uct(rng, y):
    if MEAS == "full":
        S = ric_cov(y, UCT_SIG); return y + np.linalg.cholesky(S) @ rng.standard_normal(6), S, Meas("full")
    def one(st):
        z = h_tracklet(y, st)
        sa = SIG_ANG / max(np.cos(z[1]), 0.2); sr = SIG_RATE / max(np.cos(z[1]), 0.2)
        Rm = np.diag([sa**2, SIG_ANG**2, sr**2, SIG_RATE**2])
        zz = z + np.sqrt(np.diag(Rm)) * rng.standard_normal(4)
        zz[0] = (zz[0] + np.pi) % (2 * np.pi) - np.pi
        return zz, Rm
    if MEAS == "tracklet2":
        st1 = station_for(y, rng); st2 = station_for(y, rng)
        z1, R1 = one(st1); z2, R2 = one(st2)
        Rm = np.zeros((8, 8)); Rm[:4, :4] = R1; Rm[4:, 4:] = R2
        return np.r_[z1, z2], Rm, Meas("tracklet2", (st1, st2))
    st = station_for(y, rng); m = Meas("tracklet", st)
    zz, Rm = one(st)
    return zz, Rm, m

if __name__ == "__main__":
    N = int(sys.argv[1]); s_vI = float(sys.argv[2]); seed = int(sys.argv[3])
    tag = os.environ.get("TAG", "v3")
    L = load_labels()
    lab = L[(L.orbit_class == "LEO") & L.delta_v.notna() & (L.delta_v > 1e-4)
            & L.maneuver_type.isin(["in-track", "cross-track", "radial"])]
    sats = sorted(lab.norad_id.unique())
    priors = {s: make_priors(fold_prior(L, s)) for s in sats}
    if SIGNED:
        for s in sats:
            priors[s]["learn_sgn"] = priors[s]["learn"]
    rng = np.random.default_rng(np.random.SeedSequence([seed, int(round(1000 * s_vI)), int(MISMATCH), MEAS == "tracklet"]))
    fn = (f"e11_rows_{tag}_{MEAS}_s{s_vI}_seed{seed}_K{K2}"
          f"{'_mis' if MISMATCH else ''}{'_be%.2f' % SIG_BETA_EST if SIG_BETA_EST != 0.2 else ''}"
          f"{'_J%d' % len(TJ) if len(TJ) != 12 else ''}{'_T%g' % (T / 86400) if T != 86400 else ''}"
          f"{'_cs%g' % COV_SCALE if COV_SCALE != 1 else ''}{'_sgn' if SIGNED else ''}{'_x' if EXTRA else ''}.json")
    meta = dict(N=N, s_vI=s_vI, seed=seed, K0=K0, K1=K2, J=len(TJ), meas=MEAS, mismatch=MISMATCH,
                sig_beta_est=SIG_BETA_EST, tag=tag)
    rows = []; t0 = time.time()
    for it in range(N):
        Xref = ref_state(M=rng.uniform(0, 2 * np.pi), raan=rng.uniform(0, 2 * np.pi))
        P0, e = prior_cov(Xref, s_vI); P0 = COV_SCALE**2 * P0; Lp0 = np.linalg.cholesky(P0)
        x0 = Xref + (1.5 if MISMATCH else 1.0) * Lp0 @ rng.standard_normal(6)
        beta = SIG_BETA_TRUE * rng.standard_normal(); betas = None
        if MISMATCH:
            b = [0.4 * rng.standard_normal()]
            for _ in range(7):
                b.append(0.7 * b[-1] + 0.4 * np.sqrt(1 - 0.49) * rng.standard_normal())
            betas = np.array(b)
        held = sats[rng.integers(len(sats))]
        g = lab[lab.norad_id == held].iloc[rng.integers((lab.norad_id == held).sum())]
        man = (g.maneuver_type, float(g.delta_v))
        tm = np.round(rng.uniform(0.05, 0.95) * T / DT) * DT; sgn = rng.choice([-1, 1])
        if SIGNED and man[0] == "in-track":
            sgn = 1
        y0 = truth(x0, beta, None, rng, betas=betas); yM = truth(x0, beta, man, rng, tm, sgn, betas)
        # vizinho catalogado: estado 1.5–3σ do alvo em t0, transportado pela dinâmica; sua trilha tem erro ~ N(0, P0)
        d6 = rng.standard_normal(6); d6 /= np.linalg.norm(d6)
        xN = x0 + rng.uniform(1.5, 3.0) * Lp0 @ d6
        yD = truth(xN, beta, None, rng, betas=betas)
        xN_hat = xN + Lp0 @ rng.standard_normal(6)
        trk = Track3(Xref, P0, e)
        P0n, en = prior_cov(xN_hat, s_vI); P0n = COV_SCALE**2 * P0n; trkN = Track3(xN_hat, P0n, en)
        z0, R0, m0 = make_uct(rng, y0); zM, RM, mM = make_uct(rng, yM)
        zD, RD, mD = make_uct(rng, yD); zF, RF, mF = make_uct(rng, y0 + field_offset(rng, y0))
        pr = priors[held]
        row = dict(sat=int(held), tau=man[0], s=man[1], tm=float(tm), sgn=int(sgn), p1=pr["learn"][2],
                   Z=trk.scores(z0, R0, m0, pr), M=trk.scores(zM, RM, mM, pr), D=trk.scores(zD, RD, mD, pr),
                   F=trk.scores(zF, RF, mF, pr, refine=False))
        # associação conjunta: as mesmas UCTs contra a trilha do vizinho
        row["Mn"] = trkN.scores(zM, RM, mM, pr, refine=False); row["Dn"] = trkN.scores(zD, RD, mD, pr, refine=False)
        for c in ("Z", "M", "D", "F", "Mn", "Dn"):
            row[c] = {k: (float("%.7g" % v) if isinstance(v, float) else v) for k, v in row[c].items()}
        rows.append(row)
        if (it + 1) % 50 == 0:
            print(f"{it+1}/{N} ({time.time()-t0:.0f}s)", flush=True)
            json.dump(dict(meta=dict(meta, complete=False, n_done=len(rows)), rows=rows), open(fn + ".partial", "w"))
    meta.update(nj=len(TJ), T_days=T / 86400, cov_scale=COV_SCALE, signed=SIGNED, psgn=PSGN, extra=EXTRA,
                cnt7=dict(active=int(CNT7[0]), refused=int(CNT7[1]), ev_calls=int(CNT7[2]), active_rel=int(CNT7[3]), refused_rel=int(CNT7[4])))
    json.dump(dict(meta=dict(meta, complete=True, n_done=len(rows), seconds=time.time() - t0), rows=rows), open(fn, "w"))
    if os.path.exists(fn + ".partial"):
        os.remove(fn + ".partial")
    print("ok", fn, time.time() - t0)
