"""Batched Gaussian log-density with per-matrix validation (audit revisions 3 and 4, 2026-11-26).

loggauss(r, S) returns log N(r; 0, S) over the broadcast batch dimensions of r (...,n) and S (...,n,n).

* Matrices that are finite and positive definite (Cholesky of the correlation-scaled matrix succeeds) are evaluated
  with exactly the same arithmetic as the original routine (LU solve and slogdet), so their values are bit-identical
  to the archived runs.
* Matrices with a non-finite entry (or a non-finite residual) are rejected: log-density -inf.
* Symmetry is checked in correlation scale, a = max_ij |S_ij - S_ji| / sqrt(|S_ii S_jj|) (revision 4, R4-01).
  a = 0: the matrix is used as given. 0 < a <= ASYM_TOL (rounding-level asymmetry): the symmetrized matrix
  (S + S^T)/2 is validated and evaluated. a > ASYM_TOL: the matrix is not a covariance and is rejected (-inf).
  The largest a seen is kept in STATS["asym_max"]; the positive-definiteness test is always applied to a
  symmetric matrix, so the result does not depend on which triangle a factorization reads.
* Matrices that are finite but not positive definite are repaired individually: symmetrized, scaled to correlation
  form, eigenvalues clipped at REPAIR_TOL times the largest absolute eigenvalue, then evaluated by Cholesky. They are never mixed with
  the other matrices of the batch, so the result for one matrix does not depend on the rest of the batch.
* Every rejection and repair is counted in STATS (and the worst correlation eigenvalue ratio is kept).

SAFE_LOGGAUSS=0 restores the original routine (sign of slogdet ignored; one regularization shared by the whole batch)
and is kept only to reproduce the archived runs; it is not recommended.
"""
import os, sys
import numpy as np

SAFE = os.environ.get("SAFE_LOGGAUSS", "1") == "1"
REPAIR_TOL = 1e-12
ASYM_TOL = float(os.environ.get("ASYM_TOL", "1e-8"))
STATS = dict(calls=0, matrices=0, rejected_nonfinite=0, repaired_nonpd=0, asymmetric=0, symmetrized=0,
             rejected_asymmetric=0, asym_max=0.0, lu_fallback=0)
_LOG2PI = np.log(2 * np.pi)
ASYM_SOURCES = {}   # caller 'file:line' -> [number of asymmetric matrices, largest asymmetry] (diagnostic of their origin)


def reset_stats():
    for k in STATS:
        STATS[k] = 0.0 if k == "asym_max" else 0
    ASYM_SOURCES.clear()


def loggauss_legacy(r, S):
    """Original routine of e11_numeric.py (kept for reproduction only)."""
    try:
        sol = np.linalg.solve(S, r[..., None])[..., 0]
    except np.linalg.LinAlgError:
        d = np.einsum("...ii->...i", S); eps = 1e-10 * np.maximum(np.abs(d).mean(-1), 1e-30)
        S = S + eps[..., None, None] * np.eye(S.shape[-1])
        sol = np.linalg.solve(S, r[..., None])[..., 0]
    _, ld = np.linalg.slogdet(S)
    return -0.5 * (np.sum(r * sol, -1) + ld + r.shape[-1] * _LOG2PI)


def _plain(r, S):
    sol = np.linalg.solve(S, r[..., None])[..., 0]
    _, ld = np.linalg.slogdet(S)
    return -0.5 * (np.sum(r * sol, -1) + ld + r.shape[-1] * _LOG2PI)


def _corr(S):
    d = np.sqrt(np.abs(np.einsum("...ii->...i", S)))
    d = np.where(d > 0, d, 1.0)
    return S / (d[..., :, None] * d[..., None, :]), d


def _pd_mask(S):
    """True where the correlation-scaled matrix has a Cholesky factor (per matrix)."""
    C, _ = _corr(S)
    try:
        np.linalg.cholesky(C)
        return np.ones(S.shape[:-2], bool)
    except np.linalg.LinAlgError:
        ok = np.empty(len(C), bool)
        for i, Ci in enumerate(C):
            try:
                np.linalg.cholesky(Ci); ok[i] = True
            except np.linalg.LinAlgError:
                ok[i] = False
        return ok


def _repaired(r, S):
    S = 0.5 * (S + S.T)
    C, d = _corr(S)
    lam, V = np.linalg.eigh(C)
    lam = np.maximum(lam, REPAIR_TOL * max(np.abs(lam).max(), 1.0))   # floor relative to the largest |eigenvalue|
    Lc = np.linalg.cholesky((V * lam) @ V.T)
    u = np.linalg.solve(Lc, r / d)
    return -0.5 * (u @ u + 2 * np.sum(np.log(np.diag(Lc))) + 2 * np.sum(np.log(d)) + len(r) * _LOG2PI)


def loggauss(r, S):
    if not SAFE:
        return loggauss_legacy(r, S)
    r = np.asarray(r, float); S = np.asarray(S, float); n = S.shape[-1]
    sh = np.broadcast_shapes(r.shape[:-1], S.shape[:-2])
    rb = np.broadcast_to(r, sh + (n,)).reshape(-1, n)
    Sb = np.broadcast_to(S, sh + (n, n)).reshape(-1, n, n)
    STATS["calls"] += 1; STATS["matrices"] += len(Sb)
    out = np.empty(len(Sb))
    fin = np.isfinite(Sb).all((1, 2)) & np.isfinite(rb).all(1)       # finiteness first (before any factorization)
    out[~fin] = -np.inf; STATS["rejected_nonfinite"] += int(np.sum(~fin))
    idx = np.flatnonzero(fin)
    if len(idx):
        Sf = Sb[idx]
        d = np.sqrt(np.abs(np.einsum("...ii->...i", Sf))); d = np.where(d > 0, d, 1.0)
        a = (np.abs(Sf - np.swapaxes(Sf, 1, 2)) / (d[:, :, None] * d[:, None, :])).max((1, 2))
        STATS["asymmetric"] += int(np.sum(a > 0)); STATS["asym_max"] = max(STATS["asym_max"], float(a.max()))
        if (a > 0).any():
            f = sys._getframe(1); key = f"{os.path.basename(f.f_code.co_filename)}:{f.f_lineno}"
            c = ASYM_SOURCES.setdefault(key, [0, 0.0]); c[0] += int(np.sum(a > 0)); c[1] = max(c[1], float(a.max()))
        big = a > ASYM_TOL
        out[idx[big]] = -np.inf; STATS["rejected_asymmetric"] += int(np.sum(big))
        small = (a > 0) & ~big
        if small.any():                                                # symmetrize rounding-level asymmetry
            Sb = Sb.copy()                                             # never write into the caller's array
            Sb[idx[small]] = 0.5 * (Sb[idx[small]] + np.swapaxes(Sb[idx[small]], 1, 2))
            STATS["symmetrized"] += int(np.sum(small))
        keep = idx[~big]
        if len(keep):
            pd = _pd_mask(Sb[keep])
            good = keep[pd]
            if len(good):
                try:
                    if len(good) == len(Sb):
                        out[:] = _plain(rb, Sb)                  # common case: same call as the original routine
                    else:
                        out[good] = _plain(rb[good], Sb[good])   # numpy factorizes each matrix independently
                except np.linalg.LinAlgError:                    # exactly singular LU pivot of a PD matrix (not observed)
                    STATS["lu_fallback"] += 1
                    for i in good:                               # per matrix: only the failing one is repaired
                        try:
                            out[i] = _plain(rb[i], Sb[i])
                        except np.linalg.LinAlgError:
                            out[i] = _repaired(rb[i], Sb[i])
            for i in keep[~pd]:
                out[i] = _repaired(rb[i], Sb[i]); STATS["repaired_nonpd"] += 1
    return out.reshape(sh)
