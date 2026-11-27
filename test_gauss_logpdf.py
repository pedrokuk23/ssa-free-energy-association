"""Regression tests for gauss_logpdf.loggauss (audit revision 3, R01-R02). Run: python test_gauss_logpdf.py"""
import numpy as np
import gauss_logpdf as G

def ref(r, S):
    L = np.linalg.cholesky(S); u = np.linalg.solve(L, r)
    return -0.5 * (u @ u + 2 * np.log(np.diag(L)).sum() + len(r) * np.log(2 * np.pi))

def main():
    rng = np.random.default_rng(0); r = np.array([1.0, 0.0])
    # valid matrix: agrees with the Cholesky reference and is bit-identical to the original routine
    S = np.eye(2); assert abs(G.loggauss(r, S) - ref(r, S)) < 1e-14
    A = rng.standard_normal((500, 6, 6)); P = A @ np.swapaxes(A, 1, 2) + 1e-3 * np.eye(6); x = rng.standard_normal((500, 6))
    assert np.array_equal(G.loggauss(x, P), G.loggauss_legacy(x, P)), "valid batch must be bit-identical"
    # indefinite and negative-definite (even dimension: positive determinant) matrices are not accepted as given
    G.reset_stats()
    for D in ([-1.0, 1.0], [-1.0, -1.0]):
        v = G.loggauss(r, np.diag(D))   # repaired: negative correlation eigenvalues clipped at REPAIR_TOL
        assert np.isclose(v, ref(r, np.diag([1e-12, 1e-12 if D[1] < 0 else 1.0])), rtol=1e-12), v
        assert v != G.loggauss_legacy(r, np.diag(D)), "invalid matrix must not take the original value"
    assert G.STATS["repaired_nonpd"] == 2
    # non-finite matrix or residual is rejected
    assert G.loggauss(r, np.array([[np.nan, 0], [0, 1]])) == -np.inf
    assert G.loggauss(np.array([np.inf, 0]), np.eye(2)) == -np.inf
    # the auditors' counterexample: a valid but extreme matrix must not change when batched with a singular one
    Sb = np.array([np.diag([1e-20, 1.0]), np.diag([0.0, 1.0])]); rb = np.array([[1e-10, 0.0], [0.0, 0.0]])
    alone = G.loggauss(rb[0], Sb[0]); both = G.loggauss(rb, Sb)
    assert both[0] == alone and abs(alone - 20.6879738635) < 1e-8, (alone, both)
    # invariance to batch composition and permutation
    Pm = P.copy(); Pm[[3, 77, 400]] *= -1; Pm[123, 0, 0] = np.nan
    full = G.loggauss(x, Pm); perm = rng.permutation(500)
    assert np.array_equal(G.loggauss(x[perm], Pm[perm]), full[perm])
    ok = np.isfinite(Pm).all((1, 2)) & ~np.isin(np.arange(500), [3, 77, 400])
    assert np.array_equal(full[ok], G.loggauss_legacy(x[ok], Pm[ok])), "valid entries of a mixed batch unchanged"
    for i in range(0, 500, 50):
        assert full[i] == G.loggauss(x[i], Pm[i]) or not np.isfinite(full[i])
    # broadcasting as used by the scorers (r[:, None] against a stack of covariances)
    rr = rng.standard_normal((4, 6)); Pk = P[:3]
    assert G.loggauss(rr[:, None], Pk).shape == (4, 3)
    assert np.array_equal(G.loggauss(rr[:, None], Pk), G.loggauss_legacy(rr[:, None], Pk))
    # revision 4 (R4-01): asymmetry is measured in correlation scale and decides validity
    G.reset_stats()
    S_as = np.array([[1.0, 10.0], [0.0, 1.0]]); r1 = np.ones(2)
    assert G.loggauss(r1, S_as) == -np.inf and G.loggauss(r1, S_as.T) == -np.inf, "grossly asymmetric: rejected, transpose-invariant"
    assert G.STATS["rejected_asymmetric"] == 2 and G.STATS["asym_max"] == 10.0
    A = P[7].copy(); A0 = A.copy(); A[0, 1] *= 1 + 1e-13                       # rounding-level asymmetry
    v, vt = G.loggauss(x[7], A), G.loggauss(x[7], A.T)
    assert v == vt == G.loggauss(x[7], 0.5 * (A + A.T)), "small asymmetry: symmetrized matrix evaluated"
    assert abs(v - G.loggauss_legacy(x[7], A)) < 1e-9 * max(1, abs(v)) and G.STATS["symmetrized"] == 2
    Pa = P.copy(); Pa[5, 0, 1] *= 1 + 1e-13; Pa0 = Pa.copy(); fa = G.loggauss(x, Pa)
    assert np.array_equal(Pa, Pa0), "input array not modified"
    assert np.array_equal(np.delete(fa, 5), np.delete(G.loggauss_legacy(x, P), 5)), "other matrices unchanged"
    perm = rng.permutation(500); assert np.array_equal(G.loggauss(x[perm], Pa[perm]), fa[perm])
    # revision 4 (R4-02): batched inverse/solve in e11 regularize only the failing matrix
    import os; os.environ.setdefault("K1", "41"); import e11_numeric as X
    Mb = np.stack([P[0], np.diag([0.0, 1, 1, 1, 1, 1]), P[1]]); bb = rng.standard_normal((3, 6, 1))
    Xi = X._inv(Mb); Xs = X._solve(Mb, bb)
    assert np.array_equal(Xi[0], np.linalg.inv(P[0])) and np.array_equal(Xi[2], np.linalg.inv(P[1]))
    assert np.array_equal(Xs[0], np.linalg.solve(P[0], bb[0])) and np.array_equal(Xs[2], np.linalg.solve(P[1], bb[2]))
    assert G.STATS["inv_regularized"] == 1 and G.STATS["solve_regularized"] == 1
    # undefined evidence (all mixture components rejected) is declared an invalid hypothesis, not NaN
    o = X._invalid_guard({"learn": float(np.nan), "pH1_learn": float(np.nan), "UKF": 3.0})
    assert o["learn"] == np.inf and o["UKF"] == 3.0 and np.isnan(o["pH1_learn"]) and G.STATS["invalid_hypotheses"] == 1
    print("all tests passed", G.STATS)

if __name__ == "__main__":
    main()
