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
    print("all tests passed", G.STATS)

if __name__ == "__main__":
    main()
