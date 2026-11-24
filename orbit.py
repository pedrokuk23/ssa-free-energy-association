"""Propagador vetorizado: dois corpos + J2 secular (taxas de Brouwer de 1a ordem).
Estados (..., 6) em ECI [m, m/s]. Rápido o bastante para ~1e6 partículas."""
import numpy as np

MU = 3.986004418e14
RE = 6378.137e3
J2 = 1.08262668e-3


def rv2coe(X):
    r, v = X[..., :3], X[..., 3:]
    rn = np.linalg.norm(r, axis=-1); vn2 = np.sum(v * v, -1)
    h = np.cross(r, v); hn = np.linalg.norm(h, axis=-1)
    a = 1.0 / (2.0 / rn - vn2 / MU)
    ev = (np.cross(v, h) / MU) - r / rn[..., None]
    e = np.linalg.norm(ev, axis=-1)
    i = np.arccos(np.clip(h[..., 2] / hn, -1, 1))
    nvec = np.stack([-h[..., 1], h[..., 0], np.zeros_like(hn)], -1)
    nn = np.linalg.norm(nvec, axis=-1)
    raan = np.arctan2(nvec[..., 1], nvec[..., 0])
    # argumento do perigeu e anomalia verdadeira
    cw = np.sum(nvec * ev, -1) / (nn * e)
    sw = np.sum(np.cross(nvec, ev) * h, -1) / (nn * e * hn)
    w = np.arctan2(sw, cw)
    cn = np.sum(ev * r, -1) / (e * rn)
    sn = np.sum(np.cross(ev, r) * h, -1) / (e * rn * hn)
    nu = np.arctan2(sn, cn)
    E = 2 * np.arctan2(np.sqrt(1 - e) * np.sin(nu / 2), np.sqrt(1 + e) * np.cos(nu / 2))
    M = E - e * np.sin(E)
    return a, e, i, raan, w, M


def kepler(M, e, it=8):
    E = M + e * np.sin(M)
    for _ in range(it):
        E = E - (E - e * np.sin(E) - M) / (1 - e * np.cos(E))
    return E


def coe2rv(a, e, i, raan, w, M):
    E = kepler(np.mod(M, 2 * np.pi), e)
    cE, sE = np.cos(E), np.sin(E)
    b = a * np.sqrt(1 - e * e)
    xp, yp = a * (cE - e), b * sE
    rr = a * (1 - e * cE)
    n = np.sqrt(MU / a**3)
    vxp, vyp = -a * n * sE / (1 - e * cE), b * n * cE / (1 - e * cE)
    cO, sO, cw, sw, ci, si = np.cos(raan), np.sin(raan), np.cos(w), np.sin(w), np.cos(i), np.sin(i)
    P = np.stack([cO * cw - sO * sw * ci, sO * cw + cO * sw * ci, sw * si], -1)
    Q = np.stack([-cO * sw - sO * cw * ci, -sO * sw + cO * cw * ci, cw * si], -1)
    r = xp[..., None] * P + yp[..., None] * Q
    v = vxp[..., None] * P + vyp[..., None] * Q
    return np.concatenate([r, v], -1)


def propagate(X, T):
    a, e, i, raan, w, M = rv2coe(X)
    p = a * (1 - e * e)
    n = np.sqrt(MU / a**3)
    k = 1.5 * J2 * (RE / p) ** 2
    ci = np.cos(i)
    Mdot = n * (1 + k * np.sqrt(1 - e * e) * (1 - 1.5 * np.sin(i) ** 2))
    Odot = -k * n * ci
    wdot = 0.5 * k * n * (5 * ci**2 - 1)
    return coe2rv(a, e, i, raan + Odot * T, w + wdot * T, M + Mdot * T)


def ric_frame(X):
    r, v = X[:3], X[3:]
    R = r / np.linalg.norm(r); C = np.cross(r, v); C /= np.linalg.norm(C); I = np.cross(C, R)
    return np.stack([R, I, C])          # linhas = eixos R, I, C


def ref_state(alt_km=550.0, e=0.002, inc=53.0, raan=0.3, w=1.0, M=0.0):
    return coe2rv(np.array(RE + alt_km * 1e3), np.array(e), np.radians(inc),
                  np.array(raan), np.array(w), np.array(M))
