"""Propagador numérico vetorizado (numba): zonais J2–J4 + arrasto atmosférico.
Cada estado i é integrado de t0[i] até t1[i] com RK4 de passo fixo `dt` (t0, t1 múltiplos de dt).
Arrasto: densidade exponencial por faixas (Vallado, Tab. 8-4), atmosfera girante,
coeficiente balístico B = Cd·A/m e fator multiplicativo de densidade exp(beta[i]) incerto."""
import numpy as np
from numba import njit, prange

MU = 3.986004418e14
RE = 6378.137e3
J2, J3, J4 = 1.08262668e-3, -2.53265649e-6, -1.61962159e-6
OMEGA_E = 7.292115e-5
B_COEF = 0.022            # Cd·A/m [m²/kg]  (Cd = 2.2, A/m = 0.01 m²/kg)
DT = 30.0

# faixas (altitude base [km], densidade base [kg/m³], escala [km])
_ATM = np.array([[400, 3.725e-12, 58.515], [450, 1.585e-12, 60.828], [500, 6.967e-13, 63.822],
                 [600, 1.454e-13, 71.835], [700, 3.614e-14, 88.667], [800, 1.170e-14, 124.64]])


@njit(cache=True, fastmath=True)
def _accel(x, beta, atm):
    rx, ry, rz, vx, vy, vz = x[0], x[1], x[2], x[3], x[4], x[5]
    r2 = rx * rx + ry * ry + rz * rz; r = np.sqrt(r2)
    z2r = rz * rz / r2
    k0 = -MU / (r2 * r)
    ax, ay, az = k0 * rx, k0 * ry, k0 * rz
    # J2
    c2 = -1.5 * J2 * MU * RE**2 / r2**2 / r
    ax += c2 * rx * (1 - 5 * z2r); ay += c2 * ry * (1 - 5 * z2r); az += c2 * rz * (3 - 5 * z2r)
    # J3
    c3 = -2.5 * J3 * MU * RE**3 / (r2**3 * r)
    t3 = 3 * rz - 7 * rz**3 / r2
    ax += c3 * rx * t3; ay += c3 * ry * t3; az += c3 * (6 * rz * rz - 7 * rz**4 / r2 - 0.6 * r2)
    # J4
    c4 = 15.0 / 8.0 * J4 * MU * RE**4 / (r2**3 * r)
    t4 = 1 - 14 * z2r + 21 * z2r * z2r
    ax += c4 * rx * t4; ay += c4 * ry * t4; az += c4 * rz * (5 - 70.0 / 3.0 * z2r + 21 * z2r * z2r)
    # arrasto
    h = (r - RE) / 1e3
    j = 0
    for k in range(atm.shape[0]):
        if h >= atm[k, 0]:
            j = k
    rho = atm[j, 1] * np.exp(-(h - atm[j, 0]) / atm[j, 2]) * np.exp(beta)
    wx, wy = vx + OMEGA_E * ry, vy - OMEGA_E * rx       # v − ω×r
    wz = vz
    vr = np.sqrt(wx * wx + wy * wy + wz * wz)
    cd = -0.5 * B_COEF * rho * vr
    return vx, vy, vz, ax + cd * wx, ay + cd * wy, az + cd * wz


@njit(parallel=True, cache=True, fastmath=True)
def _integrate(X, beta, t0, t1, dt, atm):
    n = X.shape[0]
    out = np.empty_like(X)
    for i in prange(n):
        x = X[i].copy()
        ns = int(round((t1[i] - t0[i]) / dt))
        k1 = np.empty(6); k2 = np.empty(6); k3 = np.empty(6); k4 = np.empty(6); y = np.empty(6)
        for s in range(ns):
            d = _accel(x, beta[i], atm)
            for q in range(6): k1[q] = d[q]; y[q] = x[q] + 0.5 * dt * d[q]
            d = _accel(y, beta[i], atm)
            for q in range(6): k2[q] = d[q]; y[q] = x[q] + 0.5 * dt * d[q]
            d = _accel(y, beta[i], atm)
            for q in range(6): k3[q] = d[q]; y[q] = x[q] + dt * d[q]
            d = _accel(y, beta[i], atm)
            for q in range(6):
                k4[q] = d[q]
                x[q] = x[q] + dt / 6.0 * (k1[q] + 2 * k2[q] + 2 * k3[q] + k4[q])
        out[i] = x
    return out


def integrate(X, beta, t0, t1, dt=DT):
    """X (...,6), beta (...) ou escalar, t0/t1 escalares ou arrays broadcastáveis a X[...,0]."""
    X = np.asarray(X, float); shp = X.shape[:-1]
    Xf = np.ascontiguousarray(X.reshape(-1, 6))
    bf = np.broadcast_to(np.asarray(beta, float), shp).reshape(-1).copy()
    a0 = np.broadcast_to(np.asarray(t0, float), shp).reshape(-1).copy()
    a1 = np.broadcast_to(np.asarray(t1, float), shp).reshape(-1).copy()
    return _integrate(Xf, bf, a0, a1, float(dt), _ATM).reshape(shp + (6,))
