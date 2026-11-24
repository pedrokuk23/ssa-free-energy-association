"""Integrador J2–J4 + arrasto em GPU (numba-cuda, FP64, um thread por estado).
Mesma interface e mesmo algoritmo (RK4 passo fixo) de numprop.integrate; cai para a CPU se não houver GPU."""
import math, numpy as np
if not hasattr(np, "row_stack"): np.row_stack = np.vstack  # numba-cuda 0.30 x numpy 2.5
from numba import cuda, float64
import numprop as _cpu
import forces3b as _F3
exec(_F3._src())
SUNMOON = cuda.jit(device=True, inline=True)(SUNMOON); THIRD = cuda.jit(device=True, inline=True)(THIRD)
set_forces = _cpu.set_forces

def _ephem_table(a0, a1):
    """Sol e Lua (mesmas fórmulas de forces3b) numa grade de 60 s cobrindo [min t0, max t1]; coluna 6 = t inicial."""
    if _cpu.FP[0] < 0.5:
        return np.zeros((2, 7))
    key = (float(np.min(np.minimum(a0, a1))), float(np.max(np.maximum(a0, a1))), tuple(_cpu.FP))
    if _TABC.get("k") == key:
        return _TABC["v"]
    lo = float(np.min(np.minimum(a0, a1))) - 120.0; hi = float(np.max(np.maximum(a0, a1))) + 120.0
    tg = np.arange(lo, hi + 60.0, 60.0); tab = np.zeros((len(tg), 7)); sm = np.empty(6)
    for i, t in enumerate(tg):
        _cpu.SUNMOON(t, _cpu.FP, sm); tab[i, :6] = sm
    tab[:, 6] = lo; _TABC["k"] = key; _TABC["v"] = tab
    return tab

_TABC = {}

MU, RE, J2, J3, J4 = _cpu.MU, _cpu.RE, _cpu.J2, _cpu.J3, _cpu.J4
OMEGA_E, B_COEF, DT = _cpu.OMEGA_E, _cpu.B_COEF, _cpu.DT
_ATM = _cpu._ATM


@cuda.jit(device=True, inline=True)
def _accel(x, beta, atm, d, t, fp, tab):
    rx = x[0]; ry = x[1]; rz = x[2]; vx = x[3]; vy = x[4]; vz = x[5]
    r2 = rx * rx + ry * ry + rz * rz; r = math.sqrt(r2)
    z2r = rz * rz / r2
    k0 = -MU / (r2 * r)
    ax = k0 * rx; ay = k0 * ry; az = k0 * rz
    c2 = -1.5 * J2 * MU * RE * RE / (r2 * r2 * r)
    ax += c2 * rx * (1 - 5 * z2r); ay += c2 * ry * (1 - 5 * z2r); az += c2 * rz * (3 - 5 * z2r)
    c3 = -2.5 * J3 * MU * RE * RE * RE / (r2 * r2 * r2 * r)
    t3 = 3 * rz - 7 * rz * rz * rz / r2
    ax += c3 * rx * t3; ay += c3 * ry * t3; az += c3 * (6 * rz * rz - 7 * rz * rz * rz * rz / r2 - 0.6 * r2)
    c4 = 15.0 / 8.0 * J4 * MU * RE * RE * RE * RE / (r2 * r2 * r2 * r)
    t4 = 1 - 14 * z2r + 21 * z2r * z2r
    ax += c4 * rx * t4; ay += c4 * ry * t4; az += c4 * rz * (5 - 70.0 / 3.0 * z2r + 21 * z2r * z2r)
    h = (r - RE) / 1e3
    j = 0
    for k in range(atm.shape[0]):
        if h >= atm[k, 0]:
            j = k
    rho = atm[j, 1] * math.exp(-(h - atm[j, 0]) / atm[j, 2]) * math.exp(beta)
    wx = vx + OMEGA_E * ry; wy = vy - OMEGA_E * rx; wz = vz
    vr = math.sqrt(wx * wx + wy * wy + wz * wz)
    cd = -0.5 * B_COEF * rho * vr
    d[0] = vx; d[1] = vy; d[2] = vz
    if fp[0] < 0.5:
        d[3] = ax + cd * wx; d[4] = ay + cd * wy; d[5] = az + cd * wz
    else:
        sm = cuda.local.array(6, float64); acc = cuda.local.array(3, float64)
        # efemérides Sol/Lua tabeladas a cada 60 s no hospedeiro (tab[0,0] = t inicial da grade), interpolação linear
        u = (t - tab[0, 6]) / 60.0; k = int(u)
        if k < 0: k = 0
        if k > tab.shape[0] - 2: k = tab.shape[0] - 2
        w = u - k
        for q in range(6):
            sm[q] = (1.0 - w) * tab[k, q] + w * tab[k + 1, q]
        THIRD(rx, ry, rz, beta, fp, sm, acc)
        d[3] = ax + acc[0]; d[4] = ay + acc[1]; d[5] = az + acc[2]


@cuda.jit
def _kernel(X, beta, t0, t1, dt, atm, fp, tab, out):
    i = cuda.grid(1)
    if i >= X.shape[0]:
        return
    x = cuda.local.array(6, float64); y = cuda.local.array(6, float64); d = cuda.local.array(6, float64)
    k1 = cuda.local.array(6, float64); k2 = cuda.local.array(6, float64); k3 = cuda.local.array(6, float64)
    for q in range(6):
        x[q] = X[i, q]
    b = beta[i]
    ns = int((t1[i] - t0[i]) / dt + 0.5)
    for s in range(ns):
        tt = t0[i] + s * dt
        _accel(x, b, atm, d, tt, fp, tab)
        for q in range(6):
            k1[q] = d[q]; y[q] = x[q] + 0.5 * dt * d[q]
        _accel(y, b, atm, d, tt + 0.5 * dt, fp, tab)
        for q in range(6):
            k2[q] = d[q]; y[q] = x[q] + 0.5 * dt * d[q]
        _accel(y, b, atm, d, tt + 0.5 * dt, fp, tab)
        for q in range(6):
            k3[q] = d[q]; y[q] = x[q] + dt * d[q]
        _accel(y, b, atm, d, tt + dt, fp, tab)
        for q in range(6):
            x[q] = x[q] + dt / 6.0 * (k1[q] + 2 * k2[q] + 2 * k3[q] + d[q])
    for q in range(6):
        out[i, q] = x[q]


_atm_dev = None


def integrate(X, beta, t0, t1, dt=DT):
    global _atm_dev
    if not cuda.is_available():
        return _cpu.integrate(X, beta, t0, t1, dt)
    X = np.asarray(X, float); shp = X.shape[:-1]
    Xf = np.ascontiguousarray(X.reshape(-1, 6))
    bf = np.ascontiguousarray(np.broadcast_to(np.asarray(beta, float), shp).reshape(-1))
    a0 = np.ascontiguousarray(np.broadcast_to(np.asarray(t0, float), shp).reshape(-1))
    a1 = np.ascontiguousarray(np.broadcast_to(np.asarray(t1, float), shp).reshape(-1))
    n = Xf.shape[0]
    if n == 0:
        return Xf.reshape(shp + (6,))
    if _atm_dev is None:
        _atm_dev = cuda.to_device(_ATM)
    dX, db, d0, d1 = cuda.to_device(Xf), cuda.to_device(bf), cuda.to_device(a0), cuda.to_device(a1)
    dout = cuda.device_array_like(dX)
    tpb = 128
    _kernel[(n + tpb - 1) // tpb, tpb](dX, db, d0, d1, float(dt), _atm_dev, cuda.to_device(_cpu.FP), cuda.to_device(_ephem_table(a0, a1)), dout)
    return dout.copy_to_host().reshape(shp + (6,))
