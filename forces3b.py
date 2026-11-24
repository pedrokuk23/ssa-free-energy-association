"""Forças para MEO/GEO (E6c): Sol e Lua (efemérides de baixa precisão, Montenbruck & Gill 2000, §3.3.2; ~0,1-1 %
em distância/direção) e pressão de radiação solar 'cannonball' sem sombra, a = -P0*CrA/m*exp(beta)*(AU/d)^2*u.
Quadro: as posições J2000 são levadas ao equador/equinócio médio da data (precessão IAU-1976), compatível com o
quadro pseudo-inercial usado nos eventos (ITRF girado pelo GMST). FP = [modo, época J2000 (s, TT~GPS), CrA/m, P(9)].
modo 0: LEO (só J2-J4 + arrasto, como antes); modo 1: J2-J4 + Sol + Lua + SRP, sem arrasto."""
import math, numpy as np
GM_SUN, GM_MOON = 1.32712440018e20, 4.9028000661e12
AU, P0 = 1.495978707e11, 4.56e-6
EPS = math.radians(23.43929111)
AS2R = math.pi / (180 * 3600)

def precession(T):
    z_ = (2306.2181 * T + 0.30188 * T * T) * AS2R; z = (2306.2181 * T + 1.09468 * T * T) * AS2R; th = (2004.3109 * T - 0.42665 * T * T) * AS2R
    def Rz(a): c, s = math.cos(a), math.sin(a); return np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])
    def Ry(a): c, s = math.cos(a), math.sin(a); return np.array([[c, 0, -s], [0, 1, 0], [s, 0, c]])
    return Rz(-z) @ Ry(th) @ Rz(-z_)

def make_fp(mode, epoch_j2000_s=0.0, cram=0.02):
    P = precession(epoch_j2000_s / 86400.0 / 36525.0)
    return np.r_[float(mode), float(epoch_j2000_s), float(cram), P.reshape(-1)]

def _src():  # mesmo código para CPU (numba njit) e GPU (cuda device): texto único
    return '''
def SUNMOON(t, fp, out):
    T = (fp[1] + t) / 86400.0 / 36525.0
    d2r = 0.017453292519943295
    M = (357.5256 + 35999.049 * T) * d2r
    lam = (282.94 * d2r) + M + (6892.0 * math.sin(M) + 72.0 * math.sin(2 * M)) * 4.84813681109536e-06
    rs = (149.619 - 2.499 * math.cos(M) - 0.021 * math.cos(2 * M)) * 1e9
    xe = rs * math.cos(lam); ye = rs * math.sin(lam)
    ce = 0.9174820620691818; se = 0.3977771559319137
    s0 = xe; s1 = ye * ce; s2 = ye * se
    out[0] = fp[3] * s0 + fp[4] * s1 + fp[5] * s2
    out[1] = fp[6] * s0 + fp[7] * s1 + fp[8] * s2
    out[2] = fp[9] * s0 + fp[10] * s1 + fp[11] * s2
    L0 = (218.31617 + 481267.88088 * T - 1.3972 * T) * d2r
    l = (134.96292 + 477198.86753 * T) * d2r; lp = (357.52543 + 35999.04944 * T) * d2r
    F = (93.27283 + 483202.01873 * T) * d2r; D = (297.85027 + 445267.11135 * T) * d2r
    a2r = 4.84813681109536e-06
    lm = L0 + a2r * (22640 * math.sin(l) + 769 * math.sin(2 * l) - 4586 * math.sin(l - 2 * D) + 2370 * math.sin(2 * D)
                     - 668 * math.sin(lp) - 412 * math.sin(2 * F) - 212 * math.sin(2 * l - 2 * D) - 206 * math.sin(l + lp - 2 * D)
                     + 192 * math.sin(l + 2 * D) - 165 * math.sin(lp - 2 * D) + 148 * math.sin(l - lp) - 125 * math.sin(D)
                     - 110 * math.sin(l + lp) - 55 * math.sin(2 * F - 2 * D))
    bm = a2r * (18520 * math.sin(F + lm - L0 + a2r * (412 * math.sin(2 * F) + 541 * math.sin(lp))) - 526 * math.sin(F - 2 * D)
                + 44 * math.sin(l + F - 2 * D) - 31 * math.sin(-l + F - 2 * D) - 25 * math.sin(-2 * l + F)
                - 23 * math.sin(lp + F - 2 * D) + 21 * math.sin(-l + F) + 11 * math.sin(-lp + F - 2 * D))
    rm = (385000 - 20905 * math.cos(l) - 3699 * math.cos(2 * D - l) - 2956 * math.cos(2 * D) - 570 * math.cos(2 * l)
          + 246 * math.cos(2 * l - 2 * D) - 205 * math.cos(lp - 2 * D) - 171 * math.cos(l + 2 * D) - 152 * math.cos(l + lp - 2 * D)) * 1e3
    xe = rm * math.cos(lm) * math.cos(bm); ye = rm * math.sin(lm) * math.cos(bm); ze = rm * math.sin(bm)
    s0 = xe; s1 = ce * ye - se * ze; s2 = se * ye + ce * ze
    out[3] = fp[3] * s0 + fp[4] * s1 + fp[5] * s2
    out[4] = fp[6] * s0 + fp[7] * s1 + fp[8] * s2
    out[5] = fp[9] * s0 + fp[10] * s1 + fp[11] * s2

def THIRD(rx, ry, rz, beta, fp, sm, acc):
    acc[0] = 0.0; acc[1] = 0.0; acc[2] = 0.0
    for b in range(2):
        sx = sm[3 * b]; sy = sm[3 * b + 1]; sz = sm[3 * b + 2]
        gm = 1.32712440018e20 if b == 0 else 4.9028000661e12
        dx = sx - rx; dy = sy - ry; dz = sz - rz
        d3 = (dx * dx + dy * dy + dz * dz) ** 1.5; s3 = (sx * sx + sy * sy + sz * sz) ** 1.5
        acc[0] += gm * (dx / d3 - sx / s3); acc[1] += gm * (dy / d3 - sy / s3); acc[2] += gm * (dz / d3 - sz / s3)
    dx = rx - sm[0]; dy = ry - sm[1]; dz = rz - sm[2]; d = math.sqrt(dx * dx + dy * dy + dz * dz)
    k = 4.56e-6 * fp[2] * math.exp(beta) * (1.495978707e11 / d) ** 2 / d
    acc[0] += k * dx; acc[1] += k * dy; acc[2] += k * dz
'''
