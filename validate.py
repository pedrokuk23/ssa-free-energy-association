import numpy as np
from sb_core import *

A, B, n = cw_matrices(500)
T = 4000.0
q = 1e-10
Phi, W = phi_gramian(A, B, T)
G = q * W
rng = np.random.default_rng(1)

def randspd(scale_p, scale_v):
    L = rng.standard_normal((4, 4)) * 0.3 + np.eye(4)
    D = np.diag([scale_p, scale_p, scale_v, scale_v])
    S = D @ L @ L.T @ D
    return 0.5 * (S + S.T)

# 1) KL = 0 quando a marginal final é exatamente a propagação da referência
m0 = rng.standard_normal(4) * [100, 100, .1, .1]
S0 = randspd(50, .05)
m1 = Phi @ m0
S1 = Phi @ S0 @ Phi.T + G
print("KL consistente (deve ser ~0):", sb_kl(m0, S0, m1, S1, Phi, G))

# 2) forma fechada vs otimização numérica (caso bem condicionado: q maior)
Gb = 1e-6 * W
for i in range(3):
    m0 = rng.standard_normal(4) * [100, 100, .1, .1]
    m1 = Phi @ m0 + rng.standard_normal(4) * [50, 50, .05, .05]
    S0 = randspd(30, .03); S1 = randspd(30, .03)
    kc = sb_kl(m0, S0, m1, S1, Phi, Gb)[0]
    kn = sb_kl_numeric(m0, S0, m1, S1, Phi, Gb, seed=i)
    print(f"fechada={kc:.6f}  numérica={kn:.6f}  rel.err={(kn-kc)/abs(kc):.2e}")

# 3) identidade: custo MAP = 1/2 Mahalanobis da inovação; soma das partes
r = smoother_decomp(m0, S0, m1, S1, Phi, G)
print("total, c0+cu+c1:", r["total"], r["c0"] + r["cu"] + r["c1"])
