import json, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from e6_nonlinear import *
S = json.load(open("e6_summary.json")); sig = [float(k) for k in S]
C = {"UKF": "#2a78d6", "GM15": "#eb6834", "GM41": "#1baf7a"}
MK = {"UKF": "o", "GM15": "s", "GM41": "^"}
key = {"UKF": "UKF_maha", "GM15": "GM15_F", "GM41": "GM41_F"}
LAB = {"UKF": "UKF (gaussiana única)", "GM15": "Energia livre GM, K=15", "GM41": "Energia livre GM, K=41"}
ink, ink2, grid = "#0b0b0b", "#52514e", "#e6e5e0"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": ink2, "xtick.color": ink2, "ytick.color": ink2,
    "axes.grid": True, "grid.color": grid, "grid.linewidth": .6, "axes.spines.top": False,
    "axes.spines.right": False, "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb"})
fig, ax = plt.subplots(2, 2, figsize=(11, 8.6))

# (a) banana em RIC no instante T
rng = np.random.default_rng(2)
m0, P0, e, yT = make_case(rng, 1.0)
L = np.linalg.cholesky(P0); X = m0 + rng.standard_normal((20000, 6)) @ L.T
Y = propagate(X, T)[:, :3]; mT = propagate(m0, T)
Rm = ric_frame(mT); D = (Y - mT[:3]) @ Rm.T / 1e3
a = ax[0, 0]
a.scatter(D[:, 1], D[:, 0], s=1, color="#9b9a95", alpha=.35, rasterized=True, label="Partículas (densidade verdadeira)")
_, my, Py = gm_ukf_logev(yT[:3], m0, P0, e, 1)
Pp = Rm @ (Py[0] - SIG_M**2 * np.eye(3)) @ Rm.T / 1e6; c = Rm @ (my[0] - mT[:3]) / 1e3
w, V = np.linalg.eigh(Pp[np.ix_([1, 0], [1, 0])]); t = np.linspace(0, 2 * np.pi, 200)
for k in [1, 3]:
    E = V @ (np.sqrt(w)[:, None] * np.stack([np.cos(t), np.sin(t)])) * k
    a.plot(c[1] + E[0], c[0] + E[1], color=C["UKF"], lw=2, ls="-" if k == 1 else "--",
           label=f"UKF {k}σ")
a.set(xlabel="Along-track [km]", ylabel="Radial [km]",
      title="(a) Densidade prevista após 1 dia (σ_vI = 1 m/s)")
a.title.set_fontsize(10.5); a.legend(fontsize=8.5, frameon=False, loc="lower center", markerscale=6)

for a, fld, tt in [(ax[0, 1], "auc_agn", "(b) AUC — vizinho agnóstico (direção aleatória, 1,5–3σ UKF)"),
                   (ax[1, 0], "auc", "(c) AUC — vizinho em camada radial próxima")]:
    for m in ["UKF", "GM15", "GM41"]:
        a.plot(sig, [S[str(s)][fld][key[m]] for s in S], marker=MK[m], ms=7, lw=2, color=C[m], label=LAB[m])
    a.set_xscale("log"); a.set(xlabel="σ da velocidade along-track inicial [m/s]", ylabel="AUC", title=tt, ylim=(0.78, 1.005))
    a.title.set_fontsize(10.5); a.legend(fontsize=8.5, frameon=False, loc="lower right")

a = ax[1, 1]
for m in ["UKF", "GM15", "GM41"]:
    a.plot(sig, [-S[str(s)]["logev_err_median"][m] for s in S], marker=MK[m], ms=7, lw=2, color=C[m], label=LAB[m])
a.set_xscale("log"); a.set_yscale("symlog", linthresh=0.01)
a.set(xlabel="σ da velocidade along-track inicial [m/s]", ylabel="Subestimação mediana de log p(z) [nats]",
      title="(d) Erro de log p(z) vs. referência (GM K=121)", ylim=(0, 10))
a.title.set_fontsize(10.5); a.legend(fontsize=8.5, frameon=False, loc="lower right")
fig.suptitle("Dois corpos + J2, LEO 550 km / 53°, lacuna de 1 dia, fix de posição σ = 50 m, 1500 ensaios por ponto",
             fontsize=12, color=ink, y=.995)
fig.tight_layout(); fig.savefig("resultados_nao_linear.png", dpi=150); print("ok")
