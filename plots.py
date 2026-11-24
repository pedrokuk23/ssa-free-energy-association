import json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from montecarlo_utils import roc

R = np.load("mc_results.npz"); S = json.load(open("mc_summary.json"))
C = {"maha": "#2a78d6", "ctrl": "#eb6834", "sb": "#1baf7a", "glr": "#eda100"}
LS = {"maha": "-", "ctrl": "--", "sb": ":", "glr": "-."}
MK = {"maha": "o", "ctrl": "s", "sb": "^", "glr": "D"}
LAB = {"maha": "Mahalanobis (inovação)", "ctrl": "Distância de controle L2",
       "sb": "KL da ponte de Schrödinger", "glr": "GLR H₁/H₀ (suavizador)"}
keys = ["maha", "ctrl", "sb", "glr"]
ink, ink2, grid = "#0b0b0b", "#52514e", "#e6e5e0"

plt.rcParams.update({"font.size": 10, "axes.edgecolor": ink2, "axes.labelcolor": ink,
                     "xtick.color": ink2, "ytick.color": ink2, "axes.grid": True,
                     "grid.color": grid, "grid.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb"})
fig, ax = plt.subplots(2, 2, figsize=(11, 8.6))

for a, pos, title, tag in [(ax[0, 0], R["E1p"], "(a) Associação — objeto verdadeiro sem manobra", "E1"),
                           (ax[0, 1], R["E2p"], "(b) Associação — objeto verdadeiro com Δv = 0,1 m/s", "E2")]:
    for i, k in enumerate(keys[:3]):
        fpr, tpr = roc(pos[:, i], R["E1n"][:, i])
        a.plot(fpr, tpr, LS[k], color=C[k], lw=2, label=f"{LAB[k]}  (AUC {S[tag][k]:.2f})")
    a.plot([0, 1], [0, 1], color=grid, lw=1)
    a.set(xlabel="Falsa associação (objeto vizinho aceito)", ylabel="Associação correta", title=title,
          xlim=(0, 1), ylim=(0, 1.01))
    a.title.set_fontsize(10.5); a.legend(loc="lower right", fontsize=8.5, frameon=False)

dvs = [float(x) for x in S["E3"].keys()]
a = ax[1, 0]
for k in keys:
    a.plot(dvs, [S["E3"][str(d) if str(d) in S["E3"] else repr(d)][k] for d in dvs], LS[k],
           marker=MK[k], ms=7, color=C[k], lw=2, label=LAB[k])
a.set_xscale("log"); a.axhline(0.5, color=grid, lw=1)
a.set(xlabel="|Δv| verdadeiro [m/s]", ylabel="AUC de detecção de manobra",
      title="(c) Detecção de manobra (mesmo objeto, Δv vs. sem manobra)", ylim=(0.45, 1.01))
a.title.set_fontsize(10.5); a.legend(loc="upper left", fontsize=8.5, frameon=False)

a = ax[1, 1]
med_int = [S["E4"][k]["med_int"] for k in S["E4"]]
med_l1 = [S["E4"][k]["med_l1"] for k in S["E4"]]
a.plot(dvs, dvs, color=ink2, lw=1, label="Ideal (estimado = verdadeiro)")
a.plot(dvs, med_l1, "-", marker="o", ms=7, color=C["maha"], lw=2, label="∫|u| dt (custo tipo combustível)")
a.plot(dvs, med_int, "--", marker="s", ms=7, color=C["ctrl"], lw=2, label="|∫u dt| (Δv vetorial)")
a.set_xscale("log"); a.set_yscale("log")
a.set(xlabel="|Δv| verdadeiro [m/s]", ylabel="Δv reconstruído pelo suavizador (mediana) [m/s]",
      title="(d) Reconstrução da manobra sob H₁")
a.title.set_fontsize(10.5); a.legend(loc="upper left", fontsize=8.5, frameon=False)

fig.suptitle("Protótipo CW planar — LEO 500 km, lacuna de 4000 s, 3000 ensaios por cenário",
             color=ink, fontsize=12, y=0.995)
fig.tight_layout()
fig.savefig("resultados_ponte_kalman.png", dpi=150)
print("ok")
