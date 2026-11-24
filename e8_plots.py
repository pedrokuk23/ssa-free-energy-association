import json, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
import pandas as pd
import e8_uct_learned as E

S = json.load(open("e8_summary.json"))
rows = json.load(open("e8_rows_s0.5.json"))["rows"]
PR = E.learn_prior()
ink, ink2, grid = "#0b0b0b", "#52514e", "#e6e5e0"
C5 = {"UKF": "#2a78d6", "GM": "#eb6834", "CTRL": "#1baf7a", "GM+iso": "#eda100", "GM+learn": "#6250d6"}
LAB = {"UKF": "UKF", "GM": "Evidência GM (Montilla)", "CTRL": "Distância de controle (Holzinger)",
       "GM+iso": "Energia livre, manobra linearizada (isotr.)", "GM+learn": "Energia livre, manobra propagada (aprendida)"}
plt.rcParams.update({"font.size": 9.5, "axes.edgecolor": ink2, "xtick.color": ink2, "ytick.color": ink2,
    "axes.grid": True, "grid.color": grid, "grid.linewidth": .6, "axes.spines.top": False,
    "axes.spines.right": False, "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "axes.axisbelow": True})
fig, ax = plt.subplots(2, 2, figsize=(11.5, 8.8))

# (a) priori aprendida vs teste
a = ax[0, 0]; cen = PR["centers"]; edges = np.logspace(-3, 1.1, 15)
te = pd.DataFrame(PR["test"], columns=["t", "dv", "n"])
for tau, col, mk in [("in-track", "#2a78d6", "o"), ("cross-track", "#eb6834", "s")]:
    a.step(cen, PR["q"][tau], where="mid", color=col, lw=2, label=f"{tau}: treino (priori, P={PR['ptype'][tau]:.2f})")
    h, _ = np.histogram(te[te.t == tau].dv.astype(float).clip(edges[0], edges[-1] * .999), edges)
    a.plot(cen, h / h.sum(), mk, color=col, ms=6, mfc="none", mew=1.6, label=f"{tau}: satélites de teste")
a.set_xscale("log"); a.set(xlabel="|Δv| [m/s]", ylabel="Fração", title="(a) Priori de manobra aprendida de rótulos reais (DORIS/IDS)")
a.title.set_fontsize(10); a.legend(fontsize=7.8, frameon=False, loc="upper left")

# (b) aceitação do objeto manobrado contra o campo, TPR@1%
a = ax[0, 1]; M = ["UKF", "GM", "CTRL", "GM+iso", "GM+learn"]
cats = [("dv>=0.05", "|Δv| ≥ 5 cm/s"), ("cross-track", "Cross-track"), ("in-track", "In-track (todas)")]
x = np.arange(len(cats)); wbar = 0.16
for i, m in enumerate(M):
    v = [S["0.5"][c]["TPR@1%_campo (manobrado)"][m] for c, _ in cats]
    a.bar(x + (i - 2) * wbar, v, wbar * 0.88, color=C5[m], label=LAB[m])
a.set_xticks(x, [l for _, l in cats]); a.set_ylim(0.7, 1.1); a.set_yticks(np.arange(0.7, 1.01, 0.05)); a.grid(axis="x", visible=False)
a.set(ylabel="Aceitação do objeto verdadeiro", title="(b) Objeto manobrado vs. objetos do campo (1% de falsa aceitação)")
a.title.set_fontsize(10); a.legend(fontsize=7.4, frameon=False, loc="upper left", ncol=2)

# (c) papel da forma da priori: calibração vs discriminação
a = ax[1, 0]; Cb = json.load(open("e8_abl_calib.json"))
VL = {"learn": "Aprendida (dados reais)", "flat": "Log-uniforme, direção uniforme", "maxwell_0.03": "Gaussiana isotr., σ=0,03",
      "maxwell_0.3": "Gaussiana isotr., σ=0,3", "maxwell_3": "Gaussiana isotr., σ=3"}
COLV = {"learn": "#6250d6", "flat": "#1baf7a", "maxwell_0.03": "#2a78d6", "maxwell_0.3": "#eda100", "maxwell_3": "#eb6834"}
off = {"learn": (8, -14), "flat": (8, 4), "maxwell_0.03": (8, -4), "maxwell_0.3": (8, -4), "maxwell_3": (-8, 6)}
for v, l in VL.items():
    a.scatter(Cb[v]["logloss"], Cb[v]["AUC_viz_dv05"], s=70, color=COLV[v], edgecolor="#fcfcfb", lw=1.5, zorder=3)
    a.annotate(l, (Cb[v]["logloss"], Cb[v]["AUC_viz_dv05"]), xytext=off[v], textcoords="offset points",
               fontsize=8, color=ink, ha="right" if v == "maxwell_3" else "left")
a.axvline(Cb["ref_logloss_const"], color=ink2, lw=1, ls="--")
a.text(Cb["ref_logloss_const"] + 0.002, 0.705, "só a taxa base", fontsize=8, color=ink2)
a.axhline(Cb["GM"]["AUC_viz_dv05"], color=ink2, lw=1, ls=":")
a.text(0.208, Cb["GM"]["AUC_viz_dv05"] + 0.006, "sem hipótese de manobra", fontsize=8, color=ink2, ha="right")
a.set(xlabel="Log-loss de P(manobra | UCT) na taxa real (menor = melhor calibrada)",
      ylabel="AUC vs. vizinhos próximos (|Δv| ≥ 5 cm/s)", xlim=(0.115, 0.23), ylim=(0.39, 0.72),
      title="(c) A forma da priori controla a calibração (σ_vI = 0,5 m/s)")
a.title.set_fontsize(10)

# (d) caracterização
a = ax[1, 1]
sel = [r for r in rows if r["s"] >= 0.05 and r["sM"]["pH1_learn"] > 0.5]
for tau, col, mk in [("in-track", "#2a78d6", "o"), ("cross-track", "#eb6834", "s")]:
    P = [r for r in sel if r["tau"] == tau]
    ok = [(r["sM"]["p_cross"] > 0.5) == (tau == "cross-track") for r in P]
    a.scatter([r["s"] for r in P], [r["sM"]["dv_hat"] for r in P], s=22, marker=mk, color=col, alpha=.75,
              edgecolor="#fcfcfb", lw=.5, label=f"{tau} (tipo correto {np.mean(ok)*100:.0f}%)")
lim = [0.04, 12]; a.plot(lim, lim, color=ink2, lw=1); a.set_xscale("log"); a.set_yscale("log")
a.set(xlim=lim, ylim=lim, xlabel="|Δv| verdadeiro [m/s]", ylabel="|Δv| estimado (posterior) [m/s]",
      title=f"(d) Caracterização das manobras detectadas (n={len(sel)})")
a.title.set_fontsize(10); a.legend(fontsize=8, frameon=False, loc="upper left")
fig.suptitle("Correlação UCT–catálogo após 1 dia (2 corpos + J2), manobras reais de satélites de teste",
             fontsize=11.5, color=ink, y=.995)
fig.tight_layout(); fig.savefig("resultados_priori_aprendida.png", dpi=150); print("ok")
