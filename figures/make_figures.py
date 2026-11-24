"""Figures of the manuscript and supplement, from the result files of this package.
Run from the package root:  python figures/make_figures.py   (writes figures/*.png)"""
import json, sys, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
import pandas as pd
import os; sys.path.insert(0, os.getcwd())

ink, ink2, grid = "#0b0b0b", "#52514e", "#e6e5e0"
plt.rcParams.update({"font.size": 9.5, "axes.edgecolor": ink2, "xtick.color": ink2, "ytick.color": ink2,
    "axes.grid": True, "grid.color": grid, "grid.linewidth": .6, "axes.spines.top": False,
    "axes.spines.right": False, "figure.facecolor": "white", "axes.facecolor": "white", "axes.axisbelow": True})
C = {"UKF": "#2a78d6", "GM": "#eb6834", "CTRL": "#1baf7a", "iso_lin": "#eda100", "learn": "#6250d6"}
LAB = {"UKF": "Single Gaussian (UKF)", "GM": "Gaussian-mixture evidence", "CTRL": "Control distance",
       "iso_lin": "Free energy, linearized maneuver", "learn": "Free energy, propagated maneuver"}


def fig_nonlinear():
    from e9_numeric import prop7, SIG_BETA, T
    from e6n_numeric import gm_logev
    from orbit import ref_state, ric_frame
    from e6_nonlinear import prior_cov
    S = json.load(open("e6n_summary.json")); sig = [k for k in S if k != "particle_check"]
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.9))
    rng = np.random.default_rng(2)
    Xref = ref_state(M=0.7, raan=1.1); P6, e = prior_cov(Xref, 1.0)
    m = np.r_[Xref, 0]; P = np.zeros((7, 7)); P[:6, :6] = P6; P[6, 6] = SIG_BETA**2
    X = m + rng.standard_normal((20000, 7)) @ np.linalg.cholesky(P).T
    Y = prop7(X, 0.0, T)[:, :3]; mT = prop7(m, 0.0, T)
    Rm = ric_frame(mT[:6]); D = (Y - mT[:3]) @ Rm.T / 1e3
    a = ax[0]
    a.scatter(D[:, 1], D[:, 0], s=1, color="#9b9a95", alpha=.35, rasterized=True, label="Particles (true density)")
    _, my, Py = gm_logev(mT[:3], m, P, e, 1)
    Pp = Rm @ (Py[0] - 50.0**2 * np.eye(3)) @ Rm.T / 1e6; c = Rm @ (my[0] - mT[:3]) / 1e3
    w, V = np.linalg.eigh(Pp[np.ix_([1, 0], [1, 0])]); t = np.linspace(0, 2 * np.pi, 200)
    for k, ls in [(1, "-"), (3, "--")]:
        E = V @ (np.sqrt(w)[:, None] * np.stack([np.cos(t), np.sin(t)])) * k
        a.plot(c[1] + E[0], c[0] + E[1], color=C["UKF"], lw=2, ls=ls, label=f"UKF {k}$\\sigma$")
    a.set(xlabel="Along-track [km]", ylabel="Radial [km]", title="(a) Predicted density, $\\sigma_{v_I}$ = 1 m/s")
    a.legend(fontsize=8, frameon=False, loc="lower center", markerscale=6)
    s = [float(k) for k in sig]
    MK = {"UKF": ("o", "#2a78d6", "Single Gaussian (UKF)"), "GM15": ("s", "#eb6834", "Mixture, K = 15"),
          "GM41": ("^", "#1baf7a", "Mixture, K = 41")}
    a = ax[1]
    for k, (mk, col, l) in MK.items():
        a.plot(s, [S[x]["auc"][k] for x in sig], marker=mk, ms=7, lw=2, color=col, label=l)
    a.set_xscale("log"); a.set(xlabel="Initial in-track velocity $\\sigma_{v_I}$ [m/s]", ylabel="AUC",
                               title="(b) Discrimination vs. 1.5–3$\\sigma$ neighbors")
    a.legend(fontsize=8, frameon=False, loc="upper left")
    a = ax[2]
    for k, (mk, col, l) in MK.items():
        a.plot(s, [max(S[x]["logev_err_median"][k], 1e-3) for x in sig], marker=mk, ms=7, lw=2, color=col, label=l)
    a.set_xscale("log"); a.set_yscale("log")
    a.set(xlabel="Initial in-track velocity $\\sigma_{v_I}$ [m/s]", ylabel="Median underestimation of log $p(z)$ [nats]",
          title="(c) Evidence error vs. K = 81 reference")
    a.legend(fontsize=8, frameon=False, loc="upper left")
    fig.tight_layout(); fig.savefig("figures/fig_nonlinear.png", dpi=200); print("fig_nonlinear ok")


def fig_e10(path="e10_summary.json"):
    S = json.load(open(path)); k5, k05 = "v2_s0.5_seed3_K41_J12", "v2_s0.05_seed3_K41_J12"
    C2 = {"UKF": "#2a78d6", "GM": "#eb6834", "CTRLu": "#1baf7a", "lin_learn": "#eda100", "learn": "#6250d6"}
    LB2 = {"UKF": "Single Gaussian (moment-matched)", "GM": "Gaussian-mixture evidence", "CTRLu": "Control distance (uncertainty-aware)",
           "lin_learn": "Free energy, linearized hypotheses", "learn": "Free energy, propagated hypotheses"}
    L = pd.DataFrame(json.load(open("data/maneuver_labels_v0.3.json")))
    d = L[(L.orbit_class == "LEO") & L.delta_v.notna() & (L.delta_v > 1e-4) & L.maneuver_type.isin(["in-track", "cross-track", "radial"])]
    edges = np.logspace(-3, 1.1, 9); cen = np.sqrt(edges[:-1] * edges[1:])
    fig, ax = plt.subplots(2, 2, figsize=(11.5, 8.6))
    a = ax[0, 0]
    for tau, col in [("in-track", "#2a78d6"), ("cross-track", "#eb6834")]:
        h, _ = np.histogram(d[d.maneuver_type == tau].delta_v.clip(edges[0], edges[-1] * .999), edges)
        a.step(cen, h / h.sum(), where="mid", color=col, lw=2, label=f"{tau} ({(d.maneuver_type == tau).mean()*100:.0f}% of maneuvers)")
    a.set_xscale("log"); a.set(xlabel="$|\\Delta v|$ [m/s]", ylabel="Fraction", title=f"(a) Operator-reported LEO maneuvers (13 satellites, n = {len(d)})")
    a.legend(fontsize=8, frameon=False, loc="upper left")
    a = ax[0, 1]; M = list(C2)
    cats = [(k5, "acc_big", "$|\\Delta v|\\geq$ 5 cm/s\n$\\sigma_{v_I}$ = 0.5"), (k05, "acc_big", "$|\\Delta v|\\geq$ 5 cm/s\n$\\sigma_{v_I}$ = 0.05"),
            (k5, "acc_cross", "Cross-track\n$\\sigma_{v_I}$ = 0.5"), (k05, "acc_cross", "Cross-track\n$\\sigma_{v_I}$ = 0.05")]
    x = np.arange(len(cats)); wb = 0.16
    for i, m in enumerate(M):
        v = np.array([S[k]["metrics"][m][f] for k, f, _ in cats])
        lo = np.array([S[k]["ci"][f"{f}:{m}"][0] for k, f, _ in cats]); hi = np.array([S[k]["ci"][f"{f}:{m}"][1] for k, f, _ in cats])
        a.bar(x + (i - 2) * wb, v, wb * .88, color=C2[m], label=LB2[m], yerr=[v - lo, hi - v], ecolor=ink2, capsize=2, error_kw=dict(lw=.8))
    a.set_xticks(x, [l for _, _, l in cats], fontsize=8); a.set_ylim(0.3, 1.15); a.set_yticks(np.arange(0.3, 1.01, 0.1))
    a.grid(axis="x", visible=False)
    a.set(ylabel="Acceptance of the maneuvered object", title="(b) Acceptance at 1% false acceptance (field objects)")
    a.legend(fontsize=7.2, frameon=False, loc="upper left", ncol=2)
    a = ax[1, 0]; Mt = S[k5]["metrics"]; CI = S[k5]["ci"]
    VL = {"learn": "Learned (operator data)", "flat": "Log-uniform, uniform direction", "maxwell_0.03": "Isotropic Gaussian, $\\sigma$ = 0.03",
          "maxwell_0.3": "Isotropic Gaussian, $\\sigma$ = 0.3", "maxwell_3": "Isotropic Gaussian, $\\sigma$ = 3",
          "iso_lin": "Isotropic $\\sigma$ = 0.3, linearized", "lin_learn": "Learned, linearized"}
    COLV = {"learn": "#6250d6", "flat": "#1baf7a", "maxwell_0.03": "#2a78d6", "maxwell_0.3": "#eda100", "maxwell_3": "#eb6834",
            "iso_lin": "#9b9a95", "lin_learn": "#c2a8f0"}
    for v, l in VL.items():
        lo, hi = CI[f"logloss:{v}"]; ylo, yhi = CI[f"auc_D_big:{v}"]
        a.errorbar(Mt[v]["logloss"], Mt[v]["auc_D_big"], xerr=[[Mt[v]["logloss"] - lo], [hi - Mt[v]["logloss"]]],
                   yerr=[[Mt[v]["auc_D_big"] - ylo], [yhi - Mt[v]["auc_D_big"]]],
                   fmt="o", ms=8, color=COLV[v], ecolor=COLV[v], elinewidth=1, capsize=2, label=l)
    base = S[k5]["base_logloss"]
    a.axvline(base, color=ink2, lw=1, ls="--")
    a.text(base, 0.99, " base rate only", transform=a.get_xaxis_transform(), fontsize=8, color=ink2, va="top")
    a.axhline(Mt["GM"]["auc_D_big"], color=ink2, lw=1, ls=":")
    a.text(0.152, Mt["GM"]["auc_D_big"] - 0.002, "no maneuver hypothesis", fontsize=8, color=ink2, ha="center", va="top")
    a.set(xlabel="Log-loss of $P$(maneuver | UCT) at the true rate (lower is better)",
          ylabel="AUC vs. neighbors ($|\\Delta v|\\geq$ 5 cm/s)", title="(c) Calibration vs. neighbor discrimination ($\\sigma_{v_I}$ = 0.5 m/s)")
    a.legend(fontsize=7.4, frameon=False, loc="best")
    a = ax[1, 1]; labs = ["<0.5", "0.5–5", "5–50", "≥50"]; xb = np.arange(4)
    for m, ls in [("GM", "-"), ("CTRLu", "-"), ("lin_learn", "-"), ("learn", "-")]:
        for k, mk, al in [(k5, "o", 1.0), (k05, "s", 0.55)]:
            v = [b[1] for b in S[k]["acc_bins"][m]]
            a.plot(xb, v, ls, marker=mk, color=C2[m], alpha=al, lw=1.6, ms=5,
                   label=f"{LB2[m]}, $\\sigma_{{v_I}}$ = {0.5 if k == k5 else 0.05}")
    a.set_xticks(xb, [f"{l}\n(n = {S[k5]['acc_bins']['GM'][i][0]}/{S[k05]['acc_bins']['GM'][i][0]})" for i, l in enumerate(labs)], fontsize=8)
    a.set(xlabel="True $|\\Delta v|$ [cm/s]", ylabel="Acceptance at 1% false acceptance", ylim=(0.3, 1.03),
          title="(d) Acceptance versus maneuver magnitude")
    a.legend(fontsize=6.6, frameon=False, loc="lower left", ncol=1)
    fig.tight_layout(); fig.savefig("figures/fig_learned.png", dpi=200); print("fig_e10 ok")


FOREST_RC = {"axes.grid": False, "axes.spines.top": True, "axes.spines.right": True, "axes.edgecolor": "k",
             "xtick.color": "k", "ytick.color": "k"}


def _forest(ax, F, order, labels, colors, mu, lo, hi, a=18.0, b=10.0):
    y = np.arange(len(order))[::-1] + 1.5
    for yy, k in zip(y, order):
        ax.scatter(F[k][0], yy, s=a * F[k][1] + b, color=colors[k], zorder=3)
    ax.set_yticks(list(y) + [0]); ax.set_yticklabels([f"{labels[k]} ({F[k][1]})" for k in order] + ["RE (HKSJ)"])
    ax.fill([lo, mu, hi, mu], [0, 0.45, 0, -0.45], color="#1f77b4", zorder=3)
    ax.axvline(0, color="k", lw=0.8); ax.grid(False)
    return y


def fig_forest_e6s():
    """Supplementary figure (E6s): per-satellite Prop - CTRLu, correct joint assignment, from e12K_summary.json."""
    S = json.load(open("e12K_summary.json"))
    code = {"SPOT2": "SPOT-2", "SPOT4": "SPOT-4", "SPOT5": "SPOT-5", "JASO1": "Jason-1", "JASO2": "Jason-2", "JASO3": "Jason-3",
            "ENVI1": "Envisat", "CRYO2": "CryoSat-2", "SARAL": "SARAL", "SEN3A": "Sentinel-3A", "SEN3B": "Sentinel-3B", "SEN6A": "Sentinel-6A"}
    L = json.load(open("data/maneuver_labels_v0.3.json")); name = {}
    for r in L:
        name.setdefault(str(r["norad_id"]), code.get(str(r["source_ref"]).split()[0], str(r["norad_id"])))
    plt.rcParams.update(FOREST_RC)
    fig, axs = plt.subplots(1, 2, figsize=(9, 5), sharey=True)
    for ax, meas, tit in zip(axs, ("full", "tracklet"), ("Full-state UCT", "Optical tracklet")):
        P = S[meas]["primary"]; F = P["forest"]; order = sorted(F, key=lambda k: name[k])
        _forest(ax, F, order, name, {k: "#4a4a4a" for k in F}, P["mu"], *P["ci"], a=8.0, b=22.0)
        ax.plot(P["pred"], [0, 0], color="#1f77b4", alpha=0.6, lw=1.2, zorder=2)
        ax.text(P["ci"][1] + 0.005, -0.4, f"{P['mu']:+.3f} [{P['ci'][0]:+.3f}, {P['ci'][1]:+.3f}]", color="#1f77b4", fontsize=8)
        ax.set_xlim(-0.12, 0.22); ax.set_ylim(-0.95, len(F) + 1.0); ax.set_title(tit); ax.set_xlabel("Correct joint assignment, Prop $-$ CTRLu")
    fig.tight_layout(); fig.savefig("figures/fig_forest.png", dpi=250); print("fig_forest ok")


def fig_forest_e6c():
    """Main-text figure (E6c): per-satellite Prop - CTRLu, from e17K_summary.json; regimes from e17sel_*.json."""
    import re
    P = json.load(open("e17K_summary.json"))["primary"]; F = P["forest"]
    reg = {str(e["sat"]): g for g in ("meo", "qzss", "leo") for e in json.load(open(f"e17sel_{g}.json"))["events"]}
    name = {}
    for r in json.load(open("data/maneuver_labels_v0.3.json")):
        s = str(r["norad_id"])
        if s in F and r.get("source_ref"):
            m = re.search(r"SVN(\d+)/PRN(\d+)", r["source_ref"]); q = re.search(r"(QZS-\w+)", r["source_ref"])
            if m: name[s] = f"GPS SVN{m.group(1)}"
            elif q: name[s] = q.group(1)
    name.update({"90001": "SWOT", "90002": "Sentinel-3C", "90003": "Sentinel-6B"})
    order = [k for g in ("meo", "qzss", "leo") for k in sorted([k for k in F if reg[k] == g],
             key=lambda k: (name[k][:3], int(re.sub(r"\D", "", name[k]) or 0)))]
    col = {"meo": "#4a4a4a", "qzss": "#c26a00", "leo": "#1f77b4"}; lab = {"meo": "GPS (MEO)", "qzss": "QZSS (IGSO/GEO)", "leo": "LEO"}
    plt.rcParams.update(FOREST_RC)
    fig, ax = plt.subplots(figsize=(6.2, 8.6))
    y = _forest(ax, F, order, name, {k: col[reg[k]] for k in F}, P["mu"], *P["ci"])
    ax.tick_params(axis="y", labelsize=7.5)
    ax.text(P["ci"][1] + 0.005, -0.1, f"{P['mu']:+.3f} [{P['ci'][0]:+.3f}, {P['ci'][1]:+.3f}]", color="#1f77b4", fontsize=8, va="center")
    for g in col: ax.scatter([], [], color=col[g], s=40, label=lab[g])
    ax.legend(loc="lower right", fontsize=7.5, frameon=False, bbox_to_anchor=(1.0, 0.04))
    ax.set_xlim(-0.13, 0.18); ax.set_ylim(-0.9, y[0] + 0.8)
    ax.set_xlabel("Correct joint assignment, Prop $-$ CTRLu\n(mean of full-state and tracklet UCTs)", fontsize=9)
    fig.tight_layout(); fig.savefig("figures/fig_forest_e6c.png", dpi=300); print("fig_forest_e6c ok")


if __name__ == "__main__":
    fig_nonlinear()          # E4, main text (e6n_summary.json)
    fig_e10()                # E5 Study A, main text (e10_summary.json)
    fig_forest_e6c()         # E6c, main text (e17K_summary.json)
    fig_forest_e6s()         # E6s, supplement (e12K_summary.json)
