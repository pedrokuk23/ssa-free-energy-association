"""Figures of the manuscript and supplement, from the result files of this package.
Run from the package root:  python figures/make_figures.py
Writes figures/<name>.pdf (vector, used in the manuscript) and figures/<name>.png (600 dpi preview).
Every figure is drawn at its printed size in the preprint (text width 390 pt = 5.42 in), so that no text is below 7 pt."""
import json, sys, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
import os; sys.path.insert(0, os.getcwd())

TW = 390 / 72.27                                       # text width of the preprint [in]
ink, ink2, grid = "#0b0b0b", "#52514e", "#e6e5e0"
BASE_RC = {"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
           "legend.fontsize": 7, "axes.edgecolor": ink2, "xtick.color": ink2, "ytick.color": ink2, "axes.linewidth": 0.6,
           "axes.grid": True, "grid.color": grid, "grid.linewidth": .5, "axes.spines.top": False, "axes.spines.right": False,
           "figure.facecolor": "white", "axes.facecolor": "white", "axes.axisbelow": True, "pdf.fonttype": 42,
           "lines.linewidth": 1.2, "lines.markersize": 4.5}
plt.rcParams.update(BASE_RC)
# one label, colour and marker per method, as in the tables
METH = {"UKF": ("UKF", "#2a78d6", "o"), "GM": ("GM", "#eb6834", "s"), "CTRLu": ("CTRLu", "#1baf7a", "D"),
        "lin_learn": ("Lin", "#b8860b", "v"), "learn": ("Prop", "#5b3fc4", "^")}


def save(fig, name):
    fig.savefig(f"figures/{name}.pdf"); fig.savefig(f"figures/{name}.png", dpi=600); plt.close(fig); print(name, "ok")


def fig_nonlinear():
    """Fig. 1 (E4): true predicted density versus single-Gaussian ellipses (numbers in Table E4)."""
    from e9_numeric import prop7, SIG_BETA, T
    from e6n_numeric import gm_logev
    from orbit import ref_state, ric_frame
    from e6_nonlinear import prior_cov
    fig, a = plt.subplots(figsize=(0.62 * TW, 2.35))
    rng = np.random.default_rng(2)
    Xref = ref_state(M=0.7, raan=1.1); P6, e = prior_cov(Xref, 1.0)
    m = np.r_[Xref, 0]; P = np.zeros((7, 7)); P[:6, :6] = P6; P[6, 6] = SIG_BETA**2
    X = m + rng.standard_normal((20000, 7)) @ np.linalg.cholesky(P).T
    Y = prop7(X, 0.0, T)[:, :3]; mT = prop7(m, 0.0, T)
    Rm = ric_frame(mT[:6]); D = (Y - mT[:3]) @ Rm.T / 1e3
    a.scatter(D[:, 1], D[:, 0], s=0.8, color="#6f6e69", alpha=.45, lw=0, rasterized=True, label="Particles (true density)")
    _, my, Py = gm_logev(mT[:3], m, P, e, 1)
    Pp = Rm @ (Py[0] - 50.0**2 * np.eye(3)) @ Rm.T / 1e6; c = Rm @ (my[0] - mT[:3]) / 1e3
    w, V = np.linalg.eigh(Pp[np.ix_([1, 0], [1, 0])]); t = np.linspace(0, 2 * np.pi, 200)
    for k, ls in [(1, "-"), (3, "--")]:
        E = V @ (np.sqrt(w)[:, None] * np.stack([np.cos(t), np.sin(t)])) * k
        a.plot(c[1] + E[0], c[0] + E[1], color=METH["UKF"][1], lw=1.4, ls=ls, label=f"UKF {k}$\\sigma$ ellipse")
    a.set(xlabel="Along-track [km]", ylabel="Radial [km]")
    a.legend(frameon=False, loc="lower center", markerscale=5)
    fig.tight_layout(pad=0.3); fig.savefig("figures/fig_nonlinear.pdf", dpi=600); fig.savefig("figures/fig_nonlinear.png", dpi=600)
    plt.close(fig); print("fig_nonlinear ok")


def fig_e10(path="e10_summary.json"):
    """Fig. 2 (E5, Study A)."""
    import pandas as pd
    S = json.load(open(path)); k5, k05 = "v2_s0.5_seed3_K41_J12", "v2_s0.05_seed3_K41_J12"
    SIG = [(k5, "0.5", True), (k05, "0.05", False)]          # (key, label, filled marker)
    L = pd.DataFrame(json.load(open("data/maneuver_labels_v0.3.json")))
    d = L[(L.orbit_class == "LEO") & L.delta_v.notna() & (L.delta_v > 1e-4) & L.maneuver_type.isin(["in-track", "cross-track", "radial"])]
    fig, ax = plt.subplots(2, 2, figsize=(TW, 4.9))
    # (a) maneuver magnitudes, cm/s
    a = ax[0, 0]; edges = np.logspace(-1, 3.1, 9); cen = np.sqrt(edges[:-1] * edges[1:])
    for tau, col, ls in [("in-track", "#2a78d6", "-"), ("cross-track", "#eb6834", "--")]:
        h, _ = np.histogram((100 * d[d.maneuver_type == tau].delta_v).clip(edges[0], edges[-1] * .999), edges)
        a.step(cen, h / h.sum(), where="mid", color=col, ls=ls, lw=1.4, label=f"{tau} ({(d.maneuver_type == tau).mean()*100:.0f}%)")
    a.set_xscale("log"); a.set(xlabel="$|\\Delta v|$ [cm/s]", ylabel="Fraction of maneuvers", title=f"(a) Operator-reported LEO maneuvers")
    a.set_ylim(0, 0.75); a.legend(frameon=False, loc="upper left")
    # (b) paired differences in acceptance, Prop - X, with cluster-bootstrap intervals
    a = ax[0, 1]; comp = ["CTRLu", "lin_learn", "GM", "UKF"]; outs = [("acc_big", "$|\\Delta v|\\geq$5 cm/s"), ("acc_cross", "cross-track")]
    y = 0; yt = []; yl = []
    for f, fl in outs:
        yt.append(y); yl.append(fl + ":"); y -= 1
        for X in comp:
            if any((S[k]["primary"].get(f"{f}:learn-{X}") or S[k]["secondary"].get(f"{f}:learn-{X}")) is None for k, _, _ in SIG):
                continue                                         # interval not computed for this pair
            for j, (k, sl, filled) in enumerate(SIG):
                q = S[k]["primary"].get(f"{f}:learn-{X}") or S[k]["secondary"][f"{f}:learn-{X}"]
                mu = S[k]["metrics"]["learn"][f] - S[k]["metrics"][X][f]; yy = y + (0.17 if j == 0 else -0.17)
                a.errorbar(mu, yy, xerr=[[mu - q[0]], [q[1] - mu]], fmt=METH[X][2], color=METH[X][1], mfc=METH[X][1] if filled else "white",
                           ecolor=METH[X][1], elinewidth=1, capsize=1.5, ms=4)
            yt.append(y); yl.append(f"Prop $-$ {METH[X][0]}"); y -= 1
        y -= 0.4
    a.axvline(0, color=ink, lw=0.7); a.set_yticks(yt, yl); a.grid(axis="y", visible=False); a.set_ylim(y + 0.6, 0.6)
    for t in a.get_yticklabels():
        if t.get_text().endswith(":"): t.set_fontstyle("italic"); t.set_color(ink2)
    a.tick_params(axis="y", length=0)
    a.set(xlabel="Difference in acceptance", title="(b) Paired differences")
    # (c) log-loss of P(maneuver | UCT) for several maneuver priors
    a = ax[1, 0]; Mt = S[k5]["metrics"]; CI = S[k5]["ci"]
    VL = {"learn": ("Learned, propagated (Prop)", "^", True), "lin_learn": ("Learned, linearized (Lin)", "v", False),
          "flat": ("Log-uniform, propagated", "s", True), "maxwell_0.03": ("Isotropic 0.03 m/s, propagated", "o", True),
          "maxwell_0.3": ("Isotropic 0.3 m/s, propagated", "o", True), "maxwell_3": ("Isotropic 3 m/s, propagated", "o", True),
          "iso_lin": ("Isotropic 0.3 m/s, linearized", "o", False)}
    order = sorted(VL, key=lambda v: Mt[v]["logloss"])
    for yy, v in enumerate(order[::-1]):
        lo, hi = CI[f"logloss:{v}"]; l, mk, filled = VL[v]; col = METH["learn"][1] if v in ("learn", "lin_learn") else ink2
        a.errorbar(Mt[v]["logloss"], yy, xerr=[[Mt[v]["logloss"] - lo], [hi - Mt[v]["logloss"]]], fmt=mk, color=col,
                   mfc=col if filled else "white", ecolor=col, elinewidth=1, capsize=1.5, ms=4.5)
    a.set_yticks(range(len(order)), [VL[v][0] for v in order[::-1]]); a.grid(axis="y", visible=False)
    base = S[k5]["base_logloss"]; a.axvline(base, color=ink2, lw=0.8, ls="--")
    a.text(base, len(order) - 0.55, " base rate", fontsize=7, color=ink2, va="top")
    a.set(xlabel="Log-loss of $P$(maneuver | UCT)", title="(c) Calibration by prior ($\\sigma_{v_I}$ = 0.5 m/s)")
    # (d) acceptance of the largest maneuvers
    a = ax[1, 1]; mm = ["GM", "CTRLu", "lin_learn", "learn"]
    for yy, X in enumerate(mm):
        for j, (k, sl, filled) in enumerate(SIG):
            n, v = S[k]["acc_bins"][X][3]
            a.plot(v, yy + (0.15 if j == 0 else -0.15), METH[X][2], color=METH[X][1], mfc=METH[X][1] if filled else "white", ms=5)
    a.set_yticks(range(len(mm)), [METH[X][0] for X in mm]); a.grid(axis="y", visible=False); a.set_xlim(0.25, 0.75)
    n5, n05 = S[k5]["acc_bins"]["GM"][3][0], S[k05]["acc_bins"]["GM"][3][0]
    a.set(xlabel="Acceptance", title="(d) $|\\Delta v|\\geq$50 cm/s")
    h = [plt.Line2D([], [], ls="", marker="o", color=ink2, label="$\\sigma_{v_I}$ = 0.5 m/s"),
         plt.Line2D([], [], ls="", marker="o", color=ink2, mfc="white", label="$\\sigma_{v_I}$ = 0.05 m/s")]
    fig.legend(handles=h, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(pad=0.4, h_pad=1.0, w_pad=1.0, rect=(0, 0.045, 1, 1)); save(fig, "fig_learned")


FOREST_RC = {"axes.grid": False, "axes.spines.top": True, "axes.spines.right": True, "axes.edgecolor": "k",
             "xtick.color": "k", "ytick.color": "k"}
RE_COL = "#1f5fa8"


def _se(key):
    try:
        return json.load(open("figures/forest_se.json"))[key]
    except (FileNotFoundError, KeyError):
        return None                                              # run figures/forest_se.py (needs the per-event files)


def _forest(ax, F, order, labels, style, P, se, area=9.0):
    """Points: per-satellite mean (area proportional to the number of events); bars: +-1.96 pooled within-satellite
    standard error; diamond: HKSJ random-effects estimate with its two-sided 95% interval."""
    y = np.arange(len(order))[::-1] + 1.5
    for yy, k in zip(y, order):
        mk, col = style[k]
        if se:
            ax.plot([F[k][0] - 1.96 * se[k][2], F[k][0] + 1.96 * se[k][2]], [yy, yy], color=col, lw=0.6, alpha=0.6, zorder=2)
        ax.scatter(F[k][0], yy, s=area * F[k][1], marker=mk, color=col, zorder=3, lw=0)
    ax.set_yticks(list(y) + [0]); ax.set_yticklabels([f"{labels[k]} ({F[k][1]})" for k in order] + ["RE (HKSJ)"])
    mu, (lo, hi) = P["mu"], P["ci"]
    ax.fill([lo, mu, hi, mu], [0, 0.45, 0, -0.45], color=RE_COL, zorder=3)
    ax.axvline(0, color="k", lw=0.7); ax.axvline(mu, color=RE_COL, lw=0.5, ls=":", zorder=1); ax.grid(False)
    ax.tick_params(axis="y", length=0)
    return y


def fig_forest_e6s():
    """Fig. S1 (E6s): per-satellite Prop - CTRLu, correct joint assignment, from e12K_summary.json."""
    S = json.load(open("e12K_summary.json"))
    code = {"SPOT2": "SPOT-2", "SPOT4": "SPOT-4", "SPOT5": "SPOT-5", "JASO1": "Jason-1", "JASO2": "Jason-2", "JASO3": "Jason-3",
            "ENVI1": "Envisat", "CRYO2": "CryoSat-2", "SARAL": "SARAL", "SEN3A": "Sentinel-3A", "SEN3B": "Sentinel-3B", "SEN6A": "Sentinel-6A"}
    name = {}
    for r in json.load(open("data/maneuver_labels_v0.3.json")):
        name.setdefault(str(r["norad_id"]), code.get(str(r["source_ref"]).split()[0], str(r["norad_id"])))
    with plt.rc_context(FOREST_RC):
        fig, axs = plt.subplots(1, 2, figsize=(TW, 3.3), sharey=True)
        for ax, meas, tit in zip(axs, ("full", "tracklet"), ("(a) Full-state UCT", "(b) Optical tracklet")):
            P = S[meas]["primary"]; F = P["forest"]; order = sorted(F, key=lambda k: name[k])
            _forest(ax, F, order, name, {k: ("o", "#3d3d3d") for k in F}, P, _se(f"e6s_{meas}"), area=3.0)
            ax.plot(P["pred"], [0, 0], color=RE_COL, alpha=0.5, lw=1.0, zorder=2)      # 95% prediction interval
            ax.text(P["ci"][1] + 0.02, 0, f"{P['mu']:+.3f} [{P['ci'][0]:+.3f}, {P['ci'][1]:+.3f}]", color=RE_COL, fontsize=7,
                    ha="left", va="center")
            ax.set_xlim(-0.35, 0.45); ax.set_ylim(-0.9, len(F) + 1.0); ax.set_title(tit)
            ax.set_xlabel("Difference in correct-assignment rate")
        fig.tight_layout(pad=0.4); save(fig, "fig_forest")


def fig_forest_e6c():
    """Fig. 3 (E6c): per-satellite Prop - CTRLu, from e17K_summary.json; regimes from e17sel_*.json."""
    import re
    # audit 2026-11-26: corrected tracklet scores (e27_circ_summary.json) when present; the analysis as frozen is drawn as a
    # second, open diamond (e17K_summary.json)
    circ = os.path.exists("e27_circ_summary.json")
    P = json.load(open("e27_circ_summary.json" if circ else "e17K_summary.json"))["primary"]; F = P["forest"]
    P0 = json.load(open("e17K_summary.json"))["primary"]
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
    sty = {"meo": ("o", "#3d3d3d"), "qzss": ("s", "#b35c00"), "leo": ("^", "#1f77b4")}
    lab = {"meo": "GPS (MEO)", "qzss": "QZSS (IGSO/GEO)", "leo": "LEO"}
    with plt.rc_context(FOREST_RC):
        fig, ax = plt.subplots(figsize=(0.66 * TW, 6.3))
        y = _forest(ax, F, order, name, {k: sty[reg[k]] for k in F}, P, _se("e6c"))
        ax.tick_params(axis="y", labelsize=7)
        for g in ("qzss", "leo"):                                  # thin separators between regimes
            first = [yy for yy, k in zip(y, order) if reg[k] == g][0]
            ax.axhline(first + 0.5, color="#bdbdbd", lw=0.5)
        ax.text(P["ci"][1] + 0.02, 0, f"{P['mu']:+.3f} [{P['ci'][0]:+.3f}, {P['ci'][1]:+.3f}]", color=RE_COL, fontsize=7,
                ha="left", va="center")
        if circ:                                                    # analysis as frozen (original tracklet scores)
            mu0, (lo0, hi0) = P0["mu"], P0["ci"]
            ax.fill([lo0, mu0, hi0, mu0], [-1, -0.55, -1, -1.45], facecolor="none", edgecolor=RE_COL, lw=0.8, zorder=3)
            ax.text(hi0 + 0.02, -1, f"{mu0:+.3f} [{lo0:+.3f}, {hi0:+.3f}]", color=RE_COL, fontsize=7, ha="left", va="center")
            ax.set_yticks(list(ax.get_yticks()) + [-1])
            ax.set_yticklabels([t.get_text() for t in ax.get_yticklabels()][:-1] + ["RE, as frozen"])
        for g in sty: ax.scatter([], [], marker=sty[g][0], color=sty[g][1], s=22, label=lab[g])
        ax.legend(loc="lower center", bbox_to_anchor=(0.45, 1.0), ncol=3, frameon=False, fontsize=7, handletextpad=0.3, columnspacing=0.8)
        ax.set_xlim(-0.35, 0.45); ax.set_ylim(-1.9 if circ else -0.9, y[0] + 0.8)
        ax.set_xlabel("Difference in correct joint-assignment rate,\nProp $-$ CTRLu (mean of the two UCT types)")
        fig.tight_layout(pad=0.4); save(fig, "fig_forest_e6c")


if __name__ == "__main__":
    fig_nonlinear()          # E4, main text (e6n_summary.json)
    fig_e10()                # E5 Study A, main text (e10_summary.json)
    fig_forest_e6c()         # E6c, main text (e17K_summary.json)
    fig_forest_e6s()         # E6s, supplement (e12K_summary.json)
