# Free-energy association of maneuvering space objects — code, plans and results

Companion package of the manuscript *"Free-energy association of maneuvering space objects: Relating statistical
evidence and control distance"* (P. Kukulka de Albuquerque, J. Cavalcanti, L. Galembeck), prepared for submission to *Acta Astronautica*.

It contains the estimators, every simulation and real-data pipeline, the dated analysis plans with their amendments,
the E6c freeze record, the per-event result files, and the scripts that regenerate every table and figure of the
paper and its supplement. `REPRODUCE.md` maps each table and figure to its script and result file.

## Quick check (CPU, a few minutes; from the Zenodo deposit, or a git clone with it extracted)

    pip install -r requirements.txt
    python verify_freeze.py          # E6c freeze record: 15 of 17 hashes reproduced, 2 explained (see below)
    python e17K_analyze.py           # E6c pre-registered primary: +0.0264 [0.0118, 0.0410], 35 satellites
    python e25_k3_analyze.py         # E6c exploratory re-scoring: CTRLp, IMM variants, real confusers, all eligible events
    python figures/forest_se.py      # per-satellite standard errors for the forest plots (needs the per-event files)
    python figures/make_figures.py   # the four figures, as vector PDF and 600-dpi PNG, written to figures/
                                     # (needs numba: Fig. 1 re-propagates 20 000 particles, about 15 s on a CPU)

All analysis scripts read only the JSON result files shipped here. Re-running the scoring itself needs the public
precise orbits (below) and, for practical run times, an NVIDIA GPU (`USE_GPU=1`, Numba CUDA); `USE_GPU=0` uses the
CPU propagator (several times slower); Amendment 2 of `PREREG_E6c.md` reports the GPU-CPU agreement.

## Angle fix in the tracklet evidence (audit of 24 November 2026)

A pre-submission audit found that `ev()` in `e11_numeric.py` averaged the right ascension of the sigma points
arithmetically and used a non-equivariant matrix square root, which made the evidence of optical-tracklet UCTs depend
on the orientation of the inertial frame. The effect is large for the single moment-matched Gaussian (UKF, IMM). For the mixture-based scores in LEO (Prop,
Lin, CTRLu) the median change is below 0.001 nat, but 10-17% of the target scores change by more than 0.1 nat, mostly in
the tails of the prediction (the superseded Study B files also predate the regularization rule of Amendment 3). After the two-day gaps of medium Earth
and geosynchronous orbits, however, the maneuver-hypothesis blocks of Prop, Lin and CTRLp span wide arcs of right
ascension, and their tracklet evidence in the original E6c runs is degenerate (median F of order -1e10 nats for MEO
tracklets); the E6c tracklet results for MEO and QZSS therefore come from the rerun.
`e11_numeric.py` now averages angles on the circle and builds the sigma points from the unscaled spectral factor V diag(sqrt(lambda)) of the
symmetrized covariance (paired points insensitive to eigenvector sign and order) (switches `CIRC_MEAN=1`,
`EQ_SQRT=1`, the defaults); `CIRC_MEAN=0 EQ_SQRT=0` reproduces the original runs. `run22_circ.sh` repeats every
tracklet scoring with the fix; full-state results are unaffected because `ev()` does not use either step for them.

The rerun finished on 25 November 2026 and its outputs replace the tracklet files under their original names
(`e11_rows_*tracklet*`, `e12_rows_tracklet_*`, `e12K_tracklet_*_cNof3.json`, `e17K3_*_tracklet_*`, `e11_rows_pd*`,
`e11_rows_od2_*` and the summaries `e11_summary.json`, `e12_summary_seed*.json`, `e12K_summary.json`, `e14_recal.json`,
`e19_pd_summary.json`, `e23_*.json`, `e25_summary.json`); the rerun logs are in `logs_circ/`. The original files are
kept in `superseded_pre_angle_fix/`. The E6c confirmatory scores `e17K_*` were not rerun: `e17K_analyze.py` still
reproduces the primary analysis as frozen (+0.026). The corrected E6c analysis, reported in the paper as a declared
sensitivity analysis, uses the rescored `e17K3_*` rows (full-state rows identical, tracklet rows corrected):

    python e27_circ_table.py         # corrected E6c: primary (+0.039), secondary tests, Table E6c (e27_circ_summary.json, e27_circ_table.json)
    python e20_circ.py               # corrected E6c sensitivity (meta-analytic models, offsets, dv, cap): e20_circ.json, c121_circ.txt
    python e28_dominant.py > e28_dominant.txt   # weight of the dominant hypothesis term; CTRLp-Lin decision agreement
    python e29_p1nav.py > e29_p1nav.txt         # primary with pi1 from navigation satellites only (+0.037)

The one-sided lower bound printed by these scripts (0.0265) is computed from the mean and interval already rounded to four
decimals; computed from unrounded values it is 0.026479, reported in the paper as +0.026.
P-values of 0.001 or more are rounded to four decimals; smaller ones keep three significant digits (no output prints
p = 0.0000). The sign-flip test is exact (b/2^k) for up to 20 satellites and otherwise uses 200 000 random sign vectors
with the Monte Carlo estimate (b+1)/(N+1) (audit revision 3, R08). Corrected E6c: one-sided t p = 2.95e-6, sign-flip
p = 1e-5; the paper reports both as p < 0.001.

The tracklet calibration of E6c and the near-confuser check (F2) of `e17K2_*` were not repeated with the corrected
scores (the rescoring stores neither the neighbours' own UCTs nor the F2 field); the paper reports them for full-state
UCTs only. `e12K_summary.json` was recomputed locally from the three corrected tracklet chunks (the summary written on
the server also counted the six superseded chunks).

## Gaussian log-density with per-matrix validation (audit revisions 3 and 4, 26 November 2026)

`gauss_logpdf.py` replaces the `loggauss` routine of `e10_numeric.py` and `e11_numeric.py`. The original routine ignored
the sign of `slogdet` (an indefinite matrix gave a finite, too large density) and, when one matrix of a batch was
singular, added the same regularization to every matrix of the batch. The new routine checks each matrix:

* non-finite matrices (or residuals) give -inf;
* symmetry is measured in correlation scale, a = max |S_ij - S_ji| / sqrt(|S_ii S_jj|): a = 0, the matrix is used as
  given; 0 < a <= 1e-8 (`ASYM_TOL`), the symmetrized matrix is evaluated; a > 1e-8, the matrix is rejected (-inf);
  the largest a and the calling lines are recorded (`STATS["asym_max"]`, `ASYM_SOURCES`);
* matrices whose correlation-scaled Cholesky factorization fails are repaired one by one (eigenvalues of the
  correlation matrix clipped at 1e-12 times the largest absolute eigenvalue);
* valid matrices are evaluated with the original arithmetic (LU solve and slogdet); an LU failure is handled per matrix.

In `e11_numeric.py`, `_inv` and `_solve` now regularize only the failing matrix of a batch, and a hypothesis whose
evidence is undefined (for example every mixture component rejected) gets free energy +inf instead of NaN. All
counters are in `gauss_logpdf.STATS`. `SAFE_LOGGAUSS=0` restores the original routine for reproduction only. The
frozen copies in `frozen_E6c/` are unchanged.

    python test_gauss_logpdf.py      # valid, indefinite, asymmetric, non-finite and mixed batches; batch-composition,
                                     # permutation and transposition invariance; per-matrix _inv/_solve; invalid hypothesis

Diagnostic checks (sampled, not an exhaustive rerun of the campaign):

    python e32_covariance_check.py . x 3,4,5,6,7,8 > e32_covariance_check.txt   # stress test, 48 scorings, six LEO objects (one run, ~3 min)
    bash run33.sh                    # E6c production settings, real events, both UCT types: e33_*.txt
    python e34_rounding_bound.py 0.5,1 > e34_rounding_bound.txt                  # bounds on the primary (seconds)

`e33_e6c_covariance_check.py` regenerates draws of `e17_scoreK3.py` with the E6c settings (K1 = 41, refine = False,
ric_cov UCT covariances), scores each draw with the original and the new routine, and reports, per method, the
unrounded target (M) and neighbour (Mn) scores of both routines and of the archived `e17K3_*` rows, the decisions
with the analysis rule (1 / 0.5 tie / 0) of both routines and of the archive, the counters above, and (with
`PERTURB_EPS`) the scores and decisions after scaling the predicted covariance by (1 + eps).

Results (`e33_r4_*.txt`): the two routines agree at the stored score precision (largest difference 2e-7 nat, on scores up to 1e6 nat) with no
change of decision; no matrix is rejected or repaired; the tracklet covariances have rounding-level asymmetry only
(at most 4e-10 in correlation scale, from line 170 of `e11_numeric.py`); no batch fallback is triggered.
Against the archive: With full-state UCTs (8 draws: LEO events 0, 8, 9 x2; MEO 4; QZSS 0; files `e33_r4_*_full.txt`) M and Mn of every method matched the archive at its stored precision (relative difference 0), no decision changed, and the 1e-15 perturbation changed nothing. With tracklets, the Prop ('learn') and Lin ('lin_learn') scores of 3 of the 13
draws differ from the archive (up to 1 and 2.7 nat) and change by similar amounts under eps = 1e-15: the hard gate of
the iterated posterior linearization (d^2 < 49 in `ev`) makes them discontinuous, so rounding differences between
computing environments can move them (likely cause, not confirmed). No Prop decision changed; the Lin decision of the
QZSS draw did. `e34_rounding_bound.py` shows that counting every tracklet draw with a Prop margin below 0.5 nat as
wrong (9% of the tracklet draws) removes the primary gain, which is a worst case, not a measured effect; with the
tracklet half set to zero the primary is +0.023 [0.012, 0.034] (full-state half alone: +0.046 [0.024, 0.068]).
The correlation-scaled condition numbers (up to about 3e10 in e32 and 5e9 in e33) indicate potential sensitivity to
floating-point errors; no uniform bound on the log-density error is claimed.

## Git repository and Zenodo deposit

The git repository holds the code, plans, freeze record, summaries and figure script (about 7 MB). The Zenodo deposit
holds the same files plus the per-event result files (`*_rows_*`, `*_cNofM.json`) and the detector scans
(`e15_scan_*`), about 160 MB in all. To run the analysis scripts from a git clone, extract the Zenodo archive into the
repository root.

## Contents

| Study (paper) | Plan | Scoring / simulation | Analysis | Results |
|---|---|---|---|---|
| E1–E3 (Supplement S1) | — | `montecarlo.py`, `e3_v2.py` | in script | `mc_summary.json`, `e3v2_summary.json` |
| E4 (Fig. 1, Table E4) | — | `e6n_numeric.py`, `e4_is.py`, `e4_is_005.py` | in script | `e6n_summary.json`, `e4_is_summary*.json` |
| E5 Study A | `PREREG_E5v2.md` | `e10_numeric.py` (`run3.sh`) | `e10_analyze.py` | `e10_rows_*`, `e10_summary.json` |
| E5 Study B | `PREREG_E5v3.md` | `e11_numeric.py` (`run4.sh`) | `e11_analyze.py` | `e11_rows_*`, `e11_summary.json` |
| E6a / E6b (DORIS) | `PREREG_E6.md`, `PREREG_E6b.md` | `e12_fetch.py`, `e12_qc.py`, `e12_score.py` | `e12_analyze.py`, `e12_pool.py` | `e12_rows_*`, `e12_summary*.json` |
| E6s (post hoc) | `PREREG_E6s.md` | `e12_scoreK.py` | `e12K_analyze.py` | `e12K_*`, `e12K_summary.json` |
| E6c (confirmatory) | `PREREG_E6c.md`, `PREREG_E6c.freeze.json`, `frozen_E6c/` | `e13_detect.py`, `e15_*.py`, `e17_gnss_events.py`, `e17_labels.py`, `e17_select.py`, `e17_scoreK.py` | `e17K_analyze.py`, `e17K_table.py` (as frozen); `e27_circ_analyze.py`, `e27_circ_table.py` (corrected) | `e17*_*.json`, `e17K_summary.json`, `e27_circ_*.json` |
| Review analyses (exploratory) | addenda 1–2 of `PREREG_E6c.md` | `e17_scoreK2.py`, `e17_scoreK3.py`, `e19_*`, `e21_conf.py`, `e22_od.py`, `e24_decomp.py` (`run19.sh`–`run21.sh`) | `e17K2_analyze.py`, `e20_p1.py`, `e20_circ.py`, `e23_analyze.py`, `e25_k3_analyze.py` | `e17K2_*`, `e17K3_*`, `e19_*`, `e20_*`, `e21_*`, `e23_*`, `e24_*`, `e25_summary.json` |

Shared modules: `forces3b.py` (J2–J4, Sun/Moon, SRP, drag), `numprop.py` / `numprop_gpu.py` (CPU/GPU propagators),
`orbit.py`, `e6_nonlinear.py`, `e9_numeric.py`, `e10_numeric.py`, `e11_numeric.py` (free-energy estimators: Gaussian
mixtures, maneuver hypotheses, iterated posterior linearization), `e14_recal.py` (recalibration).
`gpu/` holds result summaries of GPU runs of E4-E6 (the E6 ones predate the angle fix and are kept for the record); `data/` the maneuver labels.
Other scripts (`e5_*`, `e7*`, `e8_*`, `hybrid.py`, `diag*.py`, `probe*.sh`, `*_plots.py`) are development studies that
preceded the reported ones; they are kept for completeness and are not used for any reported number.
Comments inside the scripts are partly in Portuguese.

## E6c pre-registration and freeze

`PREREG_E6c.md` is the plan as it stands: the text frozen on 2026-11-08 04:02 UTC followed, unchanged, by dated
amendments 2–3 (implementation only, before any result was inspected), the result record, and two exploratory
addenda written after the confirmatory result was known. `frozen_E6c/PREREG_E6c.md` is the frozen text itself (its SHA-256 matches the
freeze record), and it is byte-identical to the first 117 lines of `PREREG_E6c.md`. The freeze record was written
locally (hash and timestamp); it was not deposited with a third party at the time.

`verify_freeze.py` checks every hash in `PREREG_E6c.freeze.json`. All event lists, the selection, the force models,
the propagator, the event builder and the frozen estimator versions (`frozen_E6c/e10_numeric.py`,
`frozen_E6c/e11_numeric.py`) reproduce their hashes (15 of 17). Two code files changed after freezing and their frozen
copies were not retained: `numprop_gpu.py` (Amendment 2: Sun/Moon series evaluated on a 60-s host grid; GPU vs CPU
propagation differs by 3e-5 m after 2 days) and `e17_scoreK.py` (Amendment 3: deterministic resumption from partial
files). The regularization rule of Amendment 3 (regularize a batch only when one of its matrices is singular) was
implemented in `e11_numeric.py` (functions `_reg`, `_inv`, `_solve`); the `e11_numeric.py` shipped here also contains
the later, switchable exploratory options (`EXTRA`, `EXTRA2`, `SIGNED`, `COV_SCALE`, `tracklet2`), so the exact
estimator file used in the confirmatory runs is `frozen_E6c/e11_numeric.py` plus Amendment 3, not a stored file.

What the freeze record does not cover: the analysis scripts (`e17K_analyze.py` and the inference functions in
`e12K_analyze.py`) were not hashed, so their being unchanged since freezing cannot be checked from the record. As a
check of the whole chain, the exploratory rescoring with the code of that time (before the angle fix) reproduced all 252
confirmatory per-event values exactly (`e17K2_analyze.py`, maximum relative difference 0.0); the later rescoring with
the angle fix (`e17K3_*`) reproduces the full-state values exactly, and `e17K_analyze.py` reproduces
the primary result from the shipped per-event files.

Two notes on the plan text: the sign-flip test is exact (all sign vectors) for 12 satellites (E6s) and uses
2x10^5 random sign vectors for 35 satellites (E6c); and addendum 1 of `PREREG_E6c.md` says that the two-tracklet case
was "not reported", whereas it was later analysed and reported in the Supplementary Material (Section S3).

## Data not included (public, downloaded by the scripts)

DORIS/IDS precise orbits (ftp://doris.ign.fr), GFZ final multi-GNSS orbits (doi:10.5880/GFZ.1.1.2020.002),
GFZ MGEX orbits (ftp://ftp.gfz-potsdam.de/pub/GNSS/products/mgex), Swarm, COSMIC-2 and Sentinel-1 orbits, GPS NANU,
Galileo NAGU and QZSS OHI notices. The maneuver labels in `data/maneuver_labels_v0.3.json` are derived from
maneuver-detect v0.3 (CC-BY-4.0).

## License

Code: MIT (`LICENSE`). Plans, result files and figures: CC-BY-4.0 (`LICENSE-DATA`).

## Citation

Zenodo deposit: https://doi.org/10.5281/zenodo.24509633 (version 1.0.0). See `CITATION.cff`; please cite the
article and this deposit.
