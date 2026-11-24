# Free-energy association of maneuvering space objects — code, plans and results

Companion package of the manuscript *"Free-energy association of maneuvering space objects: Relating statistical
evidence and control distance"* (P. Kukulka de Albuquerque, J. Cavalcanti, L. Galembeck), submitted to *Acta Astronautica*.

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
on the orientation of the inertial frame. The effect is large for the single moment-matched Gaussian (UKF, IMM) and
negligible for the mixture-based scores in LEO (Prop, Lin, CTRLu: at most 0.1 nat). After the two-day gaps of medium Earth
and geosynchronous orbits, however, the maneuver-hypothesis blocks of Prop, Lin and CTRLp span wide arcs of right
ascension, and their tracklet evidence in the original E6c runs is degenerate (median F of order -1e10 nats for MEO
tracklets); the E6c tracklet results for MEO and QZSS therefore come from the rerun.
`e11_numeric.py` now averages angles on the circle and uses a symmetric square root (switches `CIRC_MEAN=1`,
`EQ_SQRT=1`, the defaults); `CIRC_MEAN=0 EQ_SQRT=0` reproduces the original runs. `run22_circ.sh` repeats every
tracklet scoring with the fix; full-state results are unaffected because `ev()` does not use either step for them.

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
| E6c (confirmatory) | `PREREG_E6c.md`, `PREREG_E6c.freeze.json`, `frozen_E6c/` | `e13_detect.py`, `e15_*.py`, `e17_gnss_events.py`, `e17_labels.py`, `e17_select.py`, `e17_scoreK.py` | `e17K_analyze.py`, `e17K_table.py` | `e17*_*.json`, `e17K_summary.json` |
| Review analyses (exploratory) | addenda 1–2 of `PREREG_E6c.md` | `e17_scoreK2.py`, `e17_scoreK3.py`, `e19_*`, `e21_conf.py`, `e22_od.py`, `e24_decomp.py` (`run19.sh`–`run21.sh`) | `e17K2_analyze.py`, `e20_p1.py`, `e23_analyze.py`, `e25_k3_analyze.py` | `e17K2_*`, `e17K3_*`, `e19_*`, `e20_*`, `e21_*`, `e23_*`, `e24_*`, `e25_summary.json` |

Shared modules: `forces3b.py` (J2–J4, Sun/Moon, SRP, drag), `numprop.py` / `numprop_gpu.py` (CPU/GPU propagators),
`orbit.py`, `e6_nonlinear.py`, `e9_numeric.py`, `e10_numeric.py`, `e11_numeric.py` (free-energy estimators: Gaussian
mixtures, maneuver hypotheses, iterated posterior linearization), `e14_recal.py` (recalibration).
`gpu/` holds the result summaries of the GPU runs of E4/E5; `data/` the maneuver labels.
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
check of the whole chain, the exploratory rescoring with the current code reproduced all 252 confirmatory
per-event values exactly (`e17K2_analyze.py`, maximum relative difference 0.0), and `e17K_analyze.py` reproduces
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
