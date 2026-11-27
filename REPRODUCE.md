# Where each table and figure comes from

Run every command from the package root. Analysis scripts read the shipped result files and rewrite the summary
named in the last column; the numbers in the paper are read from that summary. Every number in the tables listed
below was checked against these files.

## Main text

| Item | Content | Script | Summary file |
|---|---|---|---|
| Table 1 | Correspondence of metrics (analytical) | — | — |
| Fig. 1 | E4, predicted density against single-Gaussian ellipses | `figures/make_figures.py` (`fig_nonlinear`) | recomputed from the model (fixed seed) |
| Table E4 (supplement) | E4, AUC and evidence error (supplement) | `e6n_numeric.py`, `e4_is.py`, `e4_is_005.py` | `e6n_summary.json`, `e4_is_summary.json`, `e4_is_summary_005.json` |
| Table E5 | E5 Study A, main metrics | `e10_analyze.py` | `e10_summary.json` |
| Fig. 2 | E5 Study A: maneuver magnitudes, paired differences, calibration by prior, largest maneuvers | `figures/make_figures.py` (`fig_e10`) | `e10_summary.json`, `data/maneuver_labels_v0.3.json` |
| Table E5b | E5 Study B | `e11_analyze.py` | `e11_summary.json` |
| Table paired2 (supplement) | E5 Study B, pre-specified tests | `e11_analyze.py` | `e11_summary.json` |
| Table chrono | Plan chronology | — | `PREREG_*.md`, `PREREG_E6c.freeze.json` |
| Table E6c | E6c rates by method, UCT type and regime (corrected tracklet scores) | `e27_circ_analyze.py`, `e27_circ_table.py`; as frozen: `e17K_analyze.py` | `e27_circ_summary.json`, `e27_circ_table.json`; `e17K_summary.json` |
| Fig. 3 | E6c forest plot (corrected; open diamond: as frozen) | `figures/forest_se.py`, `figures/make_figures.py` (`fig_forest_e6c`) | `e27_circ_summary.json`, `e17K_summary.json`, `e17sel_*.json`, `figures/forest_se.json` |

## Supplement

| Item | Content | Script | Summary file |
|---|---|---|---|
| S1 | E1–E3, Clohessy–Wiltshire checks | `montecarlo.py`, `e3_v2.py` | `mc_summary.json`, `e3v2_summary.json` |
| Table paired | E5 Study A, pre-specified tests | `e10_analyze.py` | `e10_summary.json` |
| Table prior | E5 prior ablation and recalibration | `e10_analyze.py`, `e14_recal.py` | `e10_summary.json`, `e14_recal.json` |
| Table s3imm, S3 text | Study B with CTRLp/IMM; IPLF-5; OD catalog | `e23_analyze.py`, `e19_pd_analyze.py` | `e23_pdx.json`, `e23_pdit5.json`, `e23_od2.json`, `e19_pd_summary.json` |
| Tables e6, paired3 | E6a/E6b pooled and primary tests | `e12_analyze.py`, `e12_pool.py` | `e12_summary_seed7.json`, `e12_summary_seed11.json`, `e12_summary_seedpool.json` |
| Fig. S1 | E6s forest plot | `figures/forest_se.py`, `figures/make_figures.py` (`fig_forest_e6s`) | `e12K_summary.json`, `figures/forest_se.json` |
| Table s5meta, S5 text | E6c sensitivity, offset, Δv, real confusers, all eligible events (corrected); full-state calibration and near confusers | `e20_circ.py`, `e25_k3_analyze.py`; `e20_p1.py`, `e17K2_analyze.py` (full state) | `e20_circ.json`, `e25_summary.json`, `e21_conf_*.json`; `e20_p1.json`, `e17K2_summary.json` |
| S6 | Evidence decomposition and independent reference | `e24_decomp.py` | `e24_full_s0.5.json`, `e24_tracklet_s0.5.json` |

## Re-running the scoring

The `run*.sh` files are the exact command lines used on the original machine (Ubuntu, Python 3.13, NumPy, SciPy,
Numba CUDA, one RTX 5090, 8 CPU cores); paths assume the package root as working directory and a virtual
environment in `~/venv`. Approximate GPU times: E5 Study A or B about one day each, E6c scoring about one day,
E6c re-scoring (`run19.sh`, `run20.sh`) about 1–1.5 days. All random draws are seeded by study, event and draw.

## Commands for the summaries that need arguments

    python e23_analyze.py 'e11_rows_pdx_*_seed31_*_x.json' e23_pdx.json
    python e23_analyze.py 'e11_rows_pdit5_*_seed31_*_x.json' e23_pdit5.json
    python e23_analyze.py 'e11_rows_od2_*_seed41_*.json' e23_od2.json
    SEED=7 python e12_analyze.py      # writes e12_summary_seed7.json (E6a)
    SEED=11 python e12_analyze.py     # E6b
    python e12_pool.py; SEED=pool python e12_analyze.py

`e20_p1.py` and `e25_k3_analyze.py` also write the plain-text logs `c121.txt` and `c125.txt`. The file
`e17K_e17_events_gnss_full_s0.05_seed13_c0of31.json` is the three-event software test run before the E6c freeze
(Amendment 1); it is not used by any analysis.

Audit additions (26 November 2026):

| Item | Content | Script | Output |
|---|---|---|---|
| E6c text, dominant term | posterior weight of the largest hypothesis term (median 22%), CTRLp-Lin decision agreement (96%/97%) | `e28_dominant.py` | `e28_dominant.txt` |
| Table s5meta, E6c text | primary with pi1 from navigation satellites only (+0.037) | `e29_p1nav.py` | `e29_p1nav.txt` |
| Supplement S4 (vii)-(ix) | post hoc diagnostics of E6a/E6b (model error, control objects, ceiling, dv and type) | `e12_diag.py` | `gpu/e12_diag.json` |
| Supplement S6, frame rotations | log-evidence under 30 rotations about the polar axis (3 objects) | `e30_frame_rotation.py` | `e30_frame_rotation.txt` |
| Supplement S6, IPL normalization | scalar benchmark of the evidence formula (y = x^2 + noise) | `e31_ipl_scalar.py` | `e31_ipl_scalar.txt` |
| Supplement S6, covariance check | positive definiteness and conditioning of all matrices passed to loggauss (48 full scorings, one run; header records versions and hashes) | `e32_covariance_check.py` | `e32_covariance_check.txt` |
| Supplement S6, E6c covariance check | E6c production settings on real LEO, MEO and QZSS events: target and neighbour scores of both routines vs archive, decisions, asymmetry, perturbation sensitivity | `e33_e6c_covariance_check.py` (`run33.sh`) | `e33_full.txt`, `e33_tracklet.txt` (revision 3), `e33_r4_*.txt` (revision 4) |
| Main text E6c, Supplement S6 | bounds on the primary from rounding sensitivity of tracklet scores | `e34_rounding_bound.py` | `e34_rounding_bound.txt`, `e34_rounding_bound.json` |
| Supplement S6, loggauss tests | per-matrix validation, repair, rejection, batch invariance | `test_gauss_logpdf.py` | stdout |
