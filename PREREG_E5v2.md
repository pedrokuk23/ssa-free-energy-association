# E5 v2 — pre-specified analysis (written 2026-11-05, before any v2 run)

Confirmatory run: `e10_numeric.py`, seed 3 (never used before), K0=41, K1=41, J=12,
N=3000 objects for each sigma_vI in {0.5, 0.05}. Pilot/seed-1/seed-2 results (e9) are exploratory only.

## Primary endpoints (8 tests, Holm-corrected at family-wise alpha = 0.05, one-sided "Prop better")
Prop = free energy with learned prior ("learn").
For each sigma_vI in {0.5, 0.05}:
1. Acceptance of maneuvered objects with |dv| >= 5 cm/s at 1% field false acceptance: Prop - lin_learn
   (same prior, linearized propagation — isolates propagation).
2. Same acceptance: Prop - CTRLu (uncertainty-aware control distance).
3. Neighbor AUC with dynamically transported neighbors (class "D"): Prop - lin_learn.
4. Same AUC: Prop - CTRLu.

Inference: two-stage cluster bootstrap (resample satellites, then objects within satellite), B = 2000,
one-sided p = fraction of bootstrap differences <= 0 (reported as bootstrap p). Holm over the 8 tests.

## Secondary (reported, not used for claims of significance)
GM, UKF (moment-matched), CTRL (L2 control distance), iso_lin, flat/Maxwell priors; old neighbors ("N");
acceptance for all maneuvers and vs |dv| bins; within-object neighbor AUC; detection AUC; log-loss of
P(maneuver) vs base rate; characterization (type accuracy, |dv| error factor) with trivial baselines;
leave-one-family-out (Jason-1/2/3+Sentinel-6A, Sentinel-3A/B, SPOT-2/4/5, other) sensitivity.

## Robustness runs (secondary)
- MISMATCH=1: true density AR(1) every 3 h (sigma 0.4, rho 0.7) and initial error 1.5x the catalog covariance,
  N=1000, sigma_vI=0.5.
- K1 convergence: K1 in {11, 21, 41, 81}, N=200, sigma_vI=0.5, same seed across K1 (paired).
