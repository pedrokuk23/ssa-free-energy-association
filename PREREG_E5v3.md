# E5 v3 — pre-specified analysis (written 2026-11-06, before any v3 run)

Code: `e11_numeric.py` (same dynamics, priors and maneuver hypotheses as v2; measurement evidence per mixture
component computed with iterated posterior linearization (IPLF, 2 iterations) — validated against importance
sampling (ESS 100–300) to within 0.1 nat for tracklets, where single-pass EKF/UT projections were off by 3–6 nats).
Seed 5 (never used before). K0 = K1 = 41, J = 12. N = 1500 catalog objects per condition.

Conditions (main): MEAS in {tracklet, full} x sigma_vI in {0.5, 0.05}.
- tracklet: optical attributable (alpha, delta, alpha-dot, delta-dot) from a ground station 0–600 km from the
  sub-satellite point (range ~550–820 km); 0.7" angles, 0.04"/s rates (8 obs at 2" over 60 s).
- Each UCT (true object, maneuvered object, transported neighbor, field object) gets its own station.
- Neighbor: catalogued object, initial state 1.5–3 sigma from the target (catalog covariance), propagated by the true
  dynamics; its own catalog track has mean error ~ N(0, P0) and covariance P0.

Joint association score (new): for each UCT, free energies against the target track (s_t) and the neighbor's own
track (s_n); J = softplus(s_t - s_n) = -log P(target | z, two tracks) with equal track priors
(for CTRL/CTRLu, which are quadratic forms, J = softplus((s_t - s_n)/2)).

## Primary endpoints (12 tests, Holm, family-wise alpha 0.05, one-sided "Prop better")
Prop = free energy, propagated hypotheses, learned prior.
1–4. Tracklet, both sigma: acceptance (|dv| >= 5 cm/s, 1% field false acceptance, single-track):
     Prop - Lin (same prior, linearized hypotheses) and Prop - CTRLu.
5–8. Both MEAS x both sigma: joint-association AUC (maneuvered |dv| >= 5 cm/s vs neighbor), Prop - CTRLu.
9–12. Both MEAS x both sigma: joint-association AUC, Prop - GM.
Inference: two-stage cluster bootstrap (satellite, then objects), B = 2000; one-sided bootstrap p; Holm over 12.

## Secondary (no significance claims)
All other metrics as in v2 (single-track neighbor AUC, cross-track acceptance, detection AUC, log-loss, acceptance
by |dv| bin), joint AUC for all methods, magnitude error factor with the continuous in-bin refinement (dv_ref) versus
bin-center estimate (dv_hat) and true-bin snapping.

## Robustness (secondary)
Drag mismatch (MISMATCH=1, full state, sigma_vI = 0.5, N = 600, seed 6): estimator density uncertainty
SIG_BETA_EST = 0.2 versus 0.4 on the same objects (paired); endpoints: log-loss of P(maneuver) and acceptance.
