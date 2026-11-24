# E6 (real orbits and real maneuvers) — pre-specified analysis (written 2026-11-06, before any download or run)

Truth: DORIS precise orbits (IGN data center; SSALTO, else GSFC, else GRGS) of the 13 LEO satellites in the
maneuver-detect labels; ITRF -> pseudo-inertial by GMST rotation. Events (`e12_fetch.py 40 25 7`): up to 40
labelled maneuvers per satellite (in-track/cross-track/radial, |dv| > 0.1 mm/s) with t0 in [t_m - 22 h, t_m - 2 h],
T = t0 + 1 day, no other labelled maneuver in [t0 - 1 h, T + 1 h]; up to 25 control windows per satellite with
no labelled maneuver in [t0 - 6 h, T + 6 h]. Estimator unchanged from E5 v3 (K0 = K1 = 41, J = 12, IPLF,
J2–J4 + drag model, leave-one-satellite-out priors). Synthetic parts: catalog error ~ N(0, P0) around the real
state, UCT noise, catalogued neighbor (1.5–3 sigma, propagated by the model), field objects.
Conditions: sigma_vI = 0.05 m/s with MEAS in {full, tracklet} (seed 7).

## Primary endpoints (6 tests, Holm, one-sided "Prop better")
For each MEAS:
1. Acceptance of real maneuvered objects with |dv| >= 5 cm/s at 1% field false acceptance: Prop - CTRLu.
2. Detection AUC, real maneuvers (|dv| >= 5 cm/s) vs real control windows (logit P(mnv) for Prop; score for
   CTRLu): Prop - CTRLu.
3. Joint-association AUC (maneuvered objects vs catalogued neighbors): Prop - CTRLu.
Two-stage cluster bootstrap (satellite, then events), B = 2000.

## Secondary
Acceptance of real control objects (false rejection caused by model mismatch); the same comparisons against GM
and Lin; type accuracy and magnitude error factor (refined) against the operator labels; one-day model
prediction error of the J2–J4 + drag model against the precise orbits (controls), by satellite.
