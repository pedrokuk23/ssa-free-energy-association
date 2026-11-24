# E6b (replication on real orbits, all large maneuvers) — pre-specified analysis (written 2026-11-06, before any download or run for E6b)

Motivation: E6 (seed 7) had only 44 maneuvers with |dv| >= 5 cm/s, too few for the acceptance and detection endpoints.
E6b is an independent replication with the same estimators, scenario and endpoints.

Events (`MINDV=0.05 EXCL=e12_events_seed7.json python e12_fetch.py 1000 40 11 <sat>`, one process per satellite):
every labelled maneuver of the 13 satellites with |dv| >= 5 cm/s (in-track, cross-track, radial) that was NOT a maneuver
event of E6, with t0 in [t_m - 22 h, t_m - 2 h], T = t0 + 1 day, no other labelled maneuver in [t0 - 1 h, T + 1 h] and
precise orbits available at t0 and T; 40 new control windows per satellite (no labelled maneuver in [t0 - 6 h, T + 6 h]).
SPOT-2 is retried; satellites whose orbit files cannot be retrieved are excluded and reported.
Scoring: `EVENTS=e12_events_b.json MEAS={full,tracklet} python e12_score.py 0.05 11` (estimator, priors, synthetic
catalog error, neighbor and field objects as in E6). Analysis: `SEED=11 python e12_analyze.py`.

## Primary endpoints (6 tests, Holm, one-sided "Prop better") — identical to E6
For each MEAS: (1) acceptance at 1% field false acceptance, |dv| >= 5 cm/s, Prop - CTRLu; (2) detection AUC, real
maneuvers (|dv| >= 5 cm/s) vs real control windows, Prop - CTRLu; (3) joint-association AUC vs catalogued neighbor,
Prop - CTRLu. Two-stage cluster bootstrap (satellite, then events), B = 2000.

## Secondary
As in E6 (vs GM, Lin, UKF; control acceptance; type and magnitude; model error). The pooled E6 + E6b sample is
reported as secondary only.

## Amendment 1 (2026-11-06, after the event download started and before any event was scored)
The first download used one analysis center per satellite (SSALTO first) and returned only 0-8 large maneuvers per
satellite, because the SSALTO series does not cover most epochs of the large maneuvers (e.g. SPOT-4 burns of 1998-2009)
and has gaps around them. The event definition is unchanged, but (i) the orbit at each epoch is taken from the first of
SSALTO, GSFC, GRGS, LCA, IGN whose file covers it; (ii) when an SP3 file has no velocities they are obtained by 9-point
Lagrange differentiation of the positions; (iii) up to 12 catalog epochs t0 are drawn per maneuver (uniform in
[t_m - 22 h, t_m - 2 h]) and the first isolated one with available orbits is kept. The download was stopped and rerun;
no score of any E6b event had been computed. All primary endpoints and the analysis are unchanged.

## Amendment 2 (2026-11-06, before any event was scored)
Scoring stopped at the first event with a singular covariance. A state check (`e12_qc.py`) showed velocities of
about 0.67 km/s in the GRGS/LCA series of SPOT-2/4/5, i.e. velocity records not in the standard SP3 unit. Velocities
are now always recomputed from the positions by 9-point Lagrange differentiation and replace the file velocities
when these are missing or differ by more than 1 m/s (median). The events of SPOT-2, SPOT-4 and SPOT-5 were
regenerated with the same seed. A quality filter, fixed before scoring, removes any event with non-finite states,
|r| outside 6600-8000 km, |v| outside 6.5-8.0 km/s, or an osculating semi-major axis change above 50 km over the day.
