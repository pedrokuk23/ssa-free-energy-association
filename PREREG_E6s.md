# E6s — declared post hoc sensitivity analysis of E6a/E6b (written 2026-11-07, before running; code, K and seed frozen)

Status: POST HOC. Written after the E6a and E6b results (and the pooled diagnostic) were seen. It is reported alongside,
never instead of, the pre-registered E6a/E6b analyses, whatever its outcome.

Motivation: in E6a/E6b each real event had one draw of the synthetic nuisance components (catalog error, catalogued
neighbor, UCT noise/station, field object); this Monte Carlo noise inflates the variance of the joint-association AUC.

Events: all maneuver events with |dv| >= 5 cm/s of E6a (44) and E6b (146): 190 events, 12 satellites.
Draws: K = 20 per event (`KDRAW=20 MEAS={full,tracklet} python e12_scoreK.py 0.05 13`, chunked), estimator as in E5 v3/E6.

Primary endpoint (per event): correct joint assignment rate = fraction of the K draws in which the UCT of the maneuvered
object has a lower free energy (score) on its own track than on the catalogued neighbor's track; paired difference
Prop - CTRLu per event. Inference at satellite level: per-satellite mean differences, DerSimonian-Laird random effects with
Hartung-Knapp-Sidik-Jonkman correction (t, k-1 df), one-sided; exact satellite-level sign-flip test as a check.
Two tests (full, tracklet), Holm.

Secondary: the same against GM, UKF, Lin, CTRL; acceptance rate per event at 1% field false acceptance (threshold from
all field draws); correct assignment of the neighbor's UCT; results by study (E6a vs E6b).
Exploratory (data-suggested, not confirmatory): events with |dv| >= 30 cm/s.

Also reported: per-satellite forest data, HKSJ estimate, prediction interval and sign-flip test for the six original
pre-registered E6a/E6b endpoints (`e12K_analyze.py`).
