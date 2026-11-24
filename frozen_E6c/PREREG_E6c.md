# E6c — confirmatory study on fresh satellites (DRAFT v0, written 2026-11-07, before any E6c data download)

Status: to be frozen (hash + timestamp, deposited on OSF/Zenodo) before the event download starts. Any change after
freezing is listed as a dated amendment.

## Question
Does joint scoring with the free energy (Prop) assign the UCT of a maneuvered object to its own catalog track,
rather than to a catalogued neighbor's, more often than the uncertainty-aware control distance (CTRLu), on satellites
never used in E5/E6a/E6b/E6s?

## Satellites (fresh only; data access verified 2026-11-07 from the server, no restricted accounts)
Tier 1 (operator maneuver files with dv, IDS ftp.ids-doris.org): SWOT (27 maneuvers >= 5 cm/s), Sentinel-3C (7),
Sentinel-6B (6); HY-2D (2, below the inclusion bar). Orbits: IDS DORIS SP3 (first covering center of SSALTO, GSFC,
GRGS, LCA, IGN).
Tier 2 (operator maneuver epochs without dv): TOPEX/Poseidon (IDS file: epochs only; dv from detector D).
Tier 3 (precise orbits, maneuvers from detector D): TerraSAR-X and TanDEM-X (GFZ ISDC, tsxtdx/ORBIT L13/L20),
CHAMP (ISDC champ/ORBIT L06), GRACE-FO C/D (ISDC Level-1B GNV1B), GRACE A/B (ISDC), Sentinel-1A/1B/1C (ESA STEP
POEORB), Swarm A/B/C (ESA swarm-diss HTTPS, SW_OPER_SP3xCOM_2_, public, verified 2026-11-07), COSMIC-2 FM1-FM6
(UCAR CDAAC leoOrb, public, verified 2026-11-07). Not used: Sentinel-2 (CDSE AUX_POEORB not yet available), Sentinel-3/6
CDSE POD (same satellites already covered by IDS DORIS orbits), ICESat-2 (no public precise-orbit product found). Files
that the author downloads manually go to data/manual/<mission>/ and are read only from there.
Inclusion: a satellite enters if it has >= 3 isolated events with |dv| >= 5 cm/s. Target: >= 12 fresh satellites.

## Maneuver detector D (for unlabelled satellites; validated before use)
Fit the estimator dynamics to the precise orbit over sliding 6-h arcs; flag epochs where the fit residual and the
jump in fitted mean semi-major axis / inclination / RAAN exceed thresholds; estimate dv and type from the jump.
Validation on the 13 DORIS satellites with IDS labels (|dv| >= 5 cm/s): precision and recall >= 0.9, median
magnitude error factor <= 1.3; otherwise tier-3 satellites are not used. Thresholds fixed on the validation set.
Validation record (2026-11-07, before any association scoring of E6c data):
- v1 (step only): magnitude factor 1.48 on CryoSat-2 (in-track burns read as cross-track: the change of a changes
  the RAAN drift rate) -> v2 adds a rate-change term to the inclination/RAAN fits and keeps only components with
  |z| > 5. The noise is almost all in the cross-track statistic (control median z_ct 6-13 vs z_it 0.3), so the
  decision statistic is normalized by the satellite's own median: zn = max(z_it/med_it, z_ct/med_ct), threshold
  zn = 4 (smallest value with no false alarm). Window mode, 6 DORIS satellites, 74 isolated labelled burns, 90
  controls: recall 0.986, precision 1.00, magnitude factor 1.02 (p75 1.07). Medians on new satellites are taken
  over all scan windows of the year.
- Scan mode (1-day steps over a year) must ALSO be validated, because 15 controls per satellite cannot bound a
  per-window false-alarm rate of ~1% and a year has ~360 windows: one-year scans of 5 labelled DORIS satellites
  (2022) are compared with all IDS labels (+-6 h). Criterion: precision >= 0.9 for isolated events >= 5 cm/s.
  If it fails, the threshold or statistic may be revised on these DORIS scans only, and tier-3 satellites are used
  only if the revised detector meets the criterion on them.
  Result (2022 scans of S3A, S3B, CryoSat-2, Jason-3, SARAL): FAILED for the cross-track part. Cross-track
  magnitudes are dominated by noise (control median z_ct 9-16, so the |z| > 5 gate always passes): 27 false
  "cross-track" events of ~0.13-0.25 m/s on S3A/S3B/CS2/SARAL; on Jason-3, 9-10 m/s jumps match an 18-s GPS/UTC
  time-scale mismatch between analysis centres (v sin i * omega * 18 s = 8.6 m/s). In-track part: no false event
  >= 5 cm/s on any of the five satellites; labelled burns >= 5 cm/s recovered 15/17; false in-track events are all
  < 2.5 cm/s except Jason-3 time-scale jumps. Decision: detector D is used in-track only (dv = dv_it, events with
  dv_it >= 5 cm/s); cross-track detections only flag windows that are then excluded from the control set; one
  analysis centre per window and the SP3 time system read from the header (fixes the Jason-3 jumps).
- Low altitude: the validation satellites fly at 700-1340 km. Satellites below 600 km (Swarm, COSMIC-2, GRACE-FO,
  TSX/TDX, CHAMP) enter only if the detector also meets the criteria against external labels at low altitude
  (GRACE-FO orbit-thruster firings, Level-1B THR1B); otherwise they are excluded.

## Scenario changes relative to E6
1. Catalog state by orbit determination: least-squares fit of the estimator model (with a drag scale factor) to
   the precise positions over [t0 - 24 h, t0] with 30 m noise; catalog covariance = fit covariance scaled to the
   empirical one-day prediction error of control windows of OTHER satellites (leave-one-satellite-out).
2. Neighbors and field objects carry real model error: their truth is the estimator-model propagation of their
   initial state plus the empirical model-error vector (RIC) of a control window of the same satellite nearest in time.
3. Field false-acceptance threshold from these field objects; control acceptance reported on real control windows.
4. Maneuver probabilities of Prop are recalibrated with the two-parameter logistic map fitted on all E6a/E6b events
   (frozen before E6c data are scored); raw and recalibrated log-loss are both reported.
5. Baselines added: CTRLp (control distance with the learned maneuver prior, MAP form) and IMM (two-mode
   interacting-multiple-model UKF: natural / maneuver process noise, mode probabilities from the learned rate).

## Primary endpoint (single test)
Per event: correct joint assignment rate over K = 20 draws (catalog error, neighbor, UCT noise/station), averaged
over the full-state and tracklet UCTs; paired difference Prop - CTRLu; events with |dv| >= 5 cm/s.
Inference: satellite-level HKSJ random effects (pooled within-satellite variance), one-sided alpha = 0.05;
exact satellite-level sign-flip test reported alongside. Success: lower 95% one-sided bound > 0.

## Secondary
Same endpoint per UCT type; vs GM, UKF, Lin, CTRLp, IMM; |dv| >= 30 cm/s (justification: Delta v above which the
one-day displacement exceeds one catalog sigma along the dominant direction — computed per satellite before
unblinding); acceptance at 1% field false acceptance; detection AUC vs real control windows; type and magnitude
accuracy; one-day prediction error of the fitted catalog model; tier-1 only (operator labels) subset.

## Power
From E6s (12 satellites): SE of the HKSJ estimate ~0.010 (full) and ~0.012 (tracklet), tau ~0.013-0.023. With
>= 12 fresh satellites and a true effect of +0.04 (the E6s full-state estimate), the one-sided test at alpha = 0.05
has power > 0.95; for +0.02 (half of it), about 0.6.

## Amendment 1 (2026-11-08, before any E6c event is scored except a 3-event software test with K=2)
Reason: the DORIS pool is exhausted (E6b already used every usable labelled burn >= 5 cm/s) and fresh LEO satellites
with operator labels are only SWOT, Sentinel-3C and Sentinel-6B; detector-labelled LEO satellites are not used
(the detector fails in scan mode for cross-track and finds few isolated in-track burns >= 5 cm/s). Power therefore
comes from navigation satellites with operator maneuver notices and public precise orbits.
Satellites and events:
- GNSS/MEO: GPS (NANU) and Galileo (NAGU) maneuvers since 2020-09 with GFZ final multi-GNSS orbits (15 min); the
  analysis centre removes the satellite on the maneuver day, so each event uses the state at t0 (last epoch before
  the notice window minus U{0..6 h}) and the observation at T = t0 + 2 days. dv and type are estimated from the
  mean elements of the truth over [T, T+P] minus those of the model propagation from x0 (controls: apparent dv
  <= 0.016 m/s, median model position error at 2 days 1.1 km). Inclusion dv >= 5 cm/s: 93 events, 28 satellites.
- QZSS (OHI notices with operator dv; GFZ MGEX orbits, 5 min, 2018-2026): same construction; PRN identified as the
  J-PRN absent on label days but present 3 days before and after.
- Fresh LEO: SWOT, Sentinel-3C, Sentinel-6B (IDS operator files), constructed as in E6 (1-day gap).
- At most 8 maneuver events per satellite (random subset, seed 17) so that no satellite dominates the run time.
- Inclusion (replaces the >= 3 rule of draft v0, which was written for detector-labelled satellites): every
  satellite with at least one event >= 5 cm/s enters; the satellite-level estimator weights satellites through
  the pooled within-satellite variance s2/n_i, so satellites with few events carry little weight.
Estimator for MEO/GEO/IGSO: same estimator with J2-J4 + Sun + Moon (low-precision analytic ephemerides,
IAU-1976 precession to mean-of-date) + cannonball SRP (Cr A/m = 0.02 GPS, 0.015 Galileo, 0.012 QZSS; uncertain
log-scale factor sigma 0.2 in place of the drag factor), no drag; 2-day horizon; J = 12 maneuver epochs; the maneuver
prior is learned leave-one-satellite-out from the other navigation satellites (dv from the events when the notice
gives none). LEO events use the E6 estimator unchanged (prior from the 13 DORIS satellites).
Synthetic components per draw exactly as in E6s (catalog error, catalogued neighbour 1.5-3 sigma with own track,
UCT noise/station, field object), K = 20 draws, full-state and tracklet UCTs, s_vI = 0.05 m/s.
Primary endpoint (unchanged): per event, correct joint assignment rate F_t(z) < F_n(z) over the K draws, averaged
over full-state and tracklet UCTs; paired difference Prop ('learn') - CTRLu; satellite-level HKSJ random effects
(pooled within-satellite variance), one-sided alpha = 0.05, success if the lower 95 % one-sided bound > 0; exact
satellite-level sign-flip test reported alongside. All included satellites (GNSS, QZSS, fresh LEO) pooled.
Secondary: same endpoint by regime (MEO, GEO/IGSO, LEO) and by UCT type; vs GM, UKF, Lin, CTRL; dv >= 30 cm/s;
acceptance at 1 % field false acceptance; raw and recalibrated maneuver-probability log-loss (recalibration map
fitted on E6a/E6b, frozen).
Power: 31-36 satellites; scaling the E6s standard error (0.010 with 12 satellites) by sqrt(12/33) gives ~0.006, i.e. power > 0.99 for +0.04 and about
0.95 for +0.02 (one-sided 0.05; the GNSS between-satellite heterogeneity is unknown and may lower this).
Not done (reported as limitations): catalog by orbit determination, CTRLp and IMM baselines.
