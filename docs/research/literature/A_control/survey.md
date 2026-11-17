# Cluster A survey: how possession gain and loss are decided

Question: how do published systems and data providers decide possession gain and loss from tracking data, and which of their rules fix the failure modes in docs/possession-state-machine.html?

Method: WebSearch only (WebFetch was blocked), so every claim comes from abstracts and search snippets. Anything not in that text is marked "not found" or "unverified". Each source has its own note in this folder.

## Sources

| Note | Source | Type | Gain / loss rule | Ball needed? | Reported result | Failure mode it addresses |
|---|---|---|---|---|---|---|
| vidal-codina-2022.md | Vidal-Codina et al. 2022, Sports Eng. (arXiv 2202.00804) | rules, per player | Possession zone 0.5-1 m (depends on provider); duel zone at least the possession zone and overrides it; a gain needs a change in ball direction or speed; a loss needs the ball to leave the zone and move past a threshold | yes | event detection above 90% in most categories | the source of failure paths 1-2 (gain waits for a touch); radius depends on provider |
| link-hoernig-2017.md | Link & Hoernig 2017, PLOS ONE | rules + ML + Bayesian network | IBA starts when a player can act on the ball and ends after 1 s without interaction; control is classified from duration, ball speed and acceleration, and opponents within 0.5-5 m; in play, some team always has possession | yes | P/R 80-92% (2016 precursor); 2017 numbers not found | paths 3-4 (no empty labels); control vs touch |
| bassek-2025-idsse.md | Bassek et al. 2025, Sci. Data (IDSSE) | dataset | BallPossession 1/2 is "defined by the player in control"; algorithm not published; flag reportedly persists during dead balls (unverified) | - | - | understanding the IDSSE truth (286 vs 511 possessions) |
| pathcrf-kim-2026.md | Kim et al. 2026, KDD (PathCRF) | learned + masked CRF + Viterbi | possession = one player-to-player edge per step; transitions restricted by a mask; Viterbi decoding | **no** | team possession per frame 92.40% with no ball; event F1 75.7% | path 5 (ball gaps), late-dated turnovers in gaps |
| kim-2023-ball-trajectory-inference.md | Kim et al. 2023, KDD | Set Transformer + Bi-LSTM | predicts possession first, then the ball | no | numbers not found | path 5 |
| borghesi-2023-tcn.md | Borghesi, Costa, Morra, Lamberti 2023, ESWA | TCN, H/A/D per frame | two stages: dead or alive, then which team | yes | possession 86.2%, dead/alive 89.2% | the whole problem; realistic ceiling |
| lang-2022-ingame-status.md | Lang, Wild, Isenko, Link 2022, Sci. Rep. | tree ensembles | dead or alive from players only | no | 92-93% per frame | cluster C; shared player-only features |
| li-link-2026-phases.md | Li & Link 2026, arXiv | TGAT + rules + filter | "Episode" = stable possession of at least 3 s; post-processing at the segment level | yes | segment F1 0.61 -> 0.71 from post-processing | splits and short possessions (segment metric like ours) |
| wang-mills-2025-last-touch.md | Wang, Mills, Billingham et al. 2025, Sports Eng. | touch probability + attribution | finds the touch frame, then the player who made it | yes (instrumented ball) | 82.5% in duels, above 90% in aerial duels | evidence that a touch is unreliable as a hard gate |
| wei-2015-ball-ownership.md | Wei, Sha, Lucey et al. 2015 (+ a-HCRF 2014) | role-ordered features | ball owner from player trajectories | no | numbers not found | path 5; features that need no player IDs |
| databallpy-kloppy-tools.md | databallpy / kloppy / unravelsports | tools | Vidal-Codina port; team possession taken from events; inferring by distance fails on contested and aerial balls; TRACAB switches on an unsuccessful pass | - | - | switch timing depends on which truth is used |
| metrica-sample-data.md | Metrica sample data | dataset | no ball-owning-team column in tracking; the ball is often placed exactly on the carrier | - | - | zone tuned on Metrica is too tight for real tracking |
| event-provider-definitions.md | Opta/Stats Perform, StatsBomb, SPADL | event definitions | possession ends only when the opponent gains control; a stoppage alone does not end it | - | - | matches our truth; restarts |
| tracking-providers-not-found.md | SkillCorner, Second Spectrum, TRACAB, Hawk-Eye | providers | **not found** (TRACAB: has a flag, rule unknown) | - | - | - |
| community-hackathon-repos.md | 3 GitHub repos on the same challenge | unverified | touch evidence kept weak; switches shorter than 1 s filtered out | - | - | - |

## What the literature agrees on
1. **In play, a team always has the ball.** Link & Hoernig, Opta, StatsBomb and SPADL all hold this. A loose ball, a block or an aerial duel without control keeps the last team. MatchLens breaks this rule; our team rule mostly keeps it.
2. **A possession changes on control, not on a touch.** Control is judged from duration and context: time in the sphere of influence, ball speed and acceleration, and opponents nearby. A change of ball direction is evidence of a touch, and even with an instrumented ball a touch is only attributed correctly about 82% of the time in duels (Wang 2025).
3. **The zone radius depends on the provider:** 0.5 m vs 1 m in Vidal-Codina, and others use 1.5 m on IDSSE/TRACAB. Metrica puts the ball on the carrier, so a radius tuned on Metrica is too tight for real tracking.
4. **The best systems decode sequences, not frames.** The TCN, PathCRF (Viterbi with transition masks) and Li & Link (segment post-processing, +0.10 F1) all work this way.
5. **Possession can be read without the ball.** PathCRF gets 92.4% of frames right with no ball data; Lang 2022 gets 92% for dead vs alive with no ball data.

## Gaps
- No provider publishes its per-frame owning-team rule: DFL/Sportec, Metrica corrected tracking, SkillCorner, Second Spectrum, TRACAB and Hawk-Eye were all searched and none was found. The only way to learn the switch convention of each truth is from our own bench.
- Thresholds in Vidal-Codina (eps_theta, eps_v, displacement, duel radius) and in Link & Hoernig (T_p, the network's thresholds) were not found in search text. Our port's values could not be checked against the papers.
- No paper studies team possession with the ball missing in long blocks (5-30 s) while it is visible elsewhere. PathCRF and Kim 2023 assume no ball at all, Borghesi assumes a full ball track, and nothing combines the two.
- No paper reports how much switch-timing bias (labelling at confirmation vs at the first control frame) costs on an IoU-matched segment metric.
- How IDSSE BallPossession handles stoppages and very short flips is unknown; it is reported only by one informal source.

## Candidate fixes (testable hypotheses)

Current code checked: `src/possess/team_control.py` switches `cur` at the frame where `run` reaches `k`. The earlier k-1 challenge frames keep the old team, so the switch is never back-dated. A ball gap does not reset `run`.

Throughout, "spellF1" means the tune-split mean (base 0.3747, iteration 3). The predicted numbers below are rough priors, not measurements.

### H1 - Back-date confirmed gains to the first control frame (offline)
- Source: Vidal-Codina back-dates committed losses; event truths (Opta, Metrica events) date the switch at the controlling action; Link & Hoernig start an IBA when the player is first able to act.
- Hypothesis: every predicted switch is late by at least k-1 visible frames, and longer when ambiguous or hidden frames fall inside the run. That shifts both neighbouring possessions and costs IoU on each of them.
- Prediction: the mean absolute offset of switch boundaries vs truth drops by about the run length (at least 0.16 s at 25 Hz with k=5, more with gaps inside the run). Wrong-team frames next to turnovers shrink by 30-60%. spellF1 rises +0.003 to +0.01, mostly from matches that now clear the IoU threshold. Possession count unchanged.
- Smallest implementation: in `team_from_distances` remember `cand_start = i` when `run` becomes 1, and on commit set `out[cand_start:i] = ctrl`. About 4 lines. Measure the median truth-vs-predicted switch offset before and after, per truth (Metrica, IDSSE).

### H2 - Sequence decoding: 3-state HMM/HSMM (H, A, D) with a switch penalty and minimum duration
- Source: PathCRF (masked CRF + Viterbi), Borghesi TCN, Li & Link segment post-processing (+0.10 F1), Link's 3 s Episode, community 1 s filter.
- Hypothesis: one global decode replaces k-frame counting, the 3 s pending timeout and forward fill. It removes short spurious possessions (deflections, defenders near passing lanes) and does not delay real switches, because Viterbi places the boundary at the most likely frame.
- Prediction: fragmentation down; predicted/true possession count closer to 1.0 on Metrica; the "wrong team" error class (153 at iteration 1) shrinks 20-40%. spellF1 +0.01 to +0.03. Risk: true possessions under 1 s in the truth get merged, so check the truth's duration histogram first.
- Smallest implementation: per-frame emissions with three cases. When the ball is visible and live: log-likelihood from team-zone distances (a soft version of the current rule, e.g. a logistic in d_home - d_away). When the ball is hidden: flat, or the H3 classifier. D: the deadball2 probability. Transitions: fixed switch cost tuned on the tune split. Optional: minimum duration by state expansion (chain of m sub-states). Pure numpy Viterbi, about 60 lines.

### H3 - Player-only team classifier for ball gaps (feeds H2)
- Source: PathCRF (92.4% team accuracy with no ball), Kim 2023, Wei 2015 (role-ordered features), Lang 2022 (tree models on player-only features).
- Hypothesis: during a 5-30 s gap, team shape already tells who has the ball: the attacking team stretches, moves toward the opponent goal and pushes up its full-backs, while the defending team compacts and moves backward. A cheap classifier can detect turnovers inside gaps (about 25% of gaps contain one) and date them better than "the moment the ball returns".
- Prediction: within gaps that contain a turnover, the "merged turnover" error class (137 at iteration 1) shrinks 25-50%. spellF1 +0.01 to +0.04. Visible-frame accuracy is unchanged.
- Smallest implementation: team-level features per frame (centroid x-velocity toward the attack direction, team length and width, x of the defensive line, number of players in the opponent half, each with a ±2 s window mean), labelled from tune-split truth on visible live frames. Train sklearn HistGradientBoosting and use the calibrated probability as the H2 emission during gaps. Train on Metrica G1/G2 and IDSSE 1 only, and keep held-out clean. Do not use PathCRF weights: they were trained on IDSSE, so they would leak held-out data.

### H4 - Zone radius per tracking source (or an adaptive radius) to fix IDSSE under-switching
- Source: Vidal-Codina (0.5 m vs 1 m by provider), third-party IDSSE carrier radius 1.5 m, Metrica places the ball on the carrier.
- Hypothesis: r_pz = r_dz = 1.0 m was tuned where the ball sits on the carrier. On TRACAB-style coordinates and on the challenge data, the true carrier is often 1-2 m away, so the new team is rarely "alone in its zone" and turnovers are missed (286 vs 511 possessions on IDSSE).
- Prediction: on IDSSE the possession count rises toward 511 and spellF1 rises, while Metrica changes little if position noise is applied. Choose the radius on IDSSE match 1 plus Metrica with noise, never on the challenge data.
- Smallest implementation: a grid over r_pz in {1.0, 1.25, 1.5, 2.0} and r_dz in {r_pz, r_pz+0.5} on the tune split with the existing bench script. Alternatively set the radius to the 80th percentile of the nearest-player distance on visible live frames for each match (no labels needed).

### H5 - Control evidence instead of counting: ball speed relative to the nearest player
- Source: Link & Hoernig (control from duration, ball speed and acceleration, opponent pressure), Vidal-Codina (ball flying over a player is not a gain), unravelsports warning about aerial and contested balls.
- Hypothesis: a defender near a passing lane "wins" on distance alone because the ball passes through the zone at speed. Require low relative velocity |v_ball - v_nearest_teammate| < v_c (or a decelerating ball) for a frame to count as control. Duel frames and pass-through frames then stay with the current team.
- Prediction: wrong-team errors on visible frames (about 6%) drop by 1-2 points; few changes elsewhere. spellF1 +0.002 to +0.01. Expect a near-zero effect on Metrica (the ball is glued to the carrier, so relative velocity is about 0) and the main effect on IDSSE.
- Smallest implementation: compute ball velocity (Savitzky-Golay, already in MatchLens) and the nearest player's velocity per team; add `and v_rel <= v_c` to `h_in` / `a_in`. Tune v_c in {3, 5, 8} m/s.

### Ranking (expected gain against cost)
1. **H1 back-dating:** near-certain direction and about 4 lines.
2. **H4 per-source radius:** a grid run only, aimed at the largest known IDSSE gap.
3. **H2 HMM decode:** largest structural gain; it also becomes the frame that holds H3 and the dead-ball evidence.
4. **H3 player-only gap classifier:** the biggest upside for the 41% of frames without a ball, at the most build cost and risk.
5. **H5 relative-velocity control:** small and data-dependent; mostly an IDSSE fix.
