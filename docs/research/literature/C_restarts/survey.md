# Cluster C survey: stoppages, restart frames and the restart team

Date 2026-11-12. All sources were read through web-search snippets only (WebFetch is blocked), so thresholds inside the papers are mostly **not found**. Each note says what was checked.

Baseline (LOOP.md, iteration 3, tune split): spellF1 **0.3747**; dead-ball recall **0.63**, precision **0.78**; perfect stoppages would give about **0.557**. After a stoppage the pre-stoppage team is kept.

## Sources
| Note | Source | What it gives us | Stoppage start/end | Restart team | Numbers |
|---|---|---|---|---|---|
| [vidal-codina-2022-event-detection](vidal-codina-2022-event-detection.md) | Vidal-Codina et al. 2022, Sports Eng. | Rule tree: possession, then set-piece triggers on dead intervals; executor within R_pz at the first live frame | Takes the provider flag or missing ball. **Does not detect stoppages** | Executor's team | >90% detection in most categories; set-piece F1 85% -> 95% with error handling |
| [mills-2026-broadcast-event-detection](mills-2026-broadcast-event-detection.md) | Mills et al. 2026, Sports Eng. | The same algorithm on broadcast tracking | Live switched too early -> wrong set piece | (typing) | Throw-in F1 0.87 (0.97 multi-camera), ±3 s match |
| [lang-2022-ingame-status](lang-2022-ingame-status.md) | Lang et al. 2022, Sci Rep | In play vs stopped from **players only** (AdaBoost) | Frame level | — | 92% frame accuracy; 81% precision for stoppages > 2 s |
| [gudmundsson-wolle-2014](gudmundsson-wolle-2014.md) | Gudmundsson & Wolle 2014, CEUS | Ball-out from ball + players | — | — | F1 0.94 (as cited, not checked) |
| [bassek-2025-idsse](bassek-2025-idsse.md) | Bassek et al. 2025, Sci Data | IDSSE BallStatus / BallPossession at 25 Hz | How BallStatus is produced: **not found** (probably set by an operator, not checked) | BallPossession during D may carry it (check) | 7 matches |
| [soccernet-v2-2021](soccernet-v2-2021.md) | Deliège et al. 2021, CVPRW | Definitions: ball out = line crossing; throw-in = release; kick-off = first pass | Start = crossing, end = restart action | — | About 64 ball-outs and 38 throw-ins per game |
| [soccertrack-v2-2025](soccertrack-v2-2025.md) | SoccerTrack v2 2025, arXiv | Per-frame ball-status flag nearly gives away the out-of-play label | — | — | — |
| [soccercpd-2022](soccercpd-2022.md) | Kim et al. 2022, KDD | Set pieces break roles (switch rate near 1, threshold about 0.9) | Set-piece windows | — | Needs player IDs |
| [fassmeyer-2021-labeling-situations](fassmeyer-2021-labeling-situations.md) | Fassmeyer, Anzer, Bauer, Brefeld 2021, Frontiers | VAE + few labels detect corners | Windows only | — | AUC > 0.90 |
| [billings-mit-thesis-setpieces](billings-mit-thesis-setpieces.md) | Billings, MIT thesis; AAU corner MSc 2024 | Learned set-piece type from motion | — | Via type | Results not found |
| [ifab-laws-restarts](ifab-laws-restarts.md) | IFAB Laws 8, 9, 13, 15, 16, 17 | Restart geometry and exclusion zones | Law 9: whole ball over the line | Goal kick = defending team; corner = attacking team; after a goal = the team that conceded; throw-in and free kick = team inside the exclusion zone | 9.15 m, 2 m, 1 m arc, 5.5 m area |
| [metrica-smart-tagging](metrica-smart-tagging.md) | Metrica Smart Tagging (local DATA.md) | 59 set pieces with type; team on about 33 (all throw-ins N/A) | Tag interval | Tag Team (non-throw-ins) | Method **not found** publicly |
| [statsbomb-opta-conventions](statsbomb-opta-conventions.md) | StatsBomb / Opta docs | Each restart starts a new possession owned by the restarting team | — | Restarting team | How often the team flips at a restart: **not found** |

## Gaps
1. **No published method detects stoppages without a provider flag except Lang et al.**, and its feature list and recall were not found. The other systems take ball status as input.
2. **No published accuracy for restart-team inference from tracking.** Vidal-Codina types set pieces (and so the executor) but its team accuracy was not found.
3. Exact trigger geometry in Vidal-Codina (Fig. 5) and the R_pz used at the restart: not found. The IFAB distances are the fallback.
4. How DFL BallStatus is produced and its timing (whistle or line crossing): not found. This decides where our D should start and end to match IDSSE truth.
5. How often the restart team differs from the team before the stoppage: not found. **Measure it on our bench first.** It caps the gain of C1.
6. Stoppages without a line cue (fouls, offside, injuries) get no help from the Law 9 geometry. Only collective slowdown (Lang, swarm F3) and the free-kick exclusion zone cover them.

## Candidate fixes (testable hypotheses)
spellF1 predictions extrapolate linearly from the oracle: recall 0.63 -> 1.0 and precision 0.78 -> 1.0 gave +0.18, so **about +0.04 spellF1 per +0.10 recall at fixed precision**. This is a rough planning number, not a measurement.

### C1. Restart team from restart geometry and exclusion zones (IFAB 8/13/15/16/17; Vidal-Codina executor)
- **Hypothesis:** at the end of each predicted D stretch, the restart team can be read without last touch: (a) ball in a goal area -> the team defending that goal; (b) ball within about 1.5 m of a corner flag -> the team attacking that goal; (c) ball at the centre spot and both teams split by the halfway line -> the team without the ball before (after a goal, the team that conceded); (d) otherwise, the team with players within 2 m of the still ball (throw-in) or alone inside 9.15 m (free kick) over the last 1 s of D; (e) fallback: the executor, i.e. the first team alone in its 1.0 m zone at the first live frame. Set the team at once (skip the k=5 challenge).
- **Prediction:** restart-team accuracy goes from the keep-team baseline (measure first; expected 50–65%) to **≥ 85%** on true D ends. spellF1 **+0.01 to +0.03** on tune. Each wrong restart currently costs the start of the post-restart spell plus a phantom switch k frames later. Possession count on IDSSE rises towards the truth (286 -> more).
- **Smallest implementation:** about 40 lines in `team_control.py`: on the D -> live transition, call `restart_team(frame_window, side_of_team)` and set HoldX directly. Measure first with a 10-line diagnostic: the truth team at the first live frame after each true D, against the team before D.

### C2. Set-piece configurations as dead-ball evidence and END-frame anchor (Vidal-Codina triggers; IFAB geometry; Mills timing)
- **Hypothesis:** many missed stoppages are the long set-up phases of corners, goal kicks, kick-offs and free kicks. Add D evidence when the ball is (nearly) still at a restart spot: corner arc (≤ 1.5 m from a flag), goal area with no attacker in the penalty box, centre spot with both teams in their own halves, or a still ball with one team excluded from a 9.15 m ring. Keep D until the executor's kick (ball speed > about 3 m/s away from the spot), not until the first ball movement (Mills: an early live switch causes errors).
- **Prediction:** dead-ball recall **0.63 -> 0.70–0.73**, precision ≥ 0.78 (these configurations are rare in open play). spellF1 **+0.03 to +0.04**.
- **Smallest implementation:** 4 boolean evidence functions in `deadball2.py`, OR-ed into the existing evidence before hysteresis, with their own short `on` count (about 10 frames), because spot geometry is stronger evidence than "far + near a line".

### C3. Players-only stoppage classifier for ball-gap frames (Lang et al. 2022)
- **Hypothesis:** on the 41% of frames without a ball, deadball2 has almost no evidence. A gradient-boosted or AdaBoost classifier on rolling team features (mean and 90th-percentile speed per team, acceleration, both-team speed ratio, team spread and overlap, change over 2–5 s windows) trained on truth D from the tune split recovers stoppages there.
- **Prediction:** dead-ball recall on ball-hidden frames rises clearly; overall recall **0.63 -> 0.70**, precision **0.75–0.80** (Lang: 81% precision for stoppages > 2 s). spellF1 **+0.02 to +0.04**. Risk: slow build-up looks like a stoppage (swarm F3), so use it only while the ball is hidden and require 2 s or more.
- **Smallest implementation:** `scripts/train_dead_clf.py` (sklearn HistGradientBoosting, about 12 features, tune split only) plus a probability threshold and hysteresis in `deadball2.py`, used only on frames without a ball.

### C4. Back-date the D start to the line crossing (Law 9, SoccerNet definition)
- **Hypothesis:** the 50-frame `on` hysteresis starts D about 2 s late, and those frames stay in the old spell. When D is confirmed, back-date its start to the last frame where the ball crossed a line, or where the ball's nearest-player distance first passed r_far. The same applies at the end: stop D at the executor's kick (C2).
- **Prediction:** D frame recall **+0.03 to +0.05** with no change to stoppage-level recall. spellF1 **+0.005 to +0.015**, more if spellF1 matches spells by overlap (check `possession_metrics.py`).
- **Smallest implementation:** a back-fill pass after hysteresis in `deadball2.py` (about 15 lines).

### C5. Smart Tagging SET PIECES as anchors on the challenge match only
- **Hypothesis:** the 59 set-piece tags mark true restarts. Force D on [tag start − 2 s, executor kick], and use the tag Team for the restart team on the about 33 non-throw-in tags (throw-ins: C1).
- **Prediction:** **cannot be measured on the bench** (Metrica Sample and IDSSE have no tags). On the challenge it touches at most about 59 of an expected 80–100 stoppages. Effect unknown. Check only on the public window, once.
- **Smallest implementation:** a loader is already in `starter/load_smart_tagging.py`; add an override layer after `deadball2`.

## Suggested order
Run the C1 diagnostic first (how often the restart team flips; caps C1). Then C2, the cheapest recall gain, then C3, then C4. C5 only as a final challenge-only layer.

## Measured check (orchestrator, 2026-11-12, external bench truth)
Which team owns the ball right after a stoppage, compared with the team right before it:

| match | stoppages between two live spells | same team | other team |
|---|---|---|---|
| metrica1 | 154 | 28 (18%) | 126 (82%) |
| metrica2 | 167 | 36 (22%) | 131 (78%) |
| idsse1 | 101 | 10 (10%) | 91 (90%) |

Our pipeline keeps the pre-stoppage team, so it starts wrong on ~80-90% of restarts. A prior of "other team restarts" is right ~85% before any position evidence; C1 (restart team from geometry) should start from that prior.
