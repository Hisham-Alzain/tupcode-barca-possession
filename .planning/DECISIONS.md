# DECISIONS (ADRs)

Format: `## ADR-NNN · date · title` then Status, Context, Options considered, Decision, Why, Decided by, Reversible?, Consequences.

## ADR-000 · 2026-11-11 · Scope and operating model
**Status:** accepted
**Context:** 7-day hackathon; AI tools must never see challenge data (docs/RULES.md).
**Decision:** Queen/worker model from Padel Lens (docs/AGENTIC_PLAYBOOK.md): envelopes, worktree builders, cold QA per branch, ADRs, STATE/LOG. Councils only for hard-to-reverse choices. All tuning on external data; challenge data run locally by a human only.
**Why:** Proven on Padel Lens; the rules make data isolation the top constraint.
**Decided by:** user
**Reversible?:** yes

## ADR-001 · 2026-11-11 · Agents may read the raw challenge data
**Status:** accepted
**Context:** ADR-000 and docs/RULES.md forbade any AI agent from reading `data/challenge/` (organisers' first clarification email). All challenge runs were blind scripts printing aggregates only.
**Decision:** Allow agents (incl. a Haiku swarm) to read raw challenge inputs, on the strength of the organisers' second "Clarification of the rules" email (hackathon.talentarena@mobileworldcapital.com, Wed 11 Nov 2026 17:01): "Yes you can use agents to read raw data", stated to take precedence over earlier information. Screenshot provided by the user in session; full rule list transcribed verbatim into docs/RULES.md. Core line: train and tune on external data only, never on the GT or the video; inspecting the provided data to understand possession is allowed.
**Still binding:** no hand labels on the game; no tuning/choosing on the public 5 min; no use of private GT; no frame-specific rules or constants fitted to this game; data not in the public repo.
**Options considered:** (a) keep blind-only; (b) allow agents on raw data. Chosen (b) per the organisers' written answer.
**Decided by:** user, on organisers' email
**Reversible?:** yes — if the full email text restricts this, revert settings.json and RULES.md.
**Consequences:** settings.json read-deny on `data/challenge/` removed (write-deny kept); technical document must cite this email.

## ADR-002 · 2026-11-11 · External bench design and first fix (loop iterations 0-1)
**Status:** accepted
**Context:** Need a validation set that behaves like the challenge input (LOOP.md protocol).
**Decision:** scripts/bench.py. Truth from Metrica events (last team; BALL LOST/BALL OUT/CHALLENGE keep the previous team) and ball-absent = dead (port of footballtactics metrica_replay._ground_truth_possession). Pipeline input: ball made visible during true stoppages (Metrica drops it; challenge keeps it on ~72% of D frames, swarm F1), then ~40% hidden in blocks drawn from the challenge gap-length histogram (input statistics only). Official possession_metrics scorer. Iteration 1: MatchLens ghost dead-ball rule disabled (run_matchlens(ghost_dead=False)).
**Why:** Without visible dead balls, "missing = dead" is trivially true on Metrica and the bench would reward the wrong rule. Ghost off: possession count 381/369 -> 356/350 (truth 357/330), mean length 5.4 s -> 10.2 s (truth ~10 s); spellF1 0.316 -> 0.326, CIs overlap.
**Options considered:** keep ghost with a longer threshold (rejected: gaps are 5-30 s, any threshold below that splits); drop ghost entirely (chosen).
**Decided by:** loop (Opus), per swarm finding F1
**Reversible?:** yes (flag)

## ADR-003 · 2026-11-11 · Team-level possession rule replaces MatchLens's per-player possessor (loop iteration 2)
**Status:** accepted (pending user review of the loop as a whole)
**Context:** Oracle ablation on the external tune split: making the team right on visible-ball frames lifts spellF1 0.326 -> 0.608; perfect stoppages only 0.436. MatchLens's PossessionTracker abstains on 18-20% of those frames until a touch is confirmed; the scorer's forward-fill then carries the previous team across hand-overs.
**Decision:** src/possess/team_control.py: MatchLens/Vidal-Codina possession and duel zones evaluated per team; a gain commits after k consecutive control frames; duel, no-one-in-zone and hidden ball keep the current team. Stoppages still from MatchLens DeadBallDetector (ghost off). Parameters chosen on the tune split only: r_pz = r_dz = 1.0 m, k = 5 frames (grid in eval/results/tune_team_control.json).
**Why:** tune spellF1 0.3256 -> 0.3562 (both matches up; CIs overlap, so held-out not opened).
**Reversible?:** yes (pipeline flag).

## ADR-004 · 2026-11-12 · Self-contained repository (no outside code)
**Status:** accepted
**Context:** The pipeline imported the private MatchLens repo through a hard-coded D:\ path and ran on that repo's virtualenv; no pinned dependencies; tests failed on a clean checkout. The submission repo must regenerate the file on its own (SUBMISSION.md).
**Decision:** Port only what the pipeline uses: MatchLens DeadBallDetector + PlacedBallDetector + SG ball speed -> src/possess/stoppage_ml.py (ghost rule off by default, ADR-002; GK-holding override not wired), Metrica CSV loader -> src/possess/metrica_csv.py. Drop the rest of MatchLens (Eventer, per-player possession tracker, set-piece/event detectors, direction, schema) and the scripts that ran it. Vendor the organisers' scorer (MIT). Pin exact versions in pyproject.toml + uv.lock (Python 3.12).
**Evidence:** ported dead-ball state agrees with the full MatchLens Eventer run on 100.0% of Metrica G1 frames (recall 0.532, precision 0.726 both); all other features identical to the loop-3 caches; fresh clone + `uv sync` + `uv run pytest` = 46 passed.
**Reversible?:** yes (git history keeps matchlens_ref and the Eventer scripts).

## ADR-005 · 2026-11-13 · Whole-half Viterbi decoder becomes the default (plan P3)
**Status:** accepted
**Context:** Frame-by-frame team rule (P1+P2) at tune 0.3724. Decoder v1 lost (0.3597). User asked for Fable on hard tasks and validation.
**Decision:** src/possess/decode.py, 4-state Viterbi (H, A, D-after-H, D-after-A) per half: soft log-distance team emission (cap 1.5 m, d0 0.25 m), stoppages fixed by the detector, the frame rule's label as a 0.3-weight emission, switch cost 4, same-team-restart cost 1, switches inside ball gaps placed at the gap middle. Parameters chosen on the tune split only.
**Evidence:** tune 0.3724 -> 0.3976 (all three matches up), paired bootstrap +0.025 [+0.019, +0.032] (Fable QA); held-out (look 1) 0.3556 -> 0.3625 (both matches up); recursion exact vs exhaustive search; no truth leakage. Public 5 min 0.454 -> 0.444 (reported only; one possession).
**Risks:** 22 configs on 3 matches (held-out gain smaller than tune gain); `cap` sensitive (1.0 -> 0.373); IDSSE under-segments (243/511); needs w_rule > 0 to stay home/away symmetric.
**Decided by:** Fable designer + Fable QA, orchestrator.
**Reversible?:** yes (`pipeline.DECODER = "rule"`).

## ADR-006 · 2026-11-14 · Team-shape models on top of the decoder (plan P5)
**Status:** accepted
**Context:** The ball is hidden on 41% of graded frames, mostly in 5-30 s blocks; oracle showed perfect team on hidden frames worth ~+0.13. Decoder (ADR-005) at tune 0.3976, held-out 0.3625.
**Decision:** src/possess/shape.py: two small numpy logistic models on ID-free, count-robust player-position features (relative team shape, line heights, 0.8 s-lag speeds, attacking direction by per-half vote) — one gives team evidence while the ball is hidden, one gives stoppage evidence — added as decoder emissions (pipeline.DEC = DEC_BASE + shape.DEC_SHAPE when positions are present). Models trained on the tune matches only; tune numbers are leave-one-match-out.
**Evidence:** tune 0.3976 -> 0.4932 (all three up); held-out look 2: 0.3625 -> 0.4789 (both up); noise gate passes (10 cm jitter 0.4923); 7-12 players per team 0.4563; two Fable QA rounds (no leakage, no bench artefact, jitter fixed). Public 5 min 0.444 -> 0.433 (same 19/17 possessions; reported only).
**Risks:** challenge tracking noisier than the bench; decoder weights chosen on the tune matches; low findings in BACKLOG.
**Decided by:** Fable designer + 2 Fable QA rounds, orchestrator.
**Reversible?:** yes (`--no-shape` / drop positions).

## ADR-007 · 2026-11-14 · Touch emission for short possessions (loop L2-8)
**Status:** accepted
**Context:** Diagnosis on tune: half of true possessions last < 4 s (deflections, tackles, clearances); the decoder caught 11% of those under 2 s.
**Decision:** decode.touch_hold — team-level port of MatchLens events/impulse.py impulse_verdict (direction/speed change thresholds unchanged, scaled by ball noise estimated from the same half without labels): a touch within 1 m of one team makes that team the holder until the next touch, at most 4 s; emission weight w_touch 0.7 in pipeline.DEC_BASE (and DEC).
**Evidence:** tune 0.4932 -> 0.5657; held-out look 3: 0.4789 -> 0.5584 (both up); Fable QA adopt; robust to 5-10 cm noise; 1-frame teleports every 2 s still 0.5376.
**Risks:** part of the tune gain is at stoppage boundaries (Metrica fill-line); the challenge ball has jumps and stale values -> ball cleaning before touch_hold (next).
**Reversible?:** yes (w_touch = 0).

## ADR-008 · 2026-11-14 · Ball cleaning before touch detection (robustness)
**Status:** accepted
**Context:** The challenge ball has jumps up to 33-40 m/s and frozen repeats; touch detection (ADR-007) fires on them (Fable QA).
**Decision:** src/possess/ballclean.py cleans only the ball fed to decode.touch_hold: teleport runs (<= 3 frames, faster than 35 m/s, >= 1.5 m off the line between neighbours, no player within r_touch) and frozen repeats of <= 10 frames become hidden and are bridged; the off-pitch rule exists but is off.
**Evidence:** clean tune 0.5657 -> 0.5679 (idsse1 unchanged); with injected faults the touch-driven score keeps most of its value (teleports every 2 s 0.523 -> 0.561); two Fable QA rounds; real kicks kept (idsse1 removed alive frames 243 -> 58, all frozen repeats).
**Risks:** long freezes (> 0.4 s) and faults in the distance/stoppage features are not cleaned.
**Reversible?:** yes.

## ADR-009 · 2026-11-14 · Team-context features for hidden-ball turnovers (loop L2-10)
**Status:** accepted
**Context:** Turnovers inside hidden-ball blocks were caught 8% of the time (diagnosis L2-8).
**Decision:** shape.TEAM_CTX = 1: the hidden-frame team model also sees how each team's shape changes over +-2 s (player positions only); models fit on the tune matches, LOMO for scoring.
**Evidence:** tune 0.5679 -> 0.5845, all six halves up, paired bootstrap +0.0167 [0.0068, 0.0275]; held-out look 4: 0.5584 -> 0.5762 (both up); noise gate passes (sigma 0.10 m 0.5835, 7-12 players 0.5487); Fable QA adopt.
**Risks:** 15 feature sets tried on the same matches (gain ~3 sigma); IDSSE alone not significant; coverage on the challenge's 7-12 players unknown.
**Reversible?:** yes (shape.TEAM_CTX = 0).

## ADR-010 · 2026-11-14 · Reject the still-ball / kick-restart stoppage rule (loop L2-11)
**Status:** rejected (code kept opt-in: pipeline.DEAD2 still_v=0, v_kick=0)
**Context:** Rule lifted tune 0.5845 -> 0.6125 on all three tune matches.
**Why rejected:** Fable QA transfer check on the provided match's input (no ground truth): a still ball is 4.5% of visible frames on the real match vs 21-42% on the bench, because IDSSE freezes the dead ball at speed 0 and Metrica's dead ball is drawn by the bench. The kick-restart part ends real stoppages when the ball is thrown back to the spot; longest dead runs grew to 199 s on the bench. The gain would not transfer.
**Consequence:** new protocol rule — ball-in-stoppage features need an input-level transfer check on the provided match before adoption (LOOP.md).
**Decided by:** Fable QA, orchestrator.

## ADR-011 · 2026-11-14 · Trust player-only stoppage evidence on visible frames (loop L2-13)
**Status:** accepted
**Context:** Audit L2-12: stoppage detection fails when the dead ball moves; decisions now on the realistic bench real1.
**Decision:** decoder weights: sdead_vis 2.5 (player-only stoppage model weighted 2.5x on visible frames), w_live 0 (the ball detector's "live" verdict is no evidence). Previous config kept as shape.DEC_SHAPE_L2_10.
**Evidence:** real1 tune 0.5276 -> 0.5731 (paired bootstrap +0.047 [0.033, 0.062], all three up); real1 held-out 0.5227 -> 0.5557; orig tune 0.5845 -> 0.6002; orig held-out 0.5762 -> 0.5716 (-0.0046, within the allowance; IDSSE slightly down on orig). Transfer check: on the provided match's input the change mostly extends existing stoppages (D share 30.9% -> 34.9%); expected real-match effect about 0 (ball mostly lost during real stoppages), downside bounded. Public 5 min 0.4588, 17 predicted vs 17 true possessions.
**Reversible?:** yes (DEC_SHAPE_L2_10).

## ADR-012 · 2026-11-14 · Distance evidence capped at the 1 m control zone (loop L2-14)
**Status:** accepted
**Decision:** shape.DEC_SHAPE cap 1.0 (was 1.5): the soft distance emission no longer grades beyond MatchLens's 1 m possession zone; touch hold and team-shape evidence now cover the 1-1.5 m band.
**Evidence:** real1 tune 0.5731 -> 0.5834, held-out 0.5557 -> 0.5619; orig tune 0.6002 -> 0.6110, held-out 0.5716 -> 0.5787; paired bootstrap +0.007..0.009 (borderline after ~35 candidates; IDSSE null); transfer: possessions per half hour -3.2% on the provided match vs -1.1..-4.4% on the bench. Expected real-match effect small (~0 to +0.003): its ball-proximity profile resembles IDSSE.
**Rejected alongside:** removing the frame-rule emission (w_rule 0) scored higher on the bench but removed 24% of possessions on the provided match (bench 1-7%): would not transfer.
**Reversible?:** yes (cap 1.5).

## ADR-013 · 2026-11-14 · Contested touches keep the last clean toucher's hold (loop L2-15)
**Status:** accepted
**Decision:** decode.touch_hold `duel=2`: a contested touch (both teams within 1 m of the ball) no longer clears the touch hold; the last uncontested toucher keeps it (MatchLens / Vidal-Codina duel rule, already used in team_control). Nothing learned. ball_impulse split out of touch_hold, behaviour identical (QA: 0 mismatches on 1800 random cases). shape.DEC_SHAPE = dict(DEC_SHAPE_L2_14, duel=2).
**Evidence:** real1 tune 0.5834 -> 0.5935 (metrica1 0.649->0.659, metrica2 0.591->0.598, idsse1 0.510->0.523; possessions 312/357, 272/330, 361/511); orig tune 0.6110 -> 0.6205 (0.675, 0.633, 0.553). Public5 0.4588 17/17 unchanged (reported, not used to choose). Held-out not looked at (still 6 looks). Transfer check: near-inert on the provided match (predicted possessions per half hour 112.46 -> 112.46; contested touch onsets 0.16/min vs 1.6-3.0 on the bench); draft submissions/draft_2026-11-14g.csv validated, differs from draft f by one frame (H 60272, A 42538, D 56047, X 40462). Expect ~0 gain on the real match. Fable cold QA PASS (`envelopes/L2-15-qa-fable.out.md`).
**Risks (non-blocking QA findings):** no paired bootstrap, ~45 candidates tried (selection bias); worse than duel=0 at 0.1-0.2 m ball noise (provided ball noise estimated 0.000); scripts/diagnose.py:204 lacks a length guard (diagnostics only).
**Reversible?:** yes (shape.DEC_SHAPE duel=0, or DEC_SHAPE_L2_14).

## ADR-014 · 2026-11-14 · Fast-player speed in the stoppage model (loop L2-16); held-out look 7 (last)
**Status:** accepted
**Decision:** the LOMO player-only stoppage logistic gets the speed of the fast players: 90th percentile of all tracked players' speeds (both teams), 1 s and 5 s centred means (ps90, ps90_5), model `dead_fast` (shape.DEAD_FAST=1). shape.MODELS refit on the three tune matches only. Player-only, no ball. Motivation: L2-10/L2-16 oracle (perfect hidden-block stoppages worth +0.040; loss concentrated at the edges of stoppages >= 3 s; hidden restarts predicted 1.2-1.7 s early).
**Evidence:** real1 tune 0.5935 -> 0.6061 (metrica1 0.665 308/357, metrica2 0.615 274/330, idsse1 0.538 368/511); orig tune 0.6205 -> 0.6283 (0.685 302, 0.647 262, 0.553 371). Public5 0.4588 -> 0.5294 17/17 (reported, never used to choose). D recall/precision hidden 0.792/0.872 -> 0.804/0.889, visible 0.788/0.932 -> 0.818/0.927. Noise gate real1: sigma 0.05 0.6046, 0.10 0.6037, 7-12 players 0.5333, both 0.5323 (all above the L2-15 default).
**Held-out look 7 (last):** real1 held-out 0.5788 (metrica3 0.634 343/386, idsse2 0.523 322/421) vs 0.5619 at L2-14; orig held-out 0.5968 (metrica3 0.651 322/386, idsse2 0.542 310/421) vs 0.5787. Reason for spending it: clear QA-passed win; L2-17's expected gain is too small to qualify alone. No held-out looks remain before submission; held-out numbers are now slightly optimistic.
**Transfer:** possessions per half hour on the provided match unchanged (112.46); hidden D share on the provided match 0.673 -> 0.711 while the real1 bench is flat; dhide sensitivity runs keep the gain (0.5890 -> 0.6013, 0.5899 -> 0.6004); QA input-only aggregates: hidden-ball frames on the provided match are genuinely slow (ps90 median 1.62 vs 3.38 m/s visible, same at equal player count). Residual risk: over-calling D on the real match.
**QA:** Fable cold QA PASS (`envelopes/L2-16-qa-fable.out.md`): LOMO intact incl. mu/sd; shipped MODELS reproduce to 4.5e-5; quantile robustness 0.5/0.75/0.9/1.0 -> 0.5937/0.6028/0.6061/0.6025; time-shifted control 0.5921. Non-blocking: speed smoothing crosses half-time (~60 frames per half); diagnose.py restores DEAD_FAST=1 hard-coded; builder envelope wording "public truth never opened" (the evaluate run reported public5, allowed as reporting).
**Not adopted:** w_sdead 0.4 (one run 0.6102, neutral); p75 / top-3 / max quantile variants 0.602-0.606 (neutral).
**Draft:** submissions/draft_2026-11-14h.csv validated: 397 possessions, mean 10.1 s; H 59272, A 40876, D 58709, X 40462; Barca share 0.592.
**Reversible?:** yes (shape.DEAD_FAST=0).

## ADR-015 · 2026-11-14 · Snap hidden stoppage exits to the fast-player speed step (loop L2-18)
**Status:** accepted
**Context:** hidden-ball stoppage exits are predicted early (median -1.02 / -0.94 / -1.12 s on the three tune matches). Diagnosis: the cause is the smear of the centred 5-15 s windows in the stoppage model (H1), not real pre-movement. Constant decoder costs cannot move an edge (M6 control). Stop rules fired for M5 (lone taker), M3 (edge-aware model) and M2 (entry shift).
**Decision:** src/possess/edges.py post-pass in decode_halves: the exit of every predicted stoppage >= 3 s that ends on a hidden-ball frame moves to the largest step-up of the raw fast-player speed (p90 of all tracked players, 1 s window) within +-2 s. Nothing learned. shape.DEC_SHAPE ed_exit=1, ed_w=1.0, ed_win=2.0; previous default kept as shape.DEC_SHAPE_L2_16. Values chosen by analogy with stoppage_ml.py:97,107 (not a port). The grid ran ed_w {1,2} x ed_win {2,4}: all exit-only cells 0.6075-0.6084 vs 0.6061 off; the chosen cell is not the best one (QA: "not tuned" is overstated, it was a 2x2 grid).
**Evidence:** real1 tune 0.6061 -> 0.6081 (metrica1 0.670, metrica2 0.615, idsse1 0.540; possession counts unchanged 308/274/368); orig 0.6283 -> 0.6305 (0.690, 0.647, 0.554). Paired bootstrap +0.0020, CI90 [+0.0002, +0.0041] after ~120 candidates; gain mostly metrica1. Noise gate 0.6068 / 0.6060 / 0.5349 vs 0.6046 / 0.6037 / 0.5333 without. dhide 0.8 / 0.95 +0.0042 each. Public5 0.5294 17/17 unchanged (reported only). Held-out not looked at (no looks left): look-7 numbers stand (real1 0.5788 / orig 0.5968, measured at L2-16).
**Transfer:** provided-match hidden D share 0.711 -> 0.724 (bench +0.005..+0.014); 36 of 83 D runs >= 3 s moved, 32 later / 4 earlier, median +1.36 s; possessions unchanged. Expect ~0 real effect.
**Not adopted:** exit+entry +-2 s (real1 0.6102, orig 0.6332) failed the designer's own D-precision gate (idsse1 hidden D precision -0.015 vs limit 0.01); gate not relaxed. Exit+entry +-4 s 0.6117 lengthens D (precision -0.019..-0.027). M4 cue as transition cost 0.6058-0.6086 (flat). M2 exit shift +0.0012. L2-19 (ball cleaning before features) specced and deprioritised: real1 has more ball faults than the provided input (cleaner removes 0.0% there).
**QA:** Fable cold QA PASS (`envelopes/L2-18-qa-fable.out.md`). Low findings: the moved span may include visible-ball frames (138 of 1107 changed frames on the provided match); "not tuned" overstated.
**Draft:** submissions/draft_2026-11-14i.csv validated: 397 possessions, mean 10.0 s; H 58780, A 40395, D 59682, X 40462; Barca share 0.593.
**Reversible?:** yes (shape.DEC_SHAPE_L2_16, or ed_exit=0).

## ADR-016 · 2026-11-14 · Bench variant real2 and the co-deciding rule (loop L2-20)
**Status:** accepted
**Context:** real1 (L2-12) still feeds the pipeline a sharp, unsmoothed ball that is hidden in independent blocks; the provided input shows a smoothed ball track, hiding tied to stoppages, and player loss in hidden blocks (provided-vs-bench gap analysis, draft i).
**Decision:** new variant `real2` in scripts/bench.py = real1 + Gaussian-smoothed ball (sigma 7 frames) + state-coupled hiding (blocks on true stoppages from a slow ball to the restart kick, lengths drawn from the provided gap list incl. the long tail) + player loss inside hidden blocks. Constants set from INPUT aggregates of the provided match only (no labels), an evaluation design as real1 was (L2-12). Audit table in `envelopes/L2-20-real2-fable.out.md`. real1 and orig cases are array-identical to before (QA).
**Switch re-score on real2 (old default 0.4303):** touch (ADR-007) -0.024 (vs +0.072 on real1), duel -0.005, ADR-011 +0.002, cap -0.001, dead_fast +0.003, ed_exit 0.000, team context +0.009, shape team/stoppage +0.030/+0.031. Several earlier gains do not survive a smoothed ball.
**Decision rule change (QA recommendation, adopted):** from now real2 and real1 CO-DECIDE. Adopt a change only if neither drops by more than 0.003 and at least one gains; orig drop <= 0.005; paired bootstrap plus a second real2 seed.
**Expected real-match score:** re-estimated as roughly real2-like (~0.48), not 0.53-0.56.
**Reversible?:** yes (evaluation design only; no pipeline change).

## ADR-017 · 2026-11-14 · Touch detection adapted to low-passed ball tracks (smooth_touch; loop L2-20)
**Status:** accepted, PENDING USER CONFIRMATION under the no-calibration rule
**Decision:** shape.DEC_SHAPE smooth_touch=1. When the ball track is low-passed (acceleration p99 < 150 m/s2, decided per half), touch detection uses 8-frame velocities, one touch per velocity-change peak, and a 2 m radius (also the clean_ball keep radius). Otherwise the old detector runs unchanged.
**Evidence:** real2 0.4303 -> 0.4833 (seed 2: 0.4331 -> 0.4823); paired block bootstrap +0.053 [+0.037, +0.070] and +0.049 [+0.030, +0.067]; all 6 match-seed pairs up. real1 0.6081 and orig 0.6305 unchanged (mode off there). Public5 0.5294 -> 0.5059 17/17: reported only, never used to choose, within one-possession noise (recorded honestly in LOOP/STATE). Lag and sigma scans all gain.
**Transfer:** provided-match touch-hold share 0.27 -> 0.67 (real2 0.66-0.75); possessions per half hour 112.5 -> 119.0.
**Compliance (no calibration on challenge data):** the branch exists because of the provided input and fires only there, which is why it is disclosed. External unsmoothed p99 is 204-619 per half; real2 36-41; provided 30 and 47. Any threshold in 48-204 gives identical output on every dataset, so the value inside that gap is not a fitted parameter; the constants (8-frame window, 2 m) were chosen on external real2 truth. Fable QA: defensible; disclose in the submission and technical document. The orchestrator asked the user to confirm.
**Draft:** submissions/draft_2026-11-14j.csv validated: 420 possessions, mean 9.5 s; H 59390, A 40599, D 58868, X 40462; Barca 0.594; 3.8% of frames changed vs draft i.
**Not adopted:** wider impulse lag alone (real2 0.436-0.437); w_touch 0.35 variant (neutral).
**Reversible?:** yes (shape.DEC_SHAPE smooth_touch=0).

## ADR-018 · 2026-11-14 · Per-match unsupervised dead-ball prior on hidden frames (sd_prior; loop L2-21)
**Status:** accepted, PENDING ORGANISER ANSWER (disclosure; question drafted in `.planning/organiser-question.md`, the user sends it)
**Context:** on real2 the stoppage model, whose prior comes from the training matches, under-calls D on hidden-ball frames in matches with a large hidden share.
**Decision:** shape.em_prior / prior_shift / dead_prior. Per match, an unsupervised EM estimate (Saerens 2002) of the dead-ball share on hidden-ball frames, from the stoppage model's own posteriors. Hidden log-odds are shifted by logit(estimate) - logit(pi_train); hidden frames without shape evidence get the estimated prior. Applied only if |shift| >= 0.6 (any value 0.49-0.97 gives identical output everywhere) and at least PRIOR_MIN_FRAMES = 40000 hidden frames with evidence (QA finding 1: validated on full matches only; per half, random-hiding shifts reach 1.4). DEC_SHAPE sd_prior=1; previous default kept as DEC_SHAPE_L2_20.
**Evidence:** real2 0.4833 -> 0.5173 (metrica1 0.569 307/357, metrica2 0.554 287/330, idsse1 0.429 321/511); hidden D predicted vs true on real2 0.727/0.714, 0.819/0.826, 0.838/0.800. Second seed 0.4824 -> 0.5152. Paired bootstrap +0.0339, CI90 [+0.0236, +0.0448]. real1 0.6081 and orig 0.6305 unchanged (gate shut). Public5 0.5059 unchanged (reported only). Noise gate +0.032..+0.035.
**Transfer on the provided match:** estimate 0.869, shift +2.27; hidden D share 0.724 -> ~0.897; D runs per half hour 24.4 -> 30.0.
**QA (Fable, PASS with changes):** EM implementation correct. Transfer: the provided match lies slightly beyond real2 on every input indicator, in the same direction (hidden share 0.41, median block 10.7 s, fast-player speed 1.61, raw model D probability 0.68, no-evidence share 0.18), so a mild extrapolation. Overestimate test on real2 (+0.5 / +1.0 log-odds: 0.5198 / 0.5146) is cheap, so no cap.
**Compliance (no calibration on challenge data):** inference, not fitting: no label, constant or threshold comes from the provided match; the estimate is recomputed per match identically, like the attacking-direction vote. But the method is literally "prior recalibration", so it is DISCLOSED in docs/TECHNICAL.md and covered by the organiser question the user will send. If the organisers refuse, set sd_prior=0 (and smooth_touch=0 if also refused) and regenerate.
**Draft:** submissions/draft_2026-11-14k.csv validated: 416 possessions, mean 8.5 s; H 52968, A 35594, D 70295, X 40462; Barca 0.598; 6.0% of frames changed vs draft j.
**Not adopted:** visible-frame EM (real2 0.5200, more scope); shift cap 1.2 (0.5140).
**Reversible?:** yes (shape.DEC_SHAPE_L2_20, or sd_prior=0).

## ADR-019 — Reappearance touch hold (L2-22) — 2026-11-14, accepted
- **Context:** on real2 the ball vanishes near a player and reappears on a kick; wrong-team share is 0.07-0.18 in the 3 s after a hidden block. Carrying the team at disappearance through the block is NEGATIVE (oracle -0.0175: the team changes inside 52 of 116 live blocks).
- **Decision:** `touch_hold(reappear=1.0)`: the first visible frame after a hidden block >= 1 s starts a touch hold for the only team within 1 m (MatchLens control zone). One config, no grid. `shape.DEC_SHAPE = dict(DEC_SHAPE_L2_21, reappear=1.0)`.
- **Evidence:** real2 0.5173 -> 0.5326 (0.580 / 0.563 / 0.455); Fable QA over 5 real2 seeds: pooled +0.0090 CI90 [+0.0055, +0.0126], median seed +0.0079, unseen seeds 1-4 +0.0075 [+0.0039, +0.0111]; 15/15 seed x match cells >= 0. Costs: real1 0.6081 -> 0.6068, orig 0.6305 -> 0.6283 (orig loss is real, CI90 [-0.0048, -0.0004], inside the 0.005 tolerance). public5 0.5059 -> 0.5294 (reported only).
- **Transfer:** fires on 77/131 provided blocks (bench 44-61%); 0.58% of frames change vs draft k; possessions/half hour 117.8 -> 120.1.
- **QA findings (open, low):** NaN distance counts as team absent (decode.py:194); smooth regime uses 2 m touch radius but 1 m reappear radius.
- **Draft:** submissions/draft_2026-11-14l.csv (424 possessions, H 52892, A 36255, D 69710, X 40462, Barca 0.593). Commit e4ee541. Reversal: DEC_SHAPE_L2_21 or `reappear=0`.

## ADR-020 — Vertex touch attribution on heavily low-passed ball tracks (L3-A4) — 2026-11-15, accepted (PENDING ORGANISER ANSWER, as ADR-017)
- **Context:** L3 ran four independent Opus solvers (paper audit, switch timing, dead ball, short possessions). A2 measured that on real2 switches are late (median +3 frames, duels +19) and that 102 of 146 late switches have no detected touch; A4 found why: on a low-passed track the detected touch point sits ~2 m from the player's feet, so the touch is credited to nobody or contested. Vidal-Codina §2.3.3 compares incoming and outgoing ball direction over the whole control interval.
- **Decision:** `decode.touch_vertex` rebuilds the contact point as the meeting point of the straight incoming and outgoing ball flight; the touch goes to the team nearest that point within 2 m (`shape.DEC_SHAPE vtx=2.0`). Runs only when the track is HEAVILY low-passed: acceleration p99 < VTX_ACC = 70 m/s2 (external sigma 5-7 tracks <= 57, sigma 3 >= 89; chosen on external data).
- **Evidence:** real2 0.5326 -> 0.5618 (metrica1 0.604 271/357, metrica2 0.582 278/330, idsse1 0.499 302/511); 6 seeds pooled +0.029 CI90 [+0.017, +0.042] (metrica1 neutral, metrica2 +0.031, idsse1 +0.057); real1 0.6068 / orig 0.6283 byte-identical. Sensitivity (Fable QA): sigma 10 +0.073, sigma 7 +0.029, sigma 5 +0.008 (n.s.), sigma 3 -0.031 before the guard, 0 frames changed with it. Uncontested touch team right 0.975-0.986 (was 0.94-0.97). public5 0.5294 -> 0.5300, 23 predicted vs 17 true (reported only).
- **Transfer:** provided match is in the heavy regime (p99 30.5 / 46.5; vertex-ball offset 2.6 m vs 1.7-2.0 at bench sigma 7); 2.5% of frames change vs draft l; possessions 424 -> 412.
- **Known limits:** touches < 0.5 s apart (one-twos) are placed 1.6-3.7 m off; peaks within 16 frames of a gap have no vertex; under-segmentation remains. New unswept constants: spread=1, 2-lag flight window, contact-time clip.
- **Other L3 results:** A1 paper audit negative (T74-T76); A2 placement fixes negative (T77-T78); A3 entry cue neutral +0.0036 (T79), kick cue rejected (T80).
- **Draft:** submissions/draft_2026-11-15a.csv (412 possessions, mean 8.6 s; H 52488, A 36550, D 69819, X 40462; Barca 0.590). Reversal: `vtx=0`.

## ADR-021 — Graded frames before a half's kick-off are D (definition alignment, L4 review item 2)
- **Context:** organisers' FAQ.md "Which frames are actually graded?": the second half is graded from 110,808 but kicks off at 113,742; those frames are all `D`. Draft 2026-11-15a wrote H 1,455 / A 44 / D 1,435 there (docs/organiser-definitions-review.md).
- **Decision:** `kickoff.py` reads the Smart Tagging "Kick Off (Start)" tag window (input data) and takes the frame in it where the ball is nearest the centre spot; `pipeline.run_challenge` marks graded frames before it `D` and starts the decoder at the kick-off. No fitted constant; the constant 113,742 is not used.
- **Result:** detected kick-offs 11,187 and 113,741 (published 113,742: 1 frame). Benches unaffected (external path untouched). Public5 0.5300 23/17 unchanged (clip is in the 1st half; reported only). Draft submissions/draft_2026-11-15b.csv validated: H 51056, A 36487, D 71314, X 40462; 1,526 frames changed (27 outside the pre-kick-off window). pytest green.
- **QA (Fable, PASS WITH CHANGES) applied:** pre-kick-off D only when the tag window starts after the graded start (first half graded from its kick-off: untouched), tags outside the half ignored, earliest tag per half wins, `label_with_kickoffs` + tests. `period` also feeds the hidden-frame prior and attack votes: 5 isolated frames elsewhere change. Draft submissions/draft_2026-11-15c.csv validated (H 51056, A 36489, D 71312, X 40462); vs b only 11,185-11,186 back to A.

## ADR-022 — Align with the organisers' control definition: deflections are not possession changes (L4, user decision)
- **Context:** CHALLENGE.md 53-80: a change of possession needs controlled possession; a touch without control, a deflection that is not an interception, an aerial duel without subsequent control are not changes. Our external truth (Metrica, IDSSE) counts them (588/1198 true possessions < 4 s), so every bench rewarded splitting on deflections (why T58 lost). User rule 2026-11-15: a pass that hits an opponent and returns to the passer's team is no change; a kept interception starts at its first touch.
- **Decision (user chose 1 s, 2026-11-15):** default `ctl_s=1.0` (decode_halves post-pass, ctl_truth.relabel). New judging truth alongside the old one: `ctl` truth, external truth relabelled with T=2 s ("a few seconds", read from the text, not fitted; pass proxy rescues 0 spells). This is a **definition change, not a gate pass**: on the old truth ctl_s=1 fails ADR-016 (pooled -0.0066 [-0.0125, -0.0007]; real1 -0.0127, orig -0.0169). From now on candidates are judged on ctl T=2 truth with the ADR-016 rules, old truth reported.
- **Evidence (Fable QA PASS WITH CHANGES, L4-ctl-qa-fable):** pooled paired 2-min block bootstrap CI90: T=2 +0.0163 [+0.0106, +0.0219]; T=3 +0.0236 [+0.0169, +0.0301]; T=1 +0.0057 [-0.0000, +0.0111]. All 18 match x bench deltas positive on T=2 and T=3. Gain grows with T (zero on real2 at T=1): ctl_s=1 pays if the organisers' control needs >= ~2 s. 1.0 over 2.0 from external numbers only (2.0 loses up to -0.030 on T=1 truth, -0.065 on old). Also gains on the independent Metrica "2 s or a completed pass" truth.
- **Transfer (provided INPUT):** possessions 405 -> 371 (draft c -> d), D share unchanged; 381 frames changed. Public5 (reported only, never used to choose): 0.5300 23/17 -> 0.5684 21/17.
- **Draft:** submissions/draft_2026-11-15d.csv validated (H 51117, A 36428, D 71312, X 40462; Barca 0.584). Reversal: `--dec ctl_s=0` / DEC_SHAPE_L3_A4.

## ADR-023 — T95 stricter control pass as default (2026-11-17, user decision)
Context: L8 autonomous agent; after `ctl_s=1.0`, a team spell < 2 s following the other team's live spell with no uncontested own-team touch goes back (`ctl_hi=2.0, ctl_touch=1`). Organisers: a touch without control is not a change (CHALLENGE.md).
Evidence (ctl T=2 truth, eval/results/a8b_q3*.json, reproduced by cold QA a8qa_*.json): real2 s0-s3 +0.0070/+0.0162/+0.0157/+0.0068 (median +0.011, s0 CI90 [+0.0009, +0.0136]); real1 +0.0017; orig +0.0035. Old truth -0.0008..-0.0027 (neutral); T=1 truth -0.0006. Gain partly by construction of T=2 truth; ~0 if organisers' threshold < 2 s. Cold QA: MERGE (minor defects: reappear touch not counted; hidden spells always given back).
Transfer: 371 -> 352 possessions, 0.26% frames changed, Barca 0.584 -> 0.582.
Decision: default on (user chose "Turn T95 on"). T98 tagged set-piece taker stays off (user, no external gain).

## ADR-024 — T100 ball flight direction as default (2026-11-17)
Context: L8 round 2. On visible frames the team attacking in the ball's direction of travel gets evidence `w_fly=0.7`, ramped 4 -> 8 m/s (`decode.flight_dir`). New, ID-free team evidence; independent of touches.
Evidence (on top of T95; ctl T=2 / old truth, reproduced bit-identically by cold QA, eval/results/a8b_r2*.json, a8qa2_*.json): real2 s0 +0.0133/+0.0137, s1 +0.0204/+0.0179, real1 +0.0159/+0.0137, all CI90 > 0. Scanned on real2 s0 + real1 only; unused s2/s3/orig +0.012/+0.015/+0.015. Neighbours w 0.5-1.0, ramps 3-6..6-10 all +0.010..+0.019; cliff at w 1.5. Passes ADR-016/022 on both truths.
Cold QA: ADOPT. Minor: velocity window crosses half boundaries (~10 frames), 18 hidden frames get evidence (decode.py:613-618), raw ball used.
Transfer: 1.7% frames changed, 352 -> 342 possessions; back-pass flights here 37% vs 20-33% external (mild extrapolation).
T108 (w_ctrl=1.6): NOT adopted (cold QA: gain ~+0.003, not confirmed on data unused for choosing).

## ADR-025 — T112 + T113 line-crossing stoppage edges as default (2026-11-17)
Context: L8 round 3; Law 9 on the visible ball, nothing learned. T112 `ed_cross=5.0, ed_cross_strict=1`: a predicted stoppage's entry moves to the nearest ball line crossing within 5 s. T113 `ed_xdead=1, ed_xdead_min=2.0`: ball > 0.5 m beyond a line, no stoppage nearby, for >= 2 s -> D until seen inside.
Evidence (vs T95+T100; ctl T=2 [CI90] / old truth; reproduced bit-identically by cold QA a8qa3_*.json): real2 s0 +0.0170 [+0.0105, +0.0242] / +0.0115, s1 +0.0140, s2 +0.0118 / +0.0126, s3 +0.0136, real1 +0.0105 [+0.0052, +0.0161] / +0.0075, orig +0.0132. Selection: 5 s window best of 4 (first-tried 3 s also passes); 2 s minimum added after real1 -0.0027 (in-sample; inside the 0.003 allowance either way).
Cold QA: ADOPT. Minor: T112 runs after ctl passes (spells < 1 s on provided input 2 -> 4); a T113 stoppage ends when the ball touches the line (one throw-in split D/3 s A/D at frames 121148-121298). No off-pitch artefact becomes a stoppage (0 of 48 runs).
Transfer: 0.28% frames, 6 new 3.0-4.3 s stoppages (4 match tagged throw-ins), possessions 342 -> 346.
L9 hybrid ball fill (agent/a9): NOT adopted - position error halved (8.7-11.0 m vs 15.5-16.6 linear) but no spellF1 gain (real1 -0.005..-0.006 CI90 < 0); team call 0.69-0.77 vs shape model 0.84-0.91.

## ADR-026 — T125 learned ball-team evidence as default (2026-11-17)
Context: L8 round 5. Logistic model (src/possess/ballteam.py, constants MODEL; reproducible via scripts/a8b_ml.py fit) of which team holds the visible ball from 21 input-only features (proximity now and over 0.5-4 s, team at last impulse, flight, zone, shape). Fitted leave-one-match-out on external tune matches only. D-neutral emission on visible frames, `w_bt=0.5`.
Evidence (vs ADR-025 default, ctl T=2, cold QA reproduced bit-identically a8qa5_*.json): real2 s0-s3 +0.0193/+0.0020/+0.0121/+0.0144 (median +0.0133; s1 CI90 crosses 0); seed-pooled +0.0120 [+0.0044, +0.0193]; seeds unused for choosing s1-s3 +0.0095 [+0.0016, +0.0172]; real1 +0.0089 [+0.0019, +0.0158]; orig +0.0034. Old truth positive (unused seeds +0.0061 [-0.0002, +0.0121]). Gate read per ADR-023 precedent (per-seed deltas, median over seeds). Expected real gain ~+0.009.
QA notes: some features look 0.2-0.6 s ahead (centred velocity); training matches' shape feature from a shape model that saw the test match (second order). No feature scale shift on the provided input; match more one-sided (59.6% home) than any tune match.
Transfer: 1.67% frames, possessions 346 -> 324, tags side with the new label 77:23 on changed one-team frames.
Gate per clause (ADR-016/022 rule, ctl T=2; appended 2026-11-17, L17-docs). Reading used: per-seed deltas, median over real2 seeds (ADR-023 precedent). Seed 1's CI90 crosses zero; the seed-pooled interval is [+0.0044, +0.0193].
| Clause | Value | Result |
|---|---|---|
| real2 drops <= 0.003 | s0-s3 +0.0193/+0.0020/+0.0121/+0.0144, median +0.0133 | pass |
| real1 drops <= 0.003 | +0.0089 [+0.0019, +0.0158] | pass |
| at least one gains | real2 pooled +0.0120 [+0.0044, +0.0193]; real1 CI90 > 0 | pass |
| orig drop <= 0.005 | +0.0034 | pass |
| paired bootstrap + second real2 seed | 4 seeds; unused seeds s1-s3 +0.0095 [+0.0016, +0.0172] | pass |
| old truth (reported) | unused seeds +0.0061 [-0.0002, +0.0121] | no loss |
| held-out (Metrica 3, IDSSE 2) | not measured | - |
Cold-QA caveats left open: features 0.2-0.6 s ahead (centred velocity); training matches' shape feature from a shape model that saw the test match; provided match more one-sided (59.6% home) than any tune match; held-out not measured.

## ADR-027 — Public-clip differences used for diagnosis only (2026-11-17, user request)
User (viewer screenshot of pred vs public truth): "spawn opus swarm to understand the difference and make a proper implementation that covers it without overfitting". Amendment to the "public 5 min: reported only" rule: the orchestrator computes a catalogue of error TYPES on the public clip (scratchpad diffcat_17d.txt, not committed); agents may read it and inspect the INPUT at those frames to form hypotheses. Unchanged: agents never open the truth file; every mechanism and constant is chosen and gated on external data only (ADR-016/022); a hypothesis must first be shown to exist as an error class on external data; the public score is reported after adoption and never used to select among variants. To be disclosed in TECHNICAL.md.
Catalogue (draft 17d, 994 of 7,500 frames differ): visible dead ball labelled as a team 18.7 s (one 10.5 s visible stoppage); opponent spell starts early / lasts long 6.4 s + extra 3.4 s spell; one 6.6 s hidden A/D block missed; short spells < 1.5 s missed or extra.

## ADR-028 — Visible dead ball: w_bd=1.0 + ed_still=1.0 default (2026-11-17, user decision; reverses the morning "keep off")
Context: L11-A (agent/a10). Public-clip error class (ADR-027): ball visible and dead, labelled as a team. Bench realism from INPUT only: ball visible in the first 3 s of 34% of tagged set-piece clips (goal kicks 80%, throw-ins 39%, free kicks 20%, corners 1%); true-D frames with a visible ball: real2 16-25%, real1/orig 53-59%.
Mechanism: w_bd = learned "visible ball is dead" logistic (balldead.MODEL, 21 team-blind features, external tune matches only) added to the visible-frame stoppage log-odds; ed_still = ball at rest >= 3 s with no player within 1 m -> D.
Evidence (ctl T=2, LOMO models; cold QA reproduced, a10qa_bd_still.json): real2 s0/s1/s2 +0.0033/+0.0006/+0.0042 (s0 CI [-0.004, +0.011]); real1 +0.0187 [+0.012, +0.026]; orig +0.0121 [+0.007, +0.018]; unused seeds real1 s1 +0.0219 [+0.015, +0.029], orig s1 +0.0076 [+0.003, +0.013]. Old truth positive. Cold QA: ADOPT, passes ADR-016 as written; no leakage; no fragmentation (43 of 44 changed runs extend existing stoppages). Caveats: still_dead gains on orig only and adds 166-291 false-D frames elsewhere; w_bd idsse1 -0.0042 on real2 s0.

## ADR-029 — Tag-gated set-piece dead ball default: sp=1 sp_hid=1 sp_take_s=0 (2026-11-17, user decision; RECORDED GATE EXCEPTION)
Context: user request from the viewer (goal kick: truth dead ~10 s, we labelled Barcelona). L11-C (agent/a12), src/possess/setpiece.py: for each Smart Tagging Goal Kick / Corner Kick / Throw-in clip, find the physical restart from 3 s before to 15 s after the clip start (ball at rest on the spot the Laws require then kicked, or reappearing there and moving); frames from our last dead evidence (within 15 s) to the kick become D; no restart found -> no change. Free kicks excluded (no fixed spot). The tag is a gate, not a clock.
Evidence (ctl T=2 [CI90], simulated tags on external; cold QA bit-identical, a12qa_*.json): real2 s0 +0.0030 [+.0006, +.0059], s1 +0.0016, s2 -0.0005, s3 +0.0033 [+.0011, +.0058]; real1 +0.0100 [+.0057, +.0150]; orig +0.0021 [+.0005, +.0041].
Gate: STRICT FAIL (real2 median ~+0.002 < +0.005; s1, s2 CI include 0); no bench drops > 0.003; real1 and orig pass. Exception granted by the user because real2 hides dead balls while the ball is visible at 80% of the provided goal kicks. Cold QA: ADOPT as recorded exception; the 12.6 s change near row 37276 is a ball at rest on the goal-area line (not live play). Open defects: setpiece.py:40 goal-area half-width 14.66 (should be 9.16); :68 throw-in spot without outer bound; :99 later "rest" beats earlier "reappear"; a 2-frame H spell left at 37060-61.
Combined result (ADR-028 + 029): submissions/draft_2026-11-17e.csv, validated; 1,839 frames differ from 17d (0.92%), mostly team -> D (1,572). Public 5 min (organisers' run_evaluation.py, REPORTED AFTER ADOPTION, not used to choose): spellF1 0.6588 (17d 0.6353), macroF1 0.8886 (0.8417), 17 predicted vs 17 true possessions.
Not adopted today: L11-B switch dating (external truth dates turnovers at first touch, organisers at control), L12 hard tag rules, L13 fuzzy tags, L9 hybrid fill; L15 camera pan pending.

## ADR-030 — ctl_s 1.0 -> 0.6 as a hedge across label readings (2026-11-17, user decision)
Context: other teams' public write-ups report that the organisers' labels keep very short possessions (0.4-1.3 s); the organisers' text says a touch without control is not a change. We do not tune on the public clip. Principle (external data only): prefer the setting with the lowest worst-case regret across both readings of the labels. L30 (agent/a20, baf2a85), five benches (real2 s0-2, real1, orig), LOMO models, full pipeline.
Evidence (mean spellF1 delta vs ctl_s=1.0; * = CI90 excludes 0): old truth +0.0050*, ctl T=1 +0.0010, T=2 -0.0015, T=3 -0.0027*. Worst-case regret 0.0102 -> 0.0052. Borderline: the old-truth gain comes from real1 (+0.0115) and orig (+0.0101), ~0 on real2; averaged by bench type the T=3 loss is -0.0037. ctl_s=0.4 fails (-0.0046 at T=3); switching T95 off loses heavily under the controlled readings. ctl_hi=2.0 / ctl_touch=1 unchanged.
Not a gate pass under ADR-016/022 (judged on ctl T=2 it is -0.0015): adopted by the user as insurance against the unknown labelling of sub-second possessions.
Result: submissions/draft_2026-11-17f.csv, validated; 38 frames differ from 17e (20 A->H, 18 H->A), possessions 326 -> 330, exactly as the agent's transfer run predicted.
Also tested tonight and rejected (external gate): L28 posterior switch placement (no movement at T<=1, loss at T=2), L29 measured transition costs (-0.001..-0.010) and emission temperature, L26 switch-dating variants under five dating conventions (stake <= 0.004), L22 first-spell-after-stoppage rule (coin flip), L15 camera pan.
