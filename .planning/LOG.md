# LOG — append-only

Format: date · phase · action · envelope/commit · outcome

2026-11-11 · setup · harness scaffolded from Padel Lens playbook (CLAUDE.md, settings deny on data/challenge, .planning/) · — · ok
2026-11-11 · D1 · official repo cloned to third_party/mtah (sparse: docs + code only, no gt/baseline CSVs) · — · ok
2026-11-11 · D1 · MatchLens Eventer replay on Metrica G1/G2 scored with official possession_metrics (scripts/score_matchlens_replay.py) · eval/results/matchlens_replay_metrica_clean.json · spellF1 0.463 / 0.417, +45–64% possessions
2026-11-11 · D1 · goals artifact published (docs/challenge-goals.html) · https://claude.ai/artifact/XNfrviFeAAbb5tkbWjABN9 · ok
2026-11-11 · D1 · MatchLens EPTS adapter (src/possess/matchlens_adapter.py) tested on public Metrica G3: 143,761 frames in ~80 s, half-time side swap detected · — · ok
2026-11-11 · D1 · blind runner scripts/blind_run_public5.py (whitelisted stdout) · — · blocked: Drive folder not public, gdown cannot list it; user to download manually into data/challenge/
2026-11-11 · D1 · user-supplied zips unpacked to data/challenge/tactical/ (atd xml+txt, smart_tagging, homography; no video) · — · ok
2026-11-11 · D1 · blind run MatchLens v0 on tactical ATD, graded halves fed 79467/79390; validator OK after fetching games.py; public window spellF1 0.2383, macroF1 0.5763, 30 vs 17 possessions · eval/results/matchlens_v0_public5.json · ok (below smooth50)
2026-11-11 · D1 · research note: ball gap-filling methods · docs/research/ball-interpolation.md · ok
2026-11-11 · D1 · v1: MatchLens two-sided linear ball fill ported into adapter (--interpolate-ball); 64,091 frames filled; public window spellF1 0.2383 (=v0), macroF1 0.5898 (+0.013), 30 possessions (=v0). Cause: deadball.py treats conf=None ghost as missing -> dead after 0.5 s · eval/results/matchlens_v1_linear_public5.json · no change
2026-11-11 · D1 · ball_fill package (one file per method: base, linear, cubic) + tests/test_ball_fill.py (7 pass); adapter uses --ball-fill; linear reproduces v1 exactly · — · ok
2026-11-11 · D1 · cubic fill blind run: spellF1 0.2286, macroF1 0.5895, 32 possessions (noise vs linear) · eval/results/matchlens_cubic_public5.json · no gain
2026-11-11 · D1 · kalman.py (CV Kalman + RTS, untuned) + 2 tests (9 pass); blind run spellF1 0.2383, macroF1 0.5877, 30 possessions = linear · eval/results/matchlens_kalman_public5.json · no gain
2026-11-11 · D1 · physics.py (two-sided drag flight, <=2.5 s, touch/out signals) + 2 tests (11 pass; tests caught a wrong backward-flight formula, fixed); blind run spellF1 0.2286, 32 possessions · eval/results/matchlens_physics_public5.json · no gain
2026-11-11 · D1 · KEY: challenge input gap stats — 147 interior gaps; 16 under 1 s hold 73 frames, 131 of 5 s+ hold 64,018 (99.9%), none 1-5 s. Interpolation cannot matter; need ball-free signals; degradation must hide ball in long blocks · same file · finding
2026-11-11 · D1 · goals artifact v3: interpolation tries table · https://claude.ai/artifact/XNfrviFeAAbb5tkbWjABN9 · ok
2026-11-11 · D1 · ADR-001: organisers' 2nd clarification allows agents on raw data; RULES.md updated verbatim; settings read-deny lifted by user approval (write-deny on tactical/ kept) · — · ok
2026-11-11 · D1 · player-anchored fill: spellF1 0.2383, 30 poss; 30/147 gaps change team inside · — · no gain
2026-11-11 · D1 · poly2/3/4/5/7 fills: spellF1 0.2286/0.2383/0.2383/0.2286/0.2286 (30 or 32 possessions only) · — · no gain
2026-11-11 · D1 · Haiku swarm x7 on raw slices (w1-w2 with public GT) · docs/research/swarm-findings.md · missing != dead; possession survives ~75% of gaps; dead = far+line+slowdown
2026-11-11 · D1 · masking test (200 hidden seen stretches per gap length): G3 cubic best 0.5-10 s; at 20 s all ~19-21 m off; challenge check favours poly5/7 on short gaps (smoothed track) · eval/results/ball_fill_masked_{g3,challenge}.json · ok
2026-11-11 · D1 · warped.py (log c=3/9/30, root 2/3) + 6 tests (24 pass); masking test: all worse than linear on G3 and challenge, worse with harder deceleration; cubic still best 0.5-10 s · eval/results/ball_fill_masked_{g3,challenge}.json · rejected
2026-11-11 · loop 0 · external bench scripts/bench.py (Metrica G1/G2 tune, G3 held-out; dead balls visible; long-block hiding); MatchLens baseline tune spellF1 0.3157 · eval/results/bench_matchlens_tune_s0.json · baseline
2026-11-11 · loop 1 · ghost dead-ball rule off (ADR-002) + tests/test_matchlens_ghost.py (3 pass); tune spellF1 0.3256, counts 356/350 vs 357/330, mean 10.2 s; CIs overlap, held-out not looked · eval/results/bench_matchlens_noghost_tune_s0.json · kept as base
2026-11-11 · loop 2 · oracle ablations (scripts/oracle.py): visible-ball team is the biggest lever (0.326 -> 0.608 ceiling); MatchLens abstains 18-20% on visible live frames; team_control.py (team-level PZ/DZ + retention) + 6 tests; tune grid best r_pz=r_dz=1.0 k=5 -> 0.3562 · eval/results/bench_team_control_v1_tune_s0.json, tune_team_control.json · kept
2026-11-11 · loop 3 · worktree isolation refused (D:/ITe vs D:/ITE path case); IDSSE builder runs in the main checkout with owned files only · .planning/envelopes/L3-idsse-01.in.md · running
2026-11-11 · loop 3 · deadball2.py (far + line + team slowdown + MatchLens D, hysteresis) + 5 tests; tune grid best 0.3747 (from 0.3562), D recall 0.63 prec 0.78; ball speed excluded as a bench artifact · eval/results/tune_deadball.json · kept provisional
2026-11-11 · loop 3 · IDSSE bench (1ae0eb7) cold QA: keep, no bugs; low: caches lack parser VERSION, test gaps, IDSSE ball visible on 100% of D frames (challenge ~72%) -> BACKLOG · — · ok
2026-11-12 · loop · stopped by the user after iteration 3; iteration 4 grid not completed (low RAM) · — · stopped
2026-11-12 · research · restart-team check on bench truth: other team restarts after 82%/78%/90% of stoppages (metrica1/2, idsse1); pipeline keeps the old team · docs/research/literature/C_restarts/survey.md · finding
2026-11-12 · repo · self-contained (ADR-004): MatchLens dead-ball port in src/possess/stoppage_ml.py (100% agreement with Eventer on Metrica G1), uv.lock, vendored scorer; clean clone 46 tests pass; full-match CLI run pending (2.3 GB free RAM, needs ~4) · 615f73b · ok
2026-11-13 · loop2-4 · streaming EPTS loader (identical to kloppy on 5000 frames), scripts/evaluate.py (tune + public5 sanity); P1+P2 kept: tune 0.3586 -> 0.3722, public5 0.432 -> 0.454 (reported only) · eval/results/eval_*.json · kept
2026-11-13 · loop2-5 · P3 Viterbi decoder (src/possess/decode.py) untuned tune 0.3541 (< 0.3722 default); decoder grid killed by Claude Code for low system RAM, not restarted · — · paused
2026-11-13 · loop2-5 · P3 decoder grid (24 configs, foreground): best 0.3597 < 0.3722, not kept; Fable designer + Fable QA spawned (user: use Fable for hard tasks and validation) · eval/results/tune_decoder.json · not kept
2026-11-13 · loop2-4qa · Fable QA applied: half-time reset (free), max_gap option off (costs 0.002-0.005 on tune); CLAUDE.md public5 line updated; tune 0.3724 · eval/results/eval_default_l4qa.json · kept
2026-11-13 · loop2-6 · decoder v2 adopted as default after Fable QA fixes + held-out look 1 (0.3556 -> 0.3625); tune 0.3976, public5 0.444 · eval/results/eval_*_HELDOUT.json · kept (ADR-005)
2026-11-14 · loop2-7 · P5 team-shape (Fable designer) reproduced tune 0.4868 LOMO vs 0.3976; committed opt-in (293159d); Fable skeptical QA on leakage/bench artefacts running · — · pending
2026-11-14 · loop2-7 · Fable QA of P5: adopt after fixes; real gain (team-only 0.4335, stoppage-only 0.4303, both 0.4868; +0.087 [0.069,0.105] over 55 chunks); HIGH: stoppage speed features break under 5-10 cm jitter (0.4246/0.4091); sent back to designer with fixes + noise gate + pipeline wiring · — · in progress
2026-11-14 · loop2-7 · P5 v2 reproduced tune 0.4932, noise gate passes; public5 0.433 (flagged, noise); Fable re-QA running; held-out look 2 after it · 3fceb32 · pending
2026-11-14 · loop2-7 · P5 adopted (ADR-006): Fable re-QA adopt; held-out look 2 0.3625 -> 0.4789; tune 0.4932; public5 0.433 · eval/results/eval_p5_HELDOUT.json · kept
2026-11-14 · submission · draft_2026-11-14.csv via uv run possess (44 s): validator OK 199,319 rows; 434 possessions mean 10.0 s; Barca 57% of live; CLI public5 0.4333 = evaluate.py (QA finding 1 closed) · submissions/ (gitignored) · ok
2026-11-14 · loop2-8 · touch emission adopted (ADR-007): Fable QA adopt; held-out look 3 0.4789 -> 0.5584; tune 0.5657; public5 0.4556 · eval/results/eval_touch_HELDOUT.json · kept
2026-11-14 · submission · draft_2026-11-14b.csv: validator OK; 446 possessions mean 9.7 s; Barca 57.4% · submissions/ · ok
2026-11-14 · loop2-9 · Fable QA ball cleaning: adopt after fixes (removes real kicks: idsse1 dip real); sent back to designer · — · in progress
2026-11-14 · loop2-9 · ball cleaner adopted (ADR-008) after 2 Fable QA rounds + re-QA fixes; tune 0.5679 · eval/results/eval_ballclean_final.json · kept
2026-11-14 · loop2-10 · hidden-ball team context adopted (ADR-009); held-out look 4 0.5584 -> 0.5762; tune 0.5845 · eval/results/eval_hidden_HELDOUT.json · kept
2026-11-14 · submission · draft_2026-11-14d.csv: validator OK; 445 possessions mean 9.9 s; Barca 58% · submissions/ · ok
2026-11-14 · loop2-11 · still-ball stoppage + kick restart (Fable designer) reproduced tune 0.6125; public5 0.4432 20/17 (dip, one possession, flagged); Fable QA with Smart-Tagging transfer check on the real match running · 6594763 · pending
2026-11-14 · loop2-11 · still/kick rule REJECTED (ADR-010): bench artefact (still ball 4.5% on real match vs 21-42% bench); reverted to tune 0.5845; protocol now requires input transfer check for ball-in-stoppage features · eval/results/eval_revert_l211.json · rejected
2026-11-14 · loop2-12 · Fable bench audit: all adopted steps still help on realistic bench real1 (default 0.5276 vs 0.5845); touch at risk but positive; biggest exposure = stoppage detector on a moving dead ball; decisions now on real1 · eval/results/audit_*.json · audit
2026-11-14 · loop2-13 · ADR-011 adopted: real1 held-out 0.5227 -> 0.5557, orig held-out -0.0046; evaluate.py --bench now allowed on held-out · eval/results/eval_l213*_HELDOUT.json · kept
2026-11-14 · submission · draft_2026-11-14e.csv: validator OK; 410 possessions mean 10.1 s; D 27.8%; Barca 58.7% of live · submissions/ · ok
2026-11-14 · loop2-14 · ADR-012 cap 1.0 adopted: held-out look 6 real1 0.5557 -> 0.5619, orig 0.5716 -> 0.5787 · eval/results/eval_cap1*_HELDOUT.json · kept
2026-11-14 · submission · draft_2026-11-14f.csv: validator OK; 397 possessions mean 10.4 s; Barca 58.6% · submissions/ · ok
2026-11-14 · repro · clean clone + uv sync: 109 tests pass; uv run possess regenerates draft f byte-identical (SUBMISSION.md reproduction rule) · — · ok
2026-11-14 L2-15 designer done (duel=2 touch rule): reproduced real1 0.5935 / orig 0.6205 / public5 0.4588; transfer near-inert on provided match; Fable QA launched
2026-11-14 L2-15 ADOPTED as default (ADR-013): Fable QA PASS; real1 0.5935 / orig 0.6205 / public5 0.4588; draft g validated (1 frame from f); docs updated
2026-11-14 L2-16 (fast-player speed in stoppage model) reproduced: real1 0.6061, orig 0.6283, public5 0.5294 (reported only); Fable QA launched
2026-11-14 L2-16 ADOPTED as default (ADR-014): Fable QA PASS; held-out look 7 (last): real1 0.5788 / orig 0.5968; draft h validated (397 possessions); docs updated
2026-11-14 L2-17 segment clean-up NEGATIVE (best combo +0.0009, ~70 candidates); code kept opt-in/off; default unchanged real1 0.6061; L2-18 stop-edges designer launched
2026-11-14 L2-19 ball-clean-before-features spec'd (envelope ready): DEPRIORITISED — real1 already has more ball faults than the provided input (cleaner removes 0.0% there), expected ~0 on provided match
2026-11-14 L2-18 ADOPTED as default (ADR-015, 6651aa9): hidden stoppage exits snapped to fast-player speed step; real1 0.6061 -> 0.6081, orig 0.6283 -> 0.6305, public5 0.5294 unchanged; Fable QA PASS; draft i validated (397 possessions); exit+entry variants rejected (D-precision gate)
2026-11-14 L2-20 ADOPTED (ADR-016 real2 bench + co-deciding rule; ADR-017 smooth_touch, 99e7547): real2 0.4303 -> 0.4833, real1 0.6081 / orig 0.6305 unchanged, public5 0.5294 -> 0.5059 (reported only); draft j validated (420 possessions); smooth_touch PENDING USER CONFIRMATION; expected real match ~0.48
2026-11-14 L2-21 ADOPTED (ADR-018 per-match unsupervised EM dead-ball prior on hidden frames, d653755): real2 0.4833 -> 0.5173 (seed 2 0.5152, bootstrap +0.0339), real1 0.6081 / orig 0.6305 unchanged, public5 0.5059 unchanged (reported only); Fable QA PASS with changes; draft k validated (416 possessions); disclosure + organiser question cover sd_prior; expected real match ~0.50-0.52
2026-11-14 L2-22 ADOPTED (ADR-019, e4ee541): reappear touch hold; real2 0.5326, real1 0.6068, orig 0.6283, public5 0.5294; draft l. LOOP STOPPED at this checkpoint on user request.
2026-11-15 L3 started: 4 independent Opus solvers in worktrees bp-wt/a1..a4 (paper audit, switch timing, dead ball, short possessions), heavy runs serialised by scripts/ramlock.py; Fable orchestrator validates + merges
2026-11-15 L3-A1 paper audit: NEGATIVE (pass-by filter, duel-zero emission, 0.5 m zone all lose on real2 and real1); paper's team-level rules already ported; T74-T76
2026-11-15 L3-A2 switch timing: NEGATIVE for placement fixes (T77, T78); diagnosis: late switches on smoothed ball come from missed touches (oracle +0.065)
2026-11-15 L3-A3 dead ball: nothing adopted (T79 entry cue neutral +0.0036, T80 kick cue rejected); diagnosis: hidden D frames right (recall 0.93-0.96), loss is stoppage edge timing (oracles +0.02..+0.056)
2026-11-15 L3-A4 ADOPTED (ADR-020, merge 91e7bc8): vertex touch on heavily low-passed tracks; real2 0.5618, real1 0.6068, orig 0.6283, public5 0.5300 (23/17); draft 2026-11-15a
2026-11-15 ADR-021: graded pre-kick-off frames -> D via Smart Tagging kick-off (kickoff.py); L4 definitions review done (docs/organiser-definitions-review.md); draft 2026-11-15b
2026-11-15 ADR-021 QA changes applied; draft 2026-11-15c; L4 control-definition (a5) reported: decision on truth T / gates pending user
2026-11-15 ADR-022 ADOPTED (user decision): ctl_s=1.0 default, judged on organisers' control definition (ctl T=2 truth); draft 2026-11-15d (371 possessions), public5 0.5684 21/17 (reported only)
2026-11-15 L5 hidden team (agent/a7 57660ee): nothing adopted; w_rev=2.0 opt-in neutral (fails real2 seed median), gap_cp rejected; real2 hidden-team ceiling +0.029 on ctl truth; TRIED T88-T90
2026-11-16 L6 smart-tagging Fable agent launched (bp-wt/a8)
2026-11-16 L6 smart-tagging: built (a8 ad8ea26), unscored, blocked by low RAM; naive review committed 0f600cb
2026-11-17 L7 docs final e8099b7; L8 autonomous agent started (bp-wt/a8)
2026-11-17 ADR-023 T95 default on (user); T98 off (user); merge cb48210
2026-11-17 ADR-024 T100 w_fly default (QA ADOPT); T108 rejected; merge f3f448a
2026-11-17 ADR-025 T112+T113 default (QA ADOPT); L9 hybrid fill rejected
2026-11-17 ADR-026 T125 w_bt=0.5 default (QA ADOPT)
2026-11-17 user: stop. Autonomous loop ended after round 7; final draft 17d (user); T137/T120 off (user); T141 not adopted
2026-11-17 ADR-027 public-clip diagnosis (user); swarm a10/a11 + viewer tags + docs
2026-11-17 ADR-028 (w_bd+ed_still) + ADR-029 (set-piece rule, gate exception) default by user decision; draft 17e, public5 0.6588 (reported)
2026-11-17 L24 generality audit (cold): FINAL > BASE in 7/7 runs, 4 truths, both sources, 4 shifted conditions (+0.044..+0.067 ctl2); w_bt/T95/set-piece condition-dependent; keep FINAL; docs/generality-method-audit.md
2026-11-17 ADR-030 ctl_s=0.6 hedge (user); draft 17f; L15/L22/L26/L28/L29 rejected
2026-11-17 final-hour tests rejected on the external gate: L33 touch candidates, L34 min_dwell + flight switch ban, L35 second-touch restart (refuted), L36 HSMM stoppage durations, L37 two-sided judge. Final = draft 17f
