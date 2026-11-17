# TRIED — registry of hypotheses already tried

**Rule:** before starting a hypothesis, search this file (id, keyword, area). A repeat needs a stated new angle (new evidence, new data, a changed constraint) written in the envelope's `context_refs`. Add a row for every new idea when its run is scored, including rejected ones. Numbers are spellF1 mean over the tune matches unless noted; `real1` = realistic bench (ADR-011/L2-12), `orig` = original bench, `public5` = reported only, never used to choose. Blank cell = not recorded in the sources. Sources: LOOP.md, DECISIONS.md, LOG.md, BACKLOG.md, envelopes, research plan, eval/results names.

Last updated 2026-11-14 (after L2-21 adopted, ADR-018; L2-20 adopted, ADR-016/017; L2-18 adopted, ADR-015; L2-17 rejected).

## Registry

| id | idea | area | result (real1 / orig / public5) | verdict | why / pointer |
|---|---|---|---|---|---|
| T01 | MatchLens v0 as-is on public 5 min (baseline) | other | public5 0.2383 (30 vs 17 poss.), macroF1 0.5763 | NEUTRAL | baseline; `matchlens_v0_public5.json` |
| T02 | MatchLens Eventer replay on Metrica G1/G2 | other | orig G1 0.463 / G2 0.417 (+45-64% possessions) | NEUTRAL | baseline; `matchlens_replay_metrica_clean.json` |
| T03 | Linear two-sided ball fill (`--interpolate-ball`) | ball-fill | public5 0.2383 (= v0), macroF1 0.5898 | REJECTED | no change; ghost conf=None read as missing -> dead after 0.5 s; `matchlens_v1_linear_public5.json` |
| T04 | Cubic ball fill | ball-fill | public5 0.2286, 32 poss. | REJECTED | noise vs linear; best only for 0.5-10 s gaps in masking test; `matchlens_cubic_public5.json` |
| T05 | Constant-velocity Kalman + RTS fill (untuned) | ball-fill | public5 0.2383 (= linear) | REJECTED | no gain; `matchlens_kalman_public5.json` |
| T06 | Physics fill (two-sided drag flight, <= 2.5 s) | ball-fill | public5 0.2286, 32 poss. | REJECTED | no gain; `matchlens_physics_public5.json` |
| T07 | Player-anchored ball fill | ball-fill | public5 0.2383, 30 poss. | REJECTED | 30 of 147 gaps change team inside; no gain |
| T08 | Polynomial fills order 2/3/4/5/7 | ball-fill | public5 0.2286 / 0.2383 / 0.2383 / 0.2286 / 0.2286 | REJECTED | 30-32 possessions only; no gain |
| T09 | Warped time fills: log c=3/9/30, root 2/3 | ball-fill | masked test: all worse than linear on G3 and challenge | REJECTED | `ball_fill_masked_{g3,challenge}.json` |
| T10 | Any interpolation as a lever | ball-fill | | REJECTED | 131 of 147 interior gaps are >= 5 s (99.9% of gap frames), none 1-5 s: interpolation cannot matter; need ball-free signals (LOG 2026-11-11) |
| T11 | "Ball unseen 0.5 s => dead" (MatchLens ghost rule) | dead-ball | orig tune 0.3157 -> 0.3256 when OFF | ADOPTED (rule removed) | ADR-002; ghost off; counts 356/350 vs 357/330 |
| T12 | Ghost rule with longer threshold | dead-ball | | REJECTED | gaps are 5-30 s, any threshold below splits possessions (ADR-002) |
| T13 | Team-level possession/duel zones + k-frame retention (`team_control.py`, r_pz=r_dz=1.0, k=5) | team-control | orig tune 0.3256 -> 0.3562 | ADOPTED | ADR-003; grid `tune_team_control.json` |
| T14 | Zone radius x k grid (r 1.0/1.5/2.0, k 2/3/5/8) | team-control | k3 0.3748 ~ k5 0.3722 (noise) | NEUTRAL | kept r1.0 k5; `eval_grid_r*_k*.json` |
| T15 | `deadball2.py` ID-free dead evidence (far + line + slowdown, hysteresis; r_far 9, d_line 2, v_slow 1.5) | dead-ball | orig tune 0.3562 -> 0.3747 | NEUTRAL | provisional in loop 3; superseded by decoder/shape stoppage evidence; ball speed deliberately excluded (bench artefact); `tune_deadball.json` |
| T16 | P1 restart prior: other team restarts after D | team-control | tune 0.3559 alone (hurts), with P2 0.3722 | ADOPTED (only with P2) | L2-4; later check: restart team already right 340/421 (L2-11), restart-from-geometry oracle +0.002 (L2-14) |
| T17 | P2 back-date committed switch to first control frame | team-control | tune 0.3680 alone; P1+P2 0.3722; public5 0.432 -> 0.454 | ADOPTED | L2-4 |
| T18 | `max_gap` (challenge across ball gaps) 50/125/250 frames | team-control | 0.3669 / 0.3694 / 0.3707 vs 0.3724 | REJECTED | off by default (L2-4qa); turnovers usually happen inside the gap |
| T19 | Half-time state reset | team-control | 0.3724 (free) | ADOPTED | L2-4qa |
| T20 | P3 Viterbi decoder v1 (hard zone emission, switch at gap edge, 24 configs) | decoder | tune best 0.3597 < 0.3722 | REJECTED (superseded) | flat zone emission; switch placement at block edge; `tune_decoder.json` |
| T21 | P3 decoder v2: soft log-distance emission (cap 1.5, d0 0.25), hard stoppages, w_rule 0.3, switch at gap middle, c_switch 4, c_same 1 | decoder | tune 0.3976; held-out 0.3556 -> 0.3625; public5 0.444 | ADOPTED | ADR-005; L2-6; `tune_decoder_v2.json` |
| T22 | Decoder v2 without frame-rule emission (simpler) | decoder | tune 0.3881 | REJECTED | 0.3976 with w_rule 0.3 (neighbours 0.373-0.391, may be noise); needed for home/away symmetry |
| T23 | Soft vs hard stoppages in decoder | decoder | soft 0.3910 vs hard 0.3881 | NEUTRAL | hard kept |
| T24 | Decoder `cap` 1.0 / 2.0 (at the time) | decoder | 0.373 / 0.374 vs 0.3976 | REJECTED | at that time; cap 1.0 later adopted with new evidence (T38) |
| T25 | P5 team-shape models on hidden-ball frames + stoppage model (numpy logistic, 15 features each, LOMO) | shape | tune 0.3976 -> 0.4932; held-out 0.3625 -> 0.4789; public5 0.433 | ADOPTED | ADR-006; L2-7; team-only 0.4333, stoppage-only 0.4310; `tune_shape*.json` |
| T26 | Shape features: frame-to-frame speed | shape | breaks under 5-10 cm jitter (0.4246 / 0.4091) | REJECTED | replaced by 0.8 s lag speed (L2-7 QA) |
| T27 | Shape weight `w_sdead` beyond 0.3 | shape | | NEUTRAL | 0.3 is grid edge; larger not tried (L2-7 risks) |
| T28 | Camera-window robustness (players outside +-25 / +-15 m dropped) | shape | 0.4820 / 0.4353 | NEUTRAL | robustness test only |
| T29 | Touch hold (`decode.touch_hold`, port of impulse.py `impulse_verdict`), w_touch 0.7, max 4 s | touch | tune 0.4932 -> 0.5657; held-out 0.4789 -> 0.5584; public5 0.4556 | ADOPTED | ADR-007; L2-8; real1 gain +0.047 (orig +0.073) |
| T30 | Touch variant r_touch 0.75 + c_switch 6 | touch | 0.5773 vs 0.5657 | NEUTRAL | not chosen; IDSSE prefers 1.0, noise (L2-8) |
| T31 | Ball cleaning before touch_hold (`ballclean.py`: teleports, short freezes) | touch | tune 0.5657 -> 0.5680 (real1 +0.002, inert on real ball) | ADOPTED | ADR-008; L2-9; two Fable QA rounds |
| T32 | Remove long freezes (> 10 frames / > 2 s) in cleaning | touch | idsse1 0.530 -> 0.512 | REJECTED | freezes are stoppages whose exit jump is the restart touch |
| T33 | Ball-clean off-pitch rule | touch | | REJECTED | rule exists but off; removes the real ball crossing the line (metrica1 46 frames) |
| T34 | Team-context features (+-2 s shape change) for hidden-ball turnovers, TEAM_CTX=1 | shape | tune 0.5679 -> 0.5845; held-out 0.5584 -> 0.5762; real1 +0.016 | ADOPTED | ADR-009; L2-10; hidden-turnover recall 0.08 -> 0.18; 15 feature sets tried |
| T35 | More shape weight on hidden frames (x2, x3); `w_shape` 0.075; nearest-opponent features | shape | 0.5605 / 0.5523; 0.5901; no gain | REJECTED | evidence not switch cost is the limit (L2-10) |
| T36 | Still-ball stoppage (still_v 0.3) + kick restart (v_kick 3) in deadball2 | dead-ball | orig tune 0.5845 -> 0.6125; public5 0.4432 | REJECTED | ADR-010: bench artefact (still ball 4.5% on real input vs 21-42% bench); `eval_still.json`, `eval_revert_l211.json`; also not adopted: w_dead=2 0.6196, out_m=0 0.6180, still_v=1.0 0.6263, still_far 2 m 0.6062 |
| T37 | Player-only stoppage evidence x2.5 on visible frames (sdead_vis 2.5), w_live 0 | decoder | real1 0.5276 -> 0.5731; orig 0.5845 -> 0.6002; held-out real1 0.5227 -> 0.5557, orig 0.5762 -> 0.5716; public5 0.4588 | ADOPTED | ADR-011; L2-13; expected real-match effect ~0 (ball mostly lost in real stoppages); 18 configs |
| T38 | Distance emission cap 1.5 -> 1.0 (= MatchLens r_pz) | decoder | real1 0.5731 -> 0.5834; orig 0.6002 -> 0.6110; held-out real1 0.5557 -> 0.5619, orig 0.5716 -> 0.5787 | ADOPTED | ADR-012; L2-14; borderline after ~35 candidates; IDSSE null |
| T39 | w_rule = 0 (drop frame-rule emission) | decoder | real1 0.5950, orig 0.6192 | REJECTED | removes 24% of possessions on the provided match (bench 1-7%); transfer check (ADR-012) |
| T40 | Lead/lag stoppage model (+-2, +-4 s) | shape | AUC +0.003, real1 0.5725 | REJECTED | removed (L2-14) |
| T41 | Duel=2: contested touch keeps previous uncontested toucher's hold | touch | real1 0.5834 -> 0.5935; orig 0.6110 -> 0.6205; public5 0.4588 | ADOPTED (ADR-013) | L2-15; near-inert on provided match (possessions/half hour 112.46 -> 112.46); ball noise reverses it (0.1 m: 0.5812 vs 0.5839); ~45 candidates; Fable QA PASS (`L2-15-qa-fable.out.md`) |
| T42 | Learned touch-team model | touch | real1 0.596-0.603 | REJECTED | L2-15 "not adopted" |
| T43 | Duel margin variant | touch | real1 0.595-0.596 | REJECTED | L2-15 "not adopted" |
| T44 | duel=1 | touch | real1 0.5873 | REJECTED | L2-15 "not adopted" |
| T45 | One-at-a-time scan of 22 decoder constants (L2-15) | decoder | none above 0.5874 | NEUTRAL | constants flat; lower switch cost adds possessions, not score |
| T46 | Redefine switch date (touch minus true switch) | segment | median 0.00 s on all three matches | REJECTED | no definition fix exists (L2-15) |
| T47 | Haiku swarm on raw slices (7 workers, public GT slice) | other | | NEUTRAL | findings only: missing != dead; possession survives ~75% of gaps; dead = far+line+slowdown (`docs/research/swarm-findings.md`) |
| T48 | Realistic bench `real1` / `dead1` / `fault1` (moving dead ball, small jumps) | bench | default 0.5845 (orig) vs 0.5276 (real1) at L2-12 | ADOPTED | L2-12 audit; decisions now on real1 |
| T49 | IDSSE bench (idsse1 tune, idsse2 held-out) | bench | idsse1 team_control 0.3353 | ADOPTED | L3-idsse-01; 286 pred vs 511 true (under-segments); QA keep |
| T50 | Bench realism: hide ~28% of ball on true dead frames too | bench | | PENDING | in BACKLOG; `dhide` option in diagnose.py covers p sweeps (+0.046 p=0 ... -0.001 p=0.95 for T37) |
| T51 | Hold/freeze faults in bench (small jump 1/min only on) | bench | | NEUTRAL | hold and freeze options exist but off (not in provided input) |
| T52 | Noise gate (sigma 0.05/0.10 m, 7-12 players) as acceptance test | bench | pass for T25, T29, T34, T37, T38, T41 | ADOPTED | protocol practice since L2-7 |
| T53 | Input-level transfer check on provided match before adopting ball-in-stoppage features | bench | | ADOPTED | LOOP.md amendment after ADR-010; `transfer_check.py`, `transfer_*.json` |
| T55 | Fast-player speed (p90 of all players, 1 s / 5 s means) in the LOMO stoppage logistic (dead_fast) | shape | real1 0.5935 -> 0.6061; orig 0.6205 -> 0.6283; held-out look 7 real1 0.5619 -> 0.5788, orig 0.5787 -> 0.5968; public5 0.4588 -> 0.5294 | ADOPTED (ADR-014) | L2-16; hidden D share on provided match 0.673 -> 0.711 (risk); Fable QA PASS (`L2-16-qa-fable.out.md`) |
| T56 | w_sdead 0.4 with dead_fast | decoder | real1 0.6102 (one run) | NEUTRAL | L2-16 "not adopted"; single run, grid-edge selection risk |
| T57 | Speed quantile variants (p75 / top-3 / max) | shape | real1 0.602-0.606 | NEUTRAL | L2-16; p90 kept (QA: 0.5/0.75/0.9/1.0 -> 0.5937/0.6028/0.6061/0.6025) |
| T58 | Segment clean-up after Viterbi (Li & Link 2026): R1 flicker absorb, R2 D-bridge, R3 island-to-D, R4 sliver merge, protect touch-backed spells (src/possess/cleanup.py, opt-in, off) | decoder | real1 off 0.6061; R1 0.4/0.8/1.2/2 s 0.6047/0.5939/0.5861/0.5550; R3 best 0.6064; R4 best 0.6065 (metrica2 loses); best combo 0.6070 / orig 0.6293, 2 runs relabelled, bootstrap +0.0009 | REJECTED | Viterbi costs already remove jitter; short runs ~50% matched = break-even; weak 5 s+ islands need evidence not duration rules. L2-17 envelope |
| T59 | Exit snap: exit of hidden-ball stoppages >= 3 s moved to the largest step-up of the fast-player speed (p90, 1 s window) within +-2 s (edges.py, nothing learned) | decoder | real1 0.6061 -> 0.6081; orig 0.6283 -> 0.6305; bootstrap +0.0020 [+0.0002, +0.0041]; public5 0.5294 unchanged; hidden D share provided match 0.711 -> 0.724 | ADOPTED (ADR-015) | L2-18; grid ed_w {1,2} x ed_win {2,4} exit-only 0.6075-0.6084; Fable QA PASS (`L2-18-qa-fable.out.md`) |
| T60 | Exit + entry snap +-2 s | decoder | real1 0.6102; orig 0.6332; idsse1 hidden D precision -0.015 (limit 0.01) | REJECTED | L2-18; failed the designer's own D-precision gate, gate not relaxed |
| T61 | Exit + entry snap +-4 s | decoder | real1 0.6117 | REJECTED | L2-18; lengthens D, precision -0.019..-0.027 |
| T62 | M4 cue as transition cost | decoder | real1 0.6058-0.6086 (flat vs 0.6061) | NEUTRAL | L2-18 |
| T63 | M2 exit shift | decoder | real1 +0.0012 | NEUTRAL | L2-18 |
| T64 | M5 lone taker / M3 edge-aware stoppage model / M2 entry shift | decoder, shape | stop rules fired before a full run | REJECTED | L2-18; constant decoder costs cannot move an edge (M6 control); edge error is the stoppage-model window smear (H1) |
| T65 | real2 bench (smoothed ball, state-coupled hiding, player loss) as co-deciding bench | bench | re-scores steps: touch -0.024, duel -0.005, others within +-0.03 | ADOPTED | L2-20, ADR-016 |
| T66 | smooth_touch: 8-frame velocities, one touch per peak, 2 m radius on low-passed ball tracks | decoder, ballclean | real2 0.4303 -> 0.4833; real1/orig unchanged; public5 0.5294 -> 0.5059 (reported) | ADOPTED | L2-20, ADR-017; pending user confirmation (compliance) |
| T67 | Wider impulse lag alone on real2 | decoder | real2 0.436-0.437 | REJECTED | L2-20; far below smooth_touch |
| T68 | w_touch 0.35 variant | decoder | neutral | NEUTRAL | L2-20; not taken |
| T69 | L2-21 per-match unsupervised EM dead-ball prior on hidden frames (Saerens 2002; shift applied if abs >= 0.6 and >= 40000 hidden frames with evidence) | shape, decoder | real2 0.4833 -> 0.5173 (seed 2 0.4824 -> 0.5152, bootstrap +0.0339); real1/orig unchanged; public5 0.5059 unchanged (reported) | ADOPTED | L2-21, ADR-018; pending organiser answer (disclosure) |
| T70 | Visible-frame EM (same prior estimate on visible frames too) | shape | real2 0.5200 | NEUTRAL | L2-21; not taken (more scope, +0.003) |
| T71 | Shift cap 1.2 on the EM log-odds shift | shape | real2 0.5140 | REJECTED | L2-21; below uncapped 0.5173 |
| T72 | Carry team at ball disappearance through hidden blocks (L2-22) | team-control | oracle -0.0175 on real2 | REJECTED | team changes inside blocks (64/116 same); L2-22 envelope |
| T73 | Reappearance touch hold: only team within 1 m when ball reappears after >= 1 s hidden (L2-22) | touch | real2 0.5173->0.5326, 5 seeds pooled +0.0090; real1 -0.0013, orig -0.0023 | ADOPTED | ADR-019 |
| T74 | Vidal-Codina §2.3.3 pass-by filter: drop control frames with no change in ball direction/speed (L3-A1) | decoder | real2 0.5302 (-0.0025), real1 0.6030 (-0.0038); vmin 5.0: 0.5312 / 0.6036 | REJECTED | on a smoothed ball 41-59% of "pass-bys" are real control; decoder already overrides the false ones. agent/a1 40e390e |
| T75 | Vidal-Codina §2.3.1 duel prevails in the distance emission (no distance evidence in a duel) (L3-A1) | decoder | real2 0.5290, real1 0.6044, CI below 0 | REJECTED | in a duel the nearer team is usually right. agent/a1 |
| T76 | Vidal-Codina 0.5 m possession zone as distance cap (L3-A1) | decoder | real2 0.5259, CI below 0 | REJECTED | agent/a1 |
| T77 | Switch-timing post-pass: move H<->A switches earlier (edges.snap_switches; best = middle of the preceding duel) (L3-A2) | decoder | real2 seed 0 +0.0051 but seeds 1-3 median +0.0026; real1 -0.0031; orig -0.0028 | REJECTED | fails seed gate and real1 gate. Diagnosis kept: real2 switches late (median +3 fr, duel +19, flight +9); timing oracle +0.065, almost all from errors > 10 frames; 102/146 late switches have NO detected touch -> fix smoothed-ball touch detection, not placement. agent/a2 86b4bba |
| T78 | End the touch hold when the ball enters the opponent 1 m zone (L3-A2) | touch | real2 0.459-0.530 vs 0.5326 | REJECTED | agent/a2 |
| T79 | Stoppage entry cue: dead-ball evidence for the first 3 s of a hidden block when the ball was last seen within 2 m of a line or beyond it (deadevid.py, L3-A3) | dead-ball | real2 seeds +0.0036 / +0.0040 / +0.0033 (CI90 > 0 each); real1 +0.0003; orig +0.0006 | NEUTRAL (not adopted) | passes drop gates, fails median seed gain >= +0.005; ~0 on provided match (0.14% frames). Candidate to bundle. agent/a3 db03792 |
| T80 | Restart kick cue: play live for 2 s when the ball reappears at >= 8 m/s (L3-A3) | dead-ball | real2 +0.0087 with entry cue; real1 -0.0064 CI below 0 | REJECTED | on real1 91-96% of flipped frames are truly dead; part of the real2 gain is a bench artefact (hidden blocks end at a kick). agent/a3 |
| T81 | Vertex touch attribution on heavily low-passed ball tracks: contact point rebuilt from straight incoming/outgoing ball flight, touch credited to the team nearest it (decode.touch_vertex, vtx=2.0; guard VTX_ACC 70) (L3-A4) | touch | real2 0.5326->0.5618; 6 seeds pooled +0.029 CI90 [+0.017,+0.042]; real1/orig identical; sigma 5 +0.008 n.s., sigma 3 off by guard (was -0.031) | ADOPTED | ADR-020; Fable QA PASS with changes. agent/a4 |
| T82 | Min distance over a few frames around the touch (L3-A4) | touch | neutral, inconsistent across widths | NEUTRAL | in code, off |
| T83 | Paper §2.3.2 loss rule: end hold when the ball enters the opponent zone (L3-A4; same as T78) | touch | -0.013..-0.073, 30-130 false flips | REJECTED | in code, off |
| T54 | HSMM duration models | decoder | | PENDING | BACKLOG; "only if time" (PLAN step 8); see Open |

## Open, never tried (research plan / backlog)

Not in the registry above because no run exists:
- **P4 restart team from geometry (Laws)** as emission at D->live: goal kick -> defending team, corner -> attacking team, throw-in/free kick -> team at still ball (plan predicts +0.01..0.03 over P1). Note: L2-14 oracle says restart team +0.002 and 340/421 already right, so low expected value; new angle needed.
- **P6 set-piece geometry as dead evidence**, held until kick; stoppage start at line crossing. Only partly covered by T15/T37 (line proximity, player-only model); explicit geometry not built.
- **P7 per-source zone radius / context switch cost** (Vidal-Codina 0.5 or 1 m by provider) aimed at IDSSE under-segmentation (~350/511); grid only, never run per source.
- **Learned sequence model (Ballradar-lite / PathCRF-style)**: planned only if P3-P6 stall; overfit risk on 5 matches.
- **Segment clean-up post-processing** (Li & Link 2026: F1 0.61/0.68 -> 0.71/0.79 with merging/min-duration rules): minimum durations never tried.
- **Contact vs distance**: control only when the ball is slow relative to nearest player (Link & Hoernig 2017).
- **Hidden-ball stoppages**: partly addressed by L2-16 (T55, fast-player speed); the hidden-edge oracle is still +0.036.
- [SPECCED L2-19, DEPRIORITISED: real1 has more ball faults than the provided input; cleaner removes 0.0% there; expected ~0] **Cleaning the ball before `features`/distances and the stoppage detector** (not only touch ball); long freezes (0.4668 vs 0.5680 clean) and jumps < 1.4 m unhandled (L2-9 risks).
- **Hidden-ball possessions < 4 s** stay invisible without the ball (L2-10); only ball-free evidence could help.
- **IDSSE flickers**: opponent pokes in a duel, 1 s for the other team before a stoppage (L2-15 diagnosis): no kinematic cue found.
- **BACKLOG QA items**: idsse_cache.VERSION in derived cache names; tests for IDSSE get_case branch and ball-N gaps; fall back to DEC_BASE when direction vote share < 0.65 (with CLI warning); weight direction votes by player count; GK rule drops a deep defender when keeper untracked; pass fps through pipeline; `bench.py --pipeline possess` ignores positions; verify CLI CSV identical to evaluate cache.
- `w_sdead` > 0.3 (grid edge, L2-7).
| T84 | Controlled-possession post-pass ctl_s=1.0 (organisers' definition; user's deflection rule) judged on ctl T=2 truth (L4) | decoder | ctl T=2 pooled +0.0163 CI90 [+0.0106,+0.0219], T=3 +0.0236; old truth -0.0066 | ADOPTED | ADR-022; agent/a5 merge 618df4d |
| T85 | ctl_s=2.0 | decoder | ctl T=2 +0.020..+0.042 (partly by construction); old -0.015..-0.065; T=1 truth to -0.030; -28% possessions on provided input | REJECTED | ADR-022 |
| T86 | confirm_s=2 (touch-level confirmation) | touch | weaker than ctl_s on ctl truth | REJECTED | agent/a5 |
| T87 | Halves detector (activity + Law 8 kick-off formation, halves.py) | halves | external in/out acc .993-.999 (synthetic padding); provided-match transfer 0.872 | NOT MERGED | agent/a6 e65f322; opt-in only |
| T88 | Hidden H<->A switch moved to the change point of the all-player flow (react.place_switches, gap_cp=1, lag 0-2 s) (L5) | decoder | ctl T=2: real1 -0.012..-0.024, orig -0.012, real2 seeds -0.000..-0.008 | REJECTED | the flow reverses ~2 s after the turnover with a wide spread; the decoder's own placement is better (46-65% vs 21-29% within 1 s). agent/a7 |
| T89 | Directional switch discount on hidden frames where the flow reverses towards the new team (w_rev=2.0, rev_lag 1 s, rev_thr 1 m/s) (L5) | decoder | ctl T=2: real1 +0.0074 CI90 [+0.0015,+0.0141]; orig +0.0077; real2 seeds +0.0036 / +0.0014 / +0.0039 / +0.0002; old truth real1 +0.0077 | NEUTRAL (not adopted) | passes the drop gates, fails the real2 seed median (+0.0025 < +0.005); ~0 on the provided match (0.03% of frames). Candidate to bundle with T79. agent/a7 |
| T90 | All found hidden switches 0.25-1.0 s earlier (offline post-pass, diag only) (L5) | decoder | ctl T=2: real1 +0.0006 / +0.0020 / +0.0008 / -0.0017; real2 +0.0002 / -0.0003 / -0.0015 / -0.0030 | REJECTED | timing-only oracle is +0.015 (real1) / +0.003 (real2); the loss is unnamed 2-4 s possessions. agent/a7 |
