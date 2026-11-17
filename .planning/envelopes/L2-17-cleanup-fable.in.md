```yaml
id: L2-17-cleanup-fable
role: Fable designer/builder (hard task)
goal: Segment clean-up after the decoder (minimum-duration / merge rules on the H/A/D label runs), so real1 tune beats the default at the time you start (0.6061 since L2-16, commit 540e7e6; re-measure first) with orig drop <= 0.005. Expected size is small (+0.003..+0.010); a clear negative with the diagnosis is a valid result.
start_condition: Start only after L2-16-hidden-stop-fable is closed (it holds the RAM slot and has uncommitted edits in shape.py, diagnose.py, tune_shape.py). Build on whatever default exists then; run `evaluate.py --variant L2-17-base --bench real1 --no-public` once and use that number as the bar.
tried_ref: .planning/TRIED.md "Open, never tried: Segment clean-up post-processing (Li & Link 2026)". Neighbours you must NOT repeat - T45 (22 decoder constants incl. c_switch scanned one at a time, flat), T14 (k-frame retention grid), T15 (deadball2 on/off hysteresis), T54 (HSMM durations, pending, not this task), T46 (switch date), T32 (long freezes).
new_angle: A uniform switch cost (T45) or frame hysteresis (T14, T15) cannot express CONTEXT-dependent minimum durations. These rules act on finished label runs and depend on (a) what flanks the run (same team both sides / D both sides / D one side), (b) run length, (c) ball hidden or visible, (d) whether a clean touch backs the run. Nothing learned, no ball-in-stoppage feature. Write `new_angle:` in your out envelope.
matchlens_start: Minimum durations already in the ported code - state which you start from and keep their constants as grid anchors. src/possess/stoppage_ml.py DeadBallDetector (leaky-bucket HYSTERESIS_S 0.5, STATIONARY_DURATION_S 2.0, GK_HOLDING_MAX_S 6.0) and PlacedBallDetector (min_hold_s 0.8); src/possess/team_control.py team_from_distances (k=5 control frames = 0.2 s before a gain, backdate; port of MatchLens possession.py zones); src/possess/deadball2.py detect (on_frames 50 = 2 s, off_frames 5 = 0.2 s); decode.touch_hold (max_hold_s 4, impulse.py port); decoder c_switch 4 / c_stop (an implicit minimum duration: an A-H-A flicker must earn > 2*c_switch = 8 log-units; with distance + rule + touch evidence up to ~2.6 per frame a 4-frame flicker can pay that). MatchLens has no segment-level merge pass in the ported files; say so after checking, with the files checked (CLAUDE.md "Start from where MatchLens ended"). Never import or path-reference MatchLens.
context_refs:
  - .planning/LOOP.md (protocol, transfer-check amendment, decide on real1, no-repeat amendment), .planning/TRIED.md, ADR-005..013
  - envelopes L2-8-segment-fable.out.md (588/1198 true possessions < 4 s; recall 0.11 under 2 s), L2-15-segment-fable.out.md (constants flat; IDSSE truth has real 1 s flickers before stoppages), L2-15-qa-fable.out.md (no paired bootstrap after ~45 candidates)
  - docs/research/literature/B_segmentation/li-link2026-intention-phases.md, A_control/li-link-2026-phases.md (3 s "Episode"), A_control/link-hoernig-2017.md (IBA ends after 1 s without interaction; contact vs distance)
  - vendor/mtah/possession_metrics.py (possession_spells, spell_overlaps, match_spells, SPELL_IOUS)
  - eval/results/eval_L2-15_real1.json, eval_L2-15.json, transfer_decoder.json
  - scripts/{diagnose.py (segment, transfer, length_table, matched_flags), evaluate.py (--bench real1 --no-public), tune_shape.py (grid, gate), transfer_check.py}
  - src/possess/{decode.py, pipeline.py, shape.py (DEC_SHAPE), team_control.py, stoppage_ml.py, deadball2.py}
rules_reminder: Agents MAY read the provided match's INPUT under data/challenge (organisers' clarification, ADR-001) for transfer checks, aggregates only; NEVER the public ground-truth CSV (third_party/mtah/evaluation/gt/*); never tune, fit or choose a threshold on the provided match. Run evaluate.py with --no-public; the orchestrator reports public5.
owned_files: [src/possess/cleanup.py (new), src/possess/decode.py (decode_halves post-pass hook only), src/possess/shape.py (DEC_SHAPE dicts only; keep the previous default under its own name), src/possess/pipeline.py (constants only), scripts/diagnose.py, scripts/tune_shape.py, tests/test_cleanup.py (new), tests/test_decode.py, .planning/envelopes/L2-17-cleanup-fable.out.md]
metric_note: |
  spellF1 = mean F1 over IoU 0.3..0.7 of a ONE-TO-ONE greedy match between true and predicted
  possession spells; a spell is a maximal run of H or of A, and any D run or team change ends it.
  F1 = 2*TP / (n_pred + n_gt). Consequences:
  - Wrong flicker (truth H for L frames, prediction H-A-H): three predicted spells, at most one
    can match, and only if the larger H piece is >= thr*L; a central flicker loses the TP at IoU
    >= 0.5 and adds 2-3 FP. Absorbing it gives IoU ~1: +1 TP at the high thresholds, -2 FP.
  - Wrong short D inside a possession (H-D-H) does the same damage with one FP fewer; bridging it
    repairs it.
  - The same edit on a REAL short spell or a real short stoppage is the mirror loss: up to 3 TP
    become 0-1 TP. 49% of true possessions are < 4 s, so a blind 3 s floor (Li & Link's Episode)
    must lose. Break-even is roughly "short predicted run of this kind is matched less than half
    the time"; measure that before choosing any threshold.
  - Counts today (real1, pred/true): metrica1 308/357, metrica2 274/330, idsse1 368/511 (L2-16). The
    pipeline already UNDER-segments and every rule below removes spells. With TP ~ 0.6*n, one
    removed unmatched spell is worth about +0.001 per match and one repaired match about +0.003;
    a rule that removes matched spells loses at the same rate. On idsse1 (0.71 of the true count)
    the room is smallest.
  - Li & Link's +0.10 came from cleaning a frame classifier with no sequence model. Ours is
    already a Viterbi path with switch and stop costs, so most of that gain is taken; what is
    left is runs of 4-50 frames carried by strong per-frame evidence, soft visible-frame D
    islands (sdead_vis), and what place_gap_switches leaves at block edges.
step_0_diagnosis (one run per bench variant, before any config; save the default labels + truth per match to a NEW cache name, then do everything else on those saved labels in one light process): |
  For predicted team spells and predicted D runs, by length bucket (< 0.4, 0.4-0.8, 0.8-1.2,
  1.2-2, 2-3, 3-5, > 5 s) x flank context (same team both sides / D both sides / D one side +
  other team / other) x hidden-or-visible majority: count, share matched at IoU 0.5, and the
  match rate of the two neighbours. For D runs: share overlapping a true D by >= 50%.
  STOP RULE: if no bucket with >= 15 runs over the three matches has a match rate < 0.4, write
  the negative result and do not run the grid.
rules (all in seconds, converted with fps; applied per half, after place_gap_switches; order R3, R2, R4, R1, repeated until nothing changes, at most 3 passes; threshold 0 = rule off): |
  R1 flicker   team spell shorter than m1 whose two neighbours are the same other team -> that team.
  R2 D-bridge  D run shorter than m2 whose two neighbours are the same team -> that team.
  R3 island    team spell shorter than m3 with D on both sides -> D.
  R4 sliver    team spell shorter than m4 between a D run and a spell of the other team -> the
               other team (D-h-A becomes D-A; A-h-D becomes A-D). Check the A-h-D direction
               separately: IDSSE truth has real 1 s spells before stoppages (L2-15).
  protect      (Link & Hoernig contact idea, from existing code only) a team spell is exempt
               from R1/R3/R4 when decode.touch_hold shows an uncontested touch by that team
               inside it. No new emission; contact-vs-distance as an emission stays open.
grid (<= 30 configs, fixed here; do not extend): |
  A  R1 m1 = 0.4, 0.8, 1.2, 2.0                      4
  B  R2 m2 = 1, 2, 3, 5                              4
  C  R3 m3 = 0.6, 1, 2, 3                            4
  D  R4 m4 = 0.4, 0.8, 1.2                           3
  E  protect=1 on the best of A and the best of D    2
  F  best of A on hidden-majority runs only / visible-majority only   2
  G  combinations of the per-rule winners: A+B, A+C, B+C, A+B+C, A+B+C+D, best of G + protect   6
  Total 25. A rule whose best value does not beat "off" is dropped from G. Prefer the smaller
  threshold inside a plateau (within 0.002).
constraints:
  - Tune ONLY on metrica1, metrica2, idsse1; decide on real1, report orig; no held-out look.
  - Gates for adoption, all required. (1) real1 tune mean > the re-measured default (0.6061 at L2-16); (2) orig tune drop <= 0.005 (>= 0.6233 against 0.6283); (3) paired bootstrap over true spells vs the default, CI90 lower bound > 0 (the scorer's own noise band is 0.04 and ~70 candidates will have been tried); (4) the chosen config, picked on any two tune matches, still does not lose on the third (read from the grid table, no extra runs); (5) noise gate at sigma 0.05 / 0.10 and 7-12 players, not below the default at the same noise (L2-16 default 0.6046 / 0.6037 / 0.5333); (6) LOMO only if you fit anything, and the envelope says nothing should be fitted; (7) transfer check below.
  - Transfer (provided match INPUT only, aggregates) with the chosen config - possessions per half hour (now 112.46), D runs per half hour (25.49), D run median s (17.62), and the share of predicted runs each rule touches. The relative change in possessions per half hour on the provided match must sit inside 0.5x..2x the range of the three bench changes and must not exceed -10% (T39 was rejected at -24% vs 1-7%). If the provided match has far more short runs of a kind than the bench, that rule fails transfer.
  - Report per match pred/true counts before and after; never hide that the count moved further below truth.
  - RAM - before every run check >= 3 GB free with `python -c "import ctypes;c=ctypes;M=type('M',(c.Structure,),{'_fields_':[('l',c.c_ulong),('m',c.c_ulong)]+[(k,c.c_ulonglong) for k in 'tp ap tpf apf tv av ae'.split()]});m=M();m.l=c.sizeof(m);c.windll.kernel32.GlobalMemoryStatusEx(c.byref(m));print(round(m.ap/2**30,2),'GB free')"`; foreground only; one run at a time; < 10 min per command; new caches only under new names; the grid runs on saved labels, not by re-decoding.
  - Code - cleanup.py is pure numpy on a label array + period + visible (+ touch mask); never crosses a half boundary; never writes X; all thresholds 0 returns the input unchanged. Parameters travel in the decoder dict (popped in decode_halves), so evaluate.py reproduces any default you set with no extra flags and the old default stays reproducible under its own name.
  - Tests (tests/test_cleanup.py) - each rule on a hand-made sequence; thresholds 0 = identity; idempotent; home/away swap symmetry; half boundary untouched; protect keeps a touched spell; default pipeline output byte-identical when the rules are off.
  - Do not commit. Do not edit docs or .planning files other than your out envelope.
deliverable: out envelope (<= 300 words) - step-0 table (the buckets that matter), grid table file path, chosen config or negative, real1 and orig tune per match + counts, bootstrap CI, noise gate, transfer numbers, reproducing commands, risks, matchlens_start / matchlens_delta, new_angle.
done_when: all gates pass with a config from the fixed grid, or a clear negative (step-0 stop rule hit, or no config beats the default) with the diagnosis table.
```
