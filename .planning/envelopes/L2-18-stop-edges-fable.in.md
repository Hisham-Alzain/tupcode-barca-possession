```yaml
id: L2-18-stop-edges-fable
role: Fable designer/builder (hard task)
goal: Place the EDGES of stoppages (D) inside hidden-ball blocks better, so real1 tune beats the default at the time you start (0.6061 since L2-16, commit 540e7e6; L2-17 may move it - re-measure first) with orig drop <= 0.005. L2-16 left a hidden-edge oracle of +0.036 - missed D in the last 5 s (+0.022) and first 5 s (+0.017) of stoppages >= 3 s, middle +0.002; hidden restarts predicted a median 1.2-1.7 s early. Realistic size +0.005..+0.012 (a quarter to a third of the oracle). A clear negative with the diagnosis is a valid result.
start_condition: Start ONLY after L2-17-cleanup-fable is decided AND committed (it holds the RAM slot and owns src/possess/*, scripts/diagnose.py, scripts/tune_shape.py, tests/*). Build on the default that exists then. Run `evaluate.py --variant L2-18-base --bench real1 --no-public` and the same without `--bench` once; those two numbers are the bars. If L2-17 adopted cleanup.py, the edge pass of this package runs BEFORE its rules (after place_gap_switches) and you report the score with the rules on and off.
tried_ref: |
  .planning/TRIED.md - read it first. Neighbours you must NOT repeat:
  T40 (lead/lag stoppage model +-2 / +-4 s: AUC +0.003, real1 0.5725, removed), T45 (22 decoder
  constants one at a time incl. c_stop: flat), T54 (HSMM durations, pending, not this task), T36
  (still ball / kick restart: bench artefact, ADR-010 - no ball feature here), T55-T57 (fast-player
  speed and its quantile variants), T27 / T56 (w_sdead), T16 + "P4 restart team" (WHICH team
  restarts; this task is only WHEN), T46 (date of H<->A switches, not D edges), T17 (back-date of
  team switches), L2-17 R2 / R3 (create or remove D runs; this package never does).
new_angle: |
  1. A fact about the decoder nobody used: with constant transition costs the Viterbi path puts a
     D edge where the cumulative emission difference peaks, i.e. at the ZERO CROSSING of
     (e_d - e_team). c_stop / c_switch / c_same cannot move the edge of a long stoppage; they only
     decide whether a run exists. That is why T45 was flat. Edge timing is an emission-shape
     problem or a post-placement problem.
  2. Hypothesis H1 (smear): the stoppage model's heaviest columns are centred long windows (ps15
     w -1.42, ps90_5 w -1.82, ps_x_border, border5; player speed itself is a centred 0.8 s
     displacement, then a centred 1 s mean). 1.5 s before a restart a centred 5 s window already
     holds 1 s of post-restart running and a 15 s window holds 6 s: log-odds fall early = restart
     early. The mirror at stoppage onset = entry late. ONE cause predicts BOTH measured losses.
     Hypothesis H2 (pre-movement): players really start moving 1-2 s before the ball is put in play.
     Step 0 tells them apart.
  3. T40 added shifted copies to the per-frame state model and judged by AUC over ALL frames;
     edge frames are a few per cent of frames, so AUC could not see them. Here every mechanism
     is judged on the EDGE (signed offset in seconds; AUC inside +-3 s of true edges), and the
     main ones move an existing edge instead of re-voting the state.
  Write `new_angle:` in your out envelope.
matchlens_start: |
  State which you start from; keep the constants as anchors, cite source_file:function at every port.
  - src/possess/stoppage_ml.py DeadBallDetector._commit_flip: committed flips are BACK-STAMPED to
    the candidacy anchor (first supporting frame) so the dead interval starts when the transition
    began, not when it was confirmed. M1 / M2 below are the offline, player-only form of that idea
    for both edges. Anchors: HYSTERESIS_S 0.5 (edge tolerance), STATIONARY_DURATION_S 2.0 (search
    half-window), STATIONARY_DISP_WINDOW_S 1.0 (step window), NO_PLAYER_RADIUS_M 2.0 (= Law 15
    opponents' distance; taker cue M5).
  - team_control.team_from_distances backdate (T17) and deadball2.detect on_frames / off_frames
    (2 s on, 0.2 s off: MatchLens is already asymmetric between entry and exit).
  - stoppage_ml.py docstring: the 4th "all-stop" signal (ball AND players slow) was deferred in
    MatchLens; shape.py's stoppage model is that signal. PlacedBallDetector (min_hold_s 0.8) and
    MatchLens detectors/set_piece_detector.py are ball-based (L2-16 checked): unusable on hidden
    frames. Only their GEOMETRY (Laws 15/16/17, docs/research/literature/C_restarts/
    ifab-laws-restarts.md) is reused, on players. Re-check set_piece_detector.py for a taker /
    wall formation test before writing M5 and say what you found, with the files checked.
  Never import or path-reference MatchLens.
context_refs:
  - .planning/LOOP.md (protocol, transfer-check amendment, decide on real1, no-repeat amendment), .planning/TRIED.md, ADR-010..014 (+ the L2-17 ADR if any)
  - envelopes L2-10-hidden-fable.out.md, L2-13-stoppage-fable.out.md, L2-16-hidden-stop-fable.out.md, L2-16-qa-fable.out.md, L2-17-cleanup-fable.{in,out}.md
  - docs/research/literature/C_restarts/{ifab-laws-restarts.md, lang-2022-ingame-status.md, soccercpd-2022.md}, B_segmentation/killick2012-pelt.md
  - src/possess/shape.py (smooth = centred mean; _player_speed centred lag; features: DEAD_FEATS, FAST_FEATS, h["_sp"] per-player speeds; fit_models; emissions), decode.py (viterbi lines "c_other, c_back = ..." and the hidden branch `half = 0.5 * w_sdead * shape_dead`; decode_halves), pipeline.py
  - scripts/diagnose.py (stoppage_timing, boundaries, stoppage_signals, cmd_transfer, cmd_dhide), scripts/tune_shape.py (LOMO fits, caches FEAT_V 3, gate), scripts/evaluate.py, scripts/oracle.py, scripts/transfer_check.py
  - eval/results/{eval_L2-16_real1.json, eval_L2-17-base_real1.json, transfer_hidden_stop.json}
rules_reminder: Agents MAY read the provided match's INPUT under data/challenge (organisers' clarification, ADR-001) for transfer checks, aggregates only; NEVER the public ground-truth CSV (third_party/mtah/evaluation/gt/*); never tune, fit, calibrate or choose a threshold on the provided match (a shift calibrated in M2 comes from the tune matches only). Run evaluate.py with --no-public; the orchestrator reports public5. No held-out look (all 7 used).
owned_files: [src/possess/edges.py (new), src/possess/shape.py (new feature columns / model "dead_edge" / DEC_SHAPE dicts; keep the previous default under its own name; MODELS refit on tune only), src/possess/decode.py (optional per-frame transition-cost arrays in viterbi, post-pass hook in decode_halves), src/possess/pipeline.py (constants only), scripts/diagnose.py, scripts/tune_shape.py, tests/test_edges.py (new), tests/test_shape.py, tests/test_decode.py, .planning/envelopes/L2-18-stop-edges-fable.out.md]
step_0 (diagnosis; no config counted; one default decode per bench, labels + columns saved under new cache names, everything after reads the saved arrays): |
  D0a EDGE TABLE. real1, per match: true stoppages >= 3 s matched to the predicted D run with the
      largest overlap. For entry and exit separately, split by "edge frame hidden / visible":
      n, signed offset pred - true (median, IQR), share within +-0.5 s, share unmatched. Re-run the
      L2-16 edge oracle (first 5 s / last 5 s / middle) on the current default.
      STOP (whole package): hidden-edge oracle < +0.015, or |median offset| < 0.4 s on both edges.
  D0b ALIGNED PROFILES. Mean over true hidden exits (and entries) of stoppages >= 3 s, t = -8..+8 s:
      shape_dead log-odds, raw p90 player speed (no 1 s / 5 s smoothing), ps90, ps90_5, ps15, count of
      players faster than 2 m/s. Record the zero crossing of the mean log-odds and the half-rise
      time of raw p90, per match.
      H1 (smear) holds if raw p90 half-rise lies within +-0.5 s of the true edge while the log-odds
      cross >= 1 s early; H2 (pre-movement) if raw p90 itself leads by >= 1 s. H1 -> M1, M3, M4.
      H2 -> M2 (and M5); M1 / M3 / M4 are then skipped, because a sharper detector of the same
      early movement stays early.
  D0c TAKER PREVALENCE (for M5). Share of true hidden exits of stoppages >= 3 s with a "taker" in
      the 3 s before the exit; share of true LIVE frames with a taker (false rate); same two input
      shares on the provided match (no truth: share of frames with a taker, hidden / visible).
  D0d TRANSFER BASELINE (input only). Provided match and the three bench matches: predicted D runs
      >= 3 s per half hour, share of their edges on hidden frames, hidden D share (0.711 vs
      0.437 / 0.357 / 0.363 at L2-16), possessions per half hour (112.46), D run median s.
mechanisms (player-only, no ball column anywhere; all act per half, never write X, never create or delete a run except where stated): |
  M1 EDGE SNAP to the speed change point (nothing learned; edges.py; post-pass on labels).
     s(t) = mean of raw p90 over (t, t+w] minus mean over [t-w, t). For every predicted D run
     >= 3 s whose exit frame is hidden: move the exit to argmax s inside [exit - win, exit + win]
     if max s >= theta, else leave it. Entry: argmin s (step down), same window. An edge never
     passes the middle of the neighbouring run and never crosses a period break. The frames
     given up by D take the neighbouring team's label; frames gained become D.
     First diagnostic: on saved labels, median hidden exit offset must move toward 0 on all three
     matches and the share within +-0.5 s must rise. STOP: fails on 2 of 3 matches with w = 1 and 2.
  M2 PER-EDGE SHIFT (one learned scalar per edge, LOMO). delta_exit / delta_entry = median hidden
     offset from D0a on the OTHER two tune matches; shift predicted hidden edges of D runs >= 3 s
     by -delta, same guards as M1. First diagnostic: sign of the median equal on all three matches
     and IQR < 3 x |median|. STOP: sign differs in any match (Metrica and IDSSE annotate D edges
     differently) or |median| < 0.4 s. The shipped scalar = median over the three tune matches.
  M3 EDGE-AWARE STOPPAGE MODEL (LOMO logistic "dead_edge" = dead_fast + 3 columns): one-sided
     means of raw p90 (trailing 3 s, leading 3 s) and short speed x edge-ness (ps90 x |lead - trail|),
     so that near an edge the model reads the short window instead of the smeared long ones.
     First diagnostic: AUC restricted to frames within +-3 s of true edges of stoppages >= 3 s
     (hidden), LOMO, against dead_fast. STOP: gain < 0.01 on 2 of 3 matches -> no decode
     (this is where T40 died; say so and move on).
  M4 CUE AS TRANSITION COST (decode.py). Per-frame costs instead of constants on hidden frames:
     leaving D costs c_stop - w_r * r(t), entering D costs c_stop - w_s * q(t), with r / q = the
     positive step-up / step-down of M1 in tune-set SD units, clipped to [0, 3]; c_same unchanged.
     A cue is an event, so it belongs on the transition, not in the state emission; the team path
     can adapt, which a post-pass cannot. Run only if M1 gains >= +0.003 (the cue carries timing).
     Default arrays = constants must reproduce the current labels byte for byte (test).
  M5 LONE TAKER (Laws 15 / 17; geometry only, no identity). Taker at t = a player within 1.5 m of a
     touchline or 2 m of a corner, slower than 1 m/s for >= 1 s (STATIONARY_DISP_WINDOW_S), no
     opponent within 2 m (NO_PLAYER_RADIUS_M). A hidden D exit is not placed while a taker stands:
     it moves to the taker's release (speed > 1.5 m/s or band left) when that lies within 4 s
     after the predicted exit. Goal kicks are left out (a keeper always stands in the goal area).
     STOP (D0c): taker before < 25% of true hidden exits, or on > 5% of live frames, or the
     provided match's taker-frame share is outside 0.5x..2x the bench range (broadcast tracking
     may drop touchline players: 18% of hidden frames there have no shape evidence).
  M6 CONTROL - asymmetric constant costs (hidden frames: c_out 8 and 16 against c_in 12). The
     claim in new_angle 1 predicts the median exit offset moves < 0.2 s. If it moves more, the
     claim is wrong: report it and reconsider M4's design before running M4.
grid (<= 30 decodes or post-passes, fixed here; do not extend): |
  M1  w 1.0 / 2.0 x win 2 / 4 s, theta 0, exit only            4
      theta 0.5 m/s on the best                                 1
      entry only (best w, win 2 / 4)                            2
      exit + entry                                              1
      visible edges too                                         1
  M2  exit only, entry only, both, both at half size            4
  M3  three column sets (one-sided only; + interaction; one-sided replacing ps90_5) + best with M1   4
  M4  w_r 2 / 4 / 8; w_s 4; both                                5
  M5  taker hold after the best of M1 / M2; alone               2
  M6  control                                                   2
  combinations of survivors (at most)                           3
  Total 29. Order: step 0 -> M6 -> M1 or M2 (by H1 / H2) -> M5 -> M3 -> M4 -> combinations.
  A mechanism whose stop rule fires costs 0 configs. Prefer the smaller change inside a plateau
  (within 0.002). Record every run in eval/results/tune_edges.json (config, per-match score, counts).
constraints:
  - Tune ONLY on metrica1, metrica2, idsse1; decide on real1, report orig; LOMO for everything learned (M2 scalar, M3 model, M4 SD units), incl. mu / sd; no held-out; never the provided match.
  - Gates for adoption, all required. (1) real1 tune mean > the re-measured default; (2) orig tune drop <= 0.005; (3) paired bootstrap over true spells vs the default, CI90 lower bound > 0 (scorer noise ~0.04 per match, ~100 candidates tried since L2-14); (4) mechanism check - median hidden exit offset closer to 0 on all three matches and hidden D recall up without hidden D precision falling by more than 0.01 (a gain without this is not the mechanism: reject); (5) the config picked on any two tune matches does not lose on the third (read from the grid table); (6) noise gate at sigma 0.05 / 0.10 and 7-12 players, not below the default at the same noise (L2-16 default 0.6046 / 0.6037 / 0.5333; re-measure) - raw p90 is noisier than ps90, so if the gate fails retry once with the 0.4 s smoothed p90 and say so; (7) `diagnose.py dhide 0.8 0.95` keeps at least half the gain (the provided match sits between those); (8) transfer below.
  - Transfer (provided match INPUT only, aggregates; write eval/results/transfer_edges.json). With the chosen config report for the provided match and each bench match - hidden D share, possessions per half hour, D runs per half hour, D run median s, share of D edges moved, mean signed move of exits and of entries in s. PASS needs - hidden D share on the provided match rises by <= +0.02 absolute (<= 0.731 against 0.711; it is already above the bench and above the dhide truth of 0.60) and by <= 2x the largest bench rise; possessions per half hour change inside 0.5x..2x the bench range and not below -10%; mean signed edge move on the provided match inside 0.5x..2x the bench range with the same sign. A mechanism that only ever lengthens D fails if the bench D share does not rise with it.
  - RAM - before every run check >= 3 GB free with `python -c "import ctypes;c=ctypes;M=type('M',(c.Structure,),{'_fields_':[('l',c.c_ulong),('m',c.c_ulong)]+[(k,c.c_ulonglong) for k in 'tp ap tpf apf tv av ae'.split()]});m=M();m.l=c.sizeof(m);c.windll.kernel32.GlobalMemoryStatusEx(c.byref(m));print(round(m.ap/2**30,2),'GB free')"`; foreground only; one run at a time; < 10 min per command; one match in memory at a time; new caches only under new names (bump FEAT_V if a cached column changes); M1 / M2 / M5 run on saved labels + saved columns, not by re-decoding.
  - Code - edges.py is pure numpy on labels + period + visible + per-frame player columns; all parameters off returns the input unchanged; home / away swap symmetric; idempotent for M1 (second pass moves nothing beyond 1 frame). Parameters travel in the decoder dict (popped in decode_halves) so evaluate.py reproduces any default you set with no extra flags, and the old default stays reproducible under its own name. Do not "fix" the half-time smoothing note of L2-16 QA here unless an edge window would read across the break (then guard it and test it).
  - Tests - each mechanism on a hand-made sequence (step placed at a known frame is found; edge never crosses the neighbour's middle or a period break; no run created or deleted); off = identity; default pipeline output byte-identical with everything off; LOMO scalar never sees the scored match.
  - Do not commit. Do not edit docs or .planning files other than your out envelope.
deliverable: out envelope (<= 300 words) - D0a / D0b verdict (H1 or H2, with the offsets per match), which stop rules fired, grid table file path, chosen config or negative, real1 and orig tune per match + counts, hidden D recall / precision and median edge offsets before / after, bootstrap CI, noise gate, dhide, transfer numbers, reproducing commands, risks, matchlens_start / matchlens_delta, new_angle.
done_when: all gates pass with a config from the fixed grid, or a clear negative (a step-0 stop rule hit, or no config beats the default) with the edge table and the aligned profiles.
```
