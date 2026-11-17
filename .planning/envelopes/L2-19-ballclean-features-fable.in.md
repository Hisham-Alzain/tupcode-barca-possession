```yaml
id: L2-19-ballclean-features-fable
role: Fable designer/builder (hard task)
goal: Decide, with numbers, whether cleaning the ball BEFORE the distances (d_home / d_away), the MatchLens stoppage detector (ml_dead) and deadball2 - not only before touch_hold (ADR-008) - is worth anything, and adopt it only if real1 tune beats the default at the time you start (0.6061 since L2-16, commit 540e7e6; L2-18 may move it - re-measure first) with orig drop <= 0.005. Realistic size - real1 +0.000..+0.004, a fault-heavier variant +0.005..+0.02, the provided match ~0. A clear negative that closes the TRIED open item is a valid and likely result.
start_condition: Start ONLY after L2-18-stop-edges-fable is decided AND committed (it holds the RAM slot and owns src/possess/*, scripts/diagnose.py, scripts/tune_shape.py, tests/*; decode.py and shape.py are dirty until then). Build on the default that exists then. Run `uv run python scripts/evaluate.py --variant L2-19-base --bench real1 --no-public` and the same without `--bench` once - those two numbers are the bars (real1 bar, orig bar).
tried_ref: |
  .planning/TRIED.md - read it first. This is the open item "Cleaning the ball before features /
  distances and the stoppage detector; long freezes and jumps < 1.4 m unhandled (L2-9 risks)".
  Neighbours you must NOT repeat:
  T31 (cleaning for touch_hold: adopted, +0.002, inert on the real ball), T32 (removing freezes > 2 s
  from the TOUCH ball: idsse1 0.530 -> 0.512, the exit jump is the restart touch), T33 (off-pitch rule:
  removes the real ball crossing the line), T36 / ADR-010 (still ball as stoppage evidence: bench
  artefact - no new ball-speed or still-ball evidence here), T03-T10 (any gap filling / interpolation
  as a lever), T51 (hold / freeze faults exist in bench.py but are off: not in the provided input),
  T41 (lesson: a touch-side change that wins clean can lose under 0.1-0.2 m ball noise).
new_angle: |
  1. L2-9 measured faults injected into the touch ball (v2: + distances) on the ORIG bench, at rates
     chosen by hand (1 per 2 s / 10 s). Nobody has compared those rates with the provided input, class
     by class. Numbers already on disk (eval/results/audit_transfer{,_real1}.json, input only):
       provided: steps > 35 m/s 4.81 / min (> 60 m/s 1.02), exact repeats 0.08% (run mean 1.6, p90 3
                 frames), freezes >= 0.4 s 0.0 / min, ball noise 0.0003 m, ballclean removes 0.0%.
       real1   : steps > 35 m/s 5.9-7.5 / min, exact repeats 1.4-2.2%, freezes >= 0.4 s 0.9-1.06 / min
                 (the placed dead ball), ballclean removes 0.05-0.16%.
     So on these three statistics real1 already holds MORE faults than the provided match, and the
     "long freezes" half of the item (0.4668 vs 0.5680) describes a fault the provided input does not
     show. What is NOT measured: (a) what the 4.8 fast steps / min on the provided match are (kicks,
     return blips under dev_min, one-way track jumps, gap-edge blips) - the cleaner removes none of
     them; (b) blips < 1.4 m, invisible to a 35 m/s gate; (c) per-axis clamps (swarm: x = -51.8 m 19
     times, readings 4-8 m beyond the lines) which `exact_repeat_share` (x AND y equal) cannot see.
  2. Step 0 is therefore a census by fault CLASS on the input, and a go / no-go, before any mechanism.
  3. T32 new angle (only if step 0 finds freezes on the provided input): the touch ball keeps the
     freeze and its exit jump; only the distance emission treats the frozen frames as "no evidence".
  Write `new_angle:` in your out envelope.
matchlens_start: |
  State which you start from; keep the constants as anchors, cite source_file:function at every port.
  - src/possess/ballclean.py (port of the MatchLens ball tracker's "coast through a bad detection" +
    events/kinematics.py maximum plausible ball speed gate; v_max 35, max_tp 3, dev_min 1.5, stale_max 10).
  - src/possess/stoppage_ml.py: ball_speed_sg(), DeadBallDetector._slow_verdict (displacement fallback:
    median position of the older half of a window vs the newer half, STATIONARY_DISP_WINDOW_S 1.0 -
    MatchLens's own jump-robust reading of a ball; anchor for a median-based distance ball),
    PlacedBallDetector (min_hold_s 0.8). deadball2.short_median / lag_speed (5-frame median, L2-11).
  - decode.ball_impulse noise scaling (noise_k 3 x the ball's own measured noise, port of
    impulse.py:impulse_verdict): the anchor for any small-jump threshold (never a fixed value under noise).
  - Re-open MatchLens events/kinematics.py and the core ball tracker for an acceleration / innovation
    gate that catches sub-1.4 m blips; say what you found, with the files checked. "Nothing covers
    per-axis clamps or one-way jumps" is a valid answer after looking. Never import or path-reference MatchLens.
context_refs:
  - .planning/LOOP.md (protocol, transfer-check amendment, decide on real1, no-repeat amendment), .planning/TRIED.md, ADR-001, ADR-008, ADR-010, ADR-011..014 (+ the L2-18 ADR if any)
  - envelopes L2-8-segment-fable.out.md, L2-9-ballclean-fable.{in,out}.md, L2-12-audit-fable.out.md, L2-15 QA, L2-16-hidden-stop-fable.out.md, L2-18-stop-edges-fable.out.md
  - docs/research/swarm-findings.md section 2 "Data problems"
  - src/possess/{ballclean.py, features.py (compute - d_home / d_away and stop.update read the RAW ball), stoppage_ml.py, deadball2.py (features - far, line, vlag), decode.py (viterbi / touch_hold ball_clean=1, ball_noise), pipeline.py}
  - scripts/bench.py (FAULTS, VARIANTS, tracking_faults, get_case, case_features), scripts/transfer_check.py (signal_stats), scripts/diagnose.py (inject_faults, faulted real_d, cmd_transfer), scripts/tune_shape.py (load_all ball_noise, positions, gate), scripts/evaluate.py
  - eval/results/{audit_transfer.json, audit_transfer_real1.json, eval_L2-9-ballclean-v2.json, eval_L2-18*.json}
rules_reminder: Agents MAY read the provided match's INPUT under data/challenge (organisers' clarification, ADR-001), aggregates only (rates, histograms, shares; no per-frame dump into context or files outside eval/results aggregates); NEVER the public ground-truth CSV (third_party/mtah/evaluation/gt/*); never write under data/challenge/tactical; never tune, fit, calibrate or choose a PIPELINE threshold on the provided match. Run evaluate.py with --no-public; the orchestrator reports public5. No held-out look (all 7 used).
owned_files: [src/possess/ballclean.py, src/possess/features.py (a `reclean` helper only; `compute` output unchanged when cleaning is off), src/possess/deadball2.py (input ball only; no new evidence term), src/possess/decode.py (which ball / distances viterbi reads; keep ball_clean=1 semantics under its name), src/possess/pipeline.py (wiring + constants), scripts/bench.py (a new VARIANTS entry + FAULTS-style dict only; "" / dead1 / real1 / fault1 byte-identical), scripts/transfer_check.py (new census keys only), scripts/diagnose.py (new `ballfeat` command), scripts/tune_shape.py (ball noise reaching the distances, new option), tests/test_ballclean.py, tests/test_features.py, .planning/envelopes/L2-19-ballclean-features-fable.out.md]
step_0 (census + go / no-go; no config counted; input statistics only, no score): |
  S0a FAULT CENSUS BY CLASS, on the provided INPUT, on orig and on real1 (three tune matches each),
      per minute of visible ball unless noted, written to eval/results/census_ballfeat.json:
      1. return blips: runs of 1-3 frames off the line between the frames around them, by deviation
         bin 0.3-0.5 / 0.5-1.0 / 1.0-1.5 / 1.5-2.4 / > 2.4 m, split "a player within 1 m / nobody";
      2. one-way jumps: a step > 35 m/s that does not come back within 3 frames, nobody within 1 m;
      3. gap-edge blips: unreachable readings in the first / last 3 frames of a visible stretch;
      4. freezes: exact repeats (x AND y) by run length 2-3 / 4-10 / 11-50 / > 50 frames, split
         "entered by a moving ball / by a resting ball";
      5. per-axis clamps: runs where x (or y) repeats exactly while the other axis moves; share of
         visible frames on a value seen >= 10 times in non-adjacent runs;
      6. off pitch: share of visible frames > 0 / 1 / 3 m outside the lines;
      7. white noise (decode.ball_noise).
      Classify the 4.8 fast steps / min of the provided match into 1 / 2 / 3 / "kick (player within
      1 m, stays near the line)".
  S0b DOWNSTREAM EXPOSURE, same three sources, no truth: share of visible frames where a "generous"
      cleaner fixed IN ADVANCE (current clean_ball + blips >= 0.5 m + 5-frame median, written down
      before it is run) changes (i) the membership d_home <= 1 m or d_away <= 1 m, (ii) deadball2
      ev_dead, (iii) ml_dead; isolated 1-2 frame flickers of "d <= 1 m" per minute; touch onsets within
      +-2 frames of a class-1/2/3 fault per minute. On the provided match also: share of frames whose
      final label changes and possessions per half hour, generous cleaner vs default.
  S0c GO / NO-GO.
      NO-GO (stop the package, write the negative, propose the TRIED row): every class rate on the
      provided input <= 1.5 x its real1 rate AND the generous cleaner changes < 0.1% of the provided
      match's labels and < 1 possession per half hour. Then real1 already contains the provided
      faults and the item is inert on the target; only the placement run A4 (below) is still made,
      once, as the record (1 config).
      GO-real1: a class is present on the provided input at <= 1.5 x real1 -> mechanisms are measured
      on real1 as it is.
      GO-real2: a class is >= 1.5 x more frequent on the provided input than on real1 (and >= 0.5 /
      min, or >= 0.2% of visible frames) -> add ONE variant "real2" = real1 with only that class's
      generator parameters moved to the provided rate and size histogram (jump_m / jump_per_min /
      jump_frames, a one-way jump or clamp generator if needed). Freeze / hold options stay OFF unless
      class 4 / 5 is actually measured on the provided input (today: 0.0 / min).
  S0d WHY real2 IS NOT TUNING ON CHALLENGE DATA - write this argument, with the numbers, in the out
      envelope; the orchestrator records it as an ADR before real2 is used for a decision:
      1. Only label-free INPUT statistics are read (ADR-001 allows it); no ground truth, no score on
         the provided match is computed, read or used.
      2. What is set from them is a parameter of the bench's degradation GENERATOR, never a pipeline
         parameter. Same act and same precedent as real1 (L2-12: FAULTS and wander set from
         transfer_check input statistics; LOOP.md "decide on real1").
      3. Every pipeline constant is then chosen by spellF1 on external truth (metrica1, metrica2,
         idsse1) or kept from MatchLens / physics. Forbidden: placing a cleaning threshold where the
         provided histogram has its mass (e.g. dev_small "because the provided blips are 0.7 m").
      4. One calibration pass, frozen before the first mechanism run: the real2 dict, its seed and
         the census table are written to the out envelope first; no second pass after seeing scores
         (a bench re-drawn until a fix wins would be the fix tuning its own exam).
      5. The provided match afterwards acts as a VETO only (transfer check): it can reject a change,
         it can never select one or rank two candidates.
      6. Residual risk, to state plainly: real2 is built to contain the fault the cleaner removes, so
         a real2 gain proves robustness to that fault, not a gain on the real match; that is why the
         adoption gate stays on real1.
mechanisms (ideas with an order, not orders; <= 30 scored configs in total, every one a row in eval/results/tune_ballfeat.json): |
  Implementation note (RAM): do not re-run the frame loop. `features.reclean(f, pos, ...)` recomputes
  d_home / d_away from the cached anonymous positions (tune_shape.positions, as diagnose.faulted does)
  and, for A3, re-runs MatchLensStoppage on the cleaned ball. FIRST check that reclean with cleaning
  OFF reproduces the cached d_home / d_away / ml_dead exactly; if ml_dead cannot be rebuilt from
  anonymous positions (goalkeeper role), say so, keep ml_dead raw and drop A3.
  A PLACEMENT of the existing cleaner (current constants, no new rule) - 4 configs:
      A1 distances only (decoder distance emission, frame rule, touch r_touch test);
      A2 A1 + deadball2 features (far, line; vlag stays opt-in off);
      A3 A2 + ml_dead from the cleaned ball;  A4 = A3 + the touch ball (everything reads one ball).
      Removed frames that are not bridged carry NO distance evidence (nan), never "99 m".
  B SMALL JUMPS (< 1.4 m; only after GO for class 1 / 2 / 3) - <= 10 configs:
      B1 return-blip rule without the speed gate: run <= 3 frames, deviation from the neighbour line
         >= max(dev_small, 3 x measured ball noise), nobody within r_touch, bridged. dev_small 0.5 /
         0.8 / 1.0 m.
      B2 5-frame median ball (deadball2.short_median) for the DISTANCES only; the touch ball keeps
         the raw + clean_ball track (a median blunts real impulses). Windows 3 / 5.
      B3 one-way jump / gap-edge blip: the frames between the jump and the nearer end of the visible
         stretch (<= 3 frames) give no distance evidence. 2 configs.
  C FREEZES / CLAMPS (ONLY if S0a finds class 4 / 5 on the provided input; otherwise skipped and
      recorded as "fault absent from the input, nothing to measure") - <= 4 configs: frozen or
      clamped runs entered by a MOVING ball give no distance / line evidence for their duration; the
      touch ball is untouched (T32). Runs entered by a resting ball are real stops and are kept.
  D COMBINATION - <= 4 configs: best placement + best B (+ C), then the same with one neighbour value.
  Budget check: A 4 + B 10 + C 4 + D 4 + NO-GO record 1 = 23; 7 spare. Gate and seed re-runs of the
  final candidate are not configs.
stop_rules: |
  - S0c NO-GO ends the package after the single A4 record run.
  - A4 within +-0.001 of the real1 bar AND (real2 built) within +0.003 of the real2 bar -> stop;
    placement does not matter, B / C cannot be judged on a ball the pipeline is indifferent to.
  - A family stops after two consecutive configs within +-0.001 of its best.
  - Any config that drops a tune match by > 0.01 on real1 is dead; do not rescue it with a second knob.
  - 30 scored configs, or 3 hours of wall time: stop and report what you have.
gates (all needed for "adopt"; report every number, failures included): |
  G1 real1 tune mean > the real1 bar (re-measured at start), no tune match down by > 0.003.
  G2 orig tune mean >= orig bar - 0.005.
  G3 paired bootstrap (2000 draws) of the real1 gain: report CI90; say how many configs were scored.
  G4 noise gate, candidate vs default under the SAME degradation, candidate must not be worse by
     > 0.002 in any row: player sigma 0.05 / 0.10 m; 7-12 players; ball noise 0.1 / 0.2 m with the
     noise reaching the DISTANCES and the cleaner (tune_shape's ball_noise today touches the touch
     ball only - add an option, keep the old one); a second fault seed for real1 (and real2).
  G5 transfer check on the provided INPUT (veto only), default vs candidate: possessions per half
     hour (112.46 at L2-16 - re-measure), D share on hidden and on visible frames (hidden 0.711 at
     L2-16), share of visible frames the cleaner removes / bridges, share of labels changed. Fail if
     possessions move by > 5% or > 3 x the bench's relative move (T39), or a D share moves by > 0.01
     against the bench's direction.
  G6 no held-out look. evaluate.py must reproduce any default you set with no extra flags; cleaning
     OFF must reproduce the bars exactly (pure wiring).
  If G1 fails but real2 gains > +0.003 with real1 and orig equal (+-0.001) and G4 / G5 pass: report
  it as a "robustness candidate at equal score" (ADR-008 precedent) - do NOT switch the default; the
  orchestrator decides.
constraints:
  - Tune ONLY on metrica1, metrica2, idsse1 (truth from the bench); decide on real1; report orig and real2.
  - RAM: check >= 3 GB free before every run (python ctypes GlobalMemoryStatusEx); foreground; one run at a time; < 10 min per command; new caches only under new names (bench `*_real2.npz`, `feat_*_real2.npz`, recleaned features `*_bc<N>.npz`; never overwrite `*_v2`, `*_real1`, FEAT_V caches); grids <= 30.
  - No learned model is refit (shape MODELS untouched; hidden mask and truth identical across variants).
  - Do not commit. Do not edit docs or .planning files other than your out envelope. Raw logs to files, not into context.
deliverable: census table (S0a / S0b, provided vs orig vs real1) + go / no-go verdict + the real2 argument (if built) + change + tests (identity when off; a blip next to a player is kept; noise does not trigger B1; a freeze entered by a resting ball is kept) + output envelope (<= 300 words) with real1 / orig / real2 tune per match + counts, G3-G5 numbers, reproducing commands, risks, matchlens_delta, new_angle, proposed TRIED rows.
done_when: G1-G6 pass (adopt), or a robustness candidate is handed to the orchestrator, or a clear negative with the census that shows why (the fault classes are absent from, or already rarer in, the provided input).
```
