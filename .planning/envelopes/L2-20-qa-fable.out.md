```yaml
id: L2-20-qa-fable
status: real2 PASS | smooth_touch=1 PASS with changes (adopt) | compliance: defensible, residual risk is the user's call
```
**Reproduced** (`evaluate.py --variant L2-20-qa[-off] --bench X --no-public [--dec smooth_touch=0]`): real2 0.4833, off 0.4303; real1 0.6081 and orig 0.6305 with the switch on and off. pytest 145 pass. No held-out run, public truth never opened; no ground-truth path in any changed file.

**real2 sound.** Rebuilt orig and real1 cases in memory with the new code: array-identical to the caches written before the change (3 matches). real2 is deterministic; its truth equals the orig truth. Constants match input aggregates only: evaluation design, as L2-12.

**Evidence.** Second seed built in memory: on 0.4823, off 0.4331. Paired 2-min block bootstrap of the gain: seed 0 +0.053 [+0.037, +0.070], seed 1 +0.049 [+0.030, +0.067] (5-95%), no draw <= 0 of 2000; all 6 match-seeds up.

**Compliance.** Acceleration p99 per half, as the detector sees it: external unsmoothed 204-619 (5 matches, all variants), real2 36-41, provided 30 / 47. Any threshold from 48 to 204 gives identical output everywhere, so 150 is not fitted. Lag 6 / 8 / 10 and sigma 3 / 5 / 7 all gain: nothing is tuned finely to the provided statistics. The 8-frame / peak / 2 m constants were chosen on external truth; the higher-scoring w_touch=0.35 was not taken. Weak point: the branch exists because of the provided input and fires only there. Record the argument in an ADR and disclose it in the submission notes.

**Findings**
1. `decode.py:110` and `shape.py:508` say "280-570"; real range 204-619 (orig idsse2 half 2 = 204). Margin above 150 is 1.36x.
2. `decode.py:267-271`: the 2 m radius also reaches `clean_ball` (teleport keep radius). Undocumented; included in the scored number.
3. `decode.py:389`: clean-up protect path ignores the smooth regime (off by default).
4. The gate is load-bearing: SMOOTH_TOUCH on real1 scores 0.539 (-0.069). Decided per half, so halves can disagree.
5. `bench.py:548`: real2 distance features still contain the lost players (negligible).
6. Pre-existing `bench.py:109` cites a D-frame share of the challenge match; confirm its provenance.

**Transfer.** Provided hold share 0.67 sits at the low end of real2 (0.66-0.75); spell count +6% (real2: mixed sign). Plausible, unverifiable. Share of frames changed on real2 not measured.

**Recommendation.** Adopt after fixing 1-2. real2 co-decides: adopt a change only if neither real1 nor real2 drops by more than 0.003 and one gains.

**Files written:** this envelope; `eval/results/eval_L2-20-qa*.json` (6, by evaluate.py).
