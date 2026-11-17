```yaml
id: L2-20-qa-fable
role: Fable cold QA (correctness + rule compliance + decision advice)
goal: Decide (1) whether the new bench variant real2 is sound (real1/orig byte-identical, audit honest, no truth of the provided match used) and whether it should co-decide with real1; (2) whether the uncommitted smooth_touch=1 change is correct, leak-free, rule-compliant and safe to make the default. Verdict per item PASS / FAIL with findings.
context_refs:
  - git diff + untracked files (scripts/bench.py, audit_bench.py, evaluate.py, tune_shape.py, src/possess/decode.py, shape.py, tests/test_smooth_touch.py, test_decode.py, test_edges.py), eval/results/audit_inputs_real2.json, l220_*.json*, transfer_smooth_touch.json, eval_L2-20*.json
  - .planning/envelopes/L2-20-real2-fable.in.md and .out.md (builder's claims), .planning/research/provided-vs-bench-gaps.md
  - CLAUDE.md, docs/RULES.md (organisers' rules incl. "Training, fitting or calibrating your algorithm on the ground-truth possession data or on the provided video" is forbidden; train/tune on external data only), ADR-001, ADR-010 (L2-11 rejected rule that did not transfer), L2-12 precedent (real1 built from the provided gap histogram)
rules_reminder: Agents MAY read the provided match's INPUT under data/challenge for aggregates; NEVER open the public ground-truth CSV (third_party/mtah/evaluation/gt/*); --no-public always; no held-out runs (all looks used).
owned_files: [.planning/envelopes/L2-20-qa-fable.out.md]
checks:
  - Reproduce: default real2 0.4833, real1 0.6081, orig 0.6305 (run orig too), `--dec smooth_touch=0` real2 0.4303; `uv run pytest` passes; real1/orig bench cases byte-identical to before (how was this verified?).
  - smooth_touch logic: regime detector (acc p99 < 150 m/s2) — computed per half or per match? robust to ball gaps/NaN? what happens near the threshold; 8-frame velocity, peak picking, 2 m radius — off-by-one, period boundaries, hidden frames.
  - RULE COMPLIANCE (most important): the 150 threshold and the smoothing-mode constants were chosen by a builder who had seen the provided match's acceleration aggregates (30/47). Is this "calibrating the algorithm on the provided video"? Can the threshold be justified independently (physics of real ball tracks: kick accelerations, raw-tracker statistics on all 5 external matches, the gap between 280-570 and smoothed values) so that any value in a wide range gives the same decision on every dataset? Report how wide the range is that yields identical output on all external matches AND the provided match (input-only check). Were the 8-frame / 2 m constants chosen on real2 (external truth) — fine — or with provided-match information?
  - Is real2 itself evaluation design (as L2-12) or calibration? Same question for its constants.
  - Evidence quality: one seed, ~20 configs, no paired bootstrap. Run a paired bootstrap (or second seed of real2 if cheap) if feasible within time.
  - Transfer: touch-hold share 0.27 -> 0.67 on the provided input, possessions/half hour 112.5 -> 119.0, 4.8% frames change. Consistent with real2's behaviour?
  - Recommendation: adopt / adopt with changes / reject, and whether real2 should co-decide with real1 (and with what rule).
constraints:
  - RAM: check >= 3 GB free before every run (python ctypes GlobalMemoryStatusEx); foreground; one run at a time; < 10 min per command.
  - Do not edit source files. Do not commit.
deliverable: output envelope <= 350 words: verdicts, compliance argument, findings with file:line, commands run.
```
