```yaml
id: L2-22-qa-fable
role: Fable cold QA (correctness + robustness of a borderline gain)
goal: Decide whether the uncommitted opt-in `reappear=1.0` (touch_hold: first visible frame after a hidden block >= 1 s starts a touch hold for the only team within 1 m) should become the default. Verdict PASS (adopt) / FAIL (keep opt-in, record as NEUTRAL) with findings.
context_refs:
  - git diff (src/possess/decode.py, scripts/diagnose.py, tests/test_decode.py), eval/results/eval_L2-22*.json, transfer_reappear.json
  - .planning/envelopes/L2-22-hidden-carry-fable.in.md + .out.md (builder's claims; note the builder went past the step-0 stop rule, which was about the carry oracle — judge whether that matters)
  - .planning/LOOP.md decision rule (ADR-016: neither real2 nor real1 drops > 0.003, at least one gains, orig drop <= 0.005, bootstrap + second real2 seed agree in sign), ADR-005..018, .planning/TRIED.md
rules_reminder: Agents MAY read the provided match's INPUT for aggregates; NEVER open the public ground-truth CSV (third_party/mtah/evaluation/gt/*); --no-public always; no held-out runs.
owned_files: [.planning/envelopes/L2-22-qa-fable.out.md]
checks:
  - Reproduce with `--dec reappear=1.0`: real2 0.5326, real1 0.6068, orig 0.6283; off reproduces 0.5173 / 0.6081 / 0.6305; pytest passes; default (no flag) unchanged.
  - Robustness (the key question): seed 0 +0.0153, seed 1 only +0.0036; noise-gate CI includes 0. Run at least 3 more real2 seeds in memory (no caches) and report the per-seed, per-match gains and a pooled paired bootstrap. Adopt only if the pooled CI90 lower bound > 0 and the median seed gain is >= +0.005.
  - Logic: block-length threshold, "only team within 1 m" with NaN players, period boundaries, interaction with smooth_touch and duel=2, the 68/207 fires on truly dead balls (does the D decoding override them?).
  - Transfer: rule fires on 77/131 provided blocks, 0.7% frames change — consistent?
constraints:
  - RAM: check >= 3 GB free before every run (python ctypes GlobalMemoryStatusEx); foreground; one run at a time; < 10 min per command.
  - Do not edit source files. Do not commit.
deliverable: output envelope <= 300 words: verdict, seed table, pooled bootstrap, findings with file:line, commands run.
```
