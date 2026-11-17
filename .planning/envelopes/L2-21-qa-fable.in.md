```yaml
id: L2-21-qa-fable
role: Fable cold QA (correctness + rule compliance + transfer judgment)
goal: Decide whether the uncommitted L2-21 change (shape.prior_shift / em_prior / dead_prior: per-match unsupervised EM estimate (Saerens 2002) of the hidden-frame dead-ball share from the stoppage model's own posteriors; hidden log-odds shifted, no-evidence hidden frames get the estimated prior; gate |shift| >= 0.6; DEC_SHAPE sd_prior=1) is correct, compliant and safe as the default. Verdict PASS / FAIL with findings and an adopt / adopt-with-changes / reject recommendation.
context_refs:
  - git diff (src/possess/shape.py, decode.py, scripts/tune_shape.py, diagnose.py, tests/test_shape.py, test_decode.py, test_edges.py), eval/results/eval_L2-21*.json, transfer_sd_prior.json
  - .planning/envelopes/L2-21-hidden-d-real2-fable.in.md + .out.md (builder's claims), L2-20 *.out.md (real2 + compliance precedent), .planning/research/provided-vs-bench-gaps.md
  - CLAUDE.md, docs/RULES.md (forbidden: "Training, fitting or calibrating your algorithm on the ground-truth possession data or on the provided video"), ADR-001, ADR-016/017, LOOP.md (real2 co-decides: neither real2 nor real1 drops > 0.003, at least one gains, orig drop <= 0.005, bootstrap + second seed)
rules_reminder: Agents MAY read the provided match's INPUT for aggregates; NEVER open the public ground-truth CSV (third_party/mtah/evaluation/gt/*); --no-public always; no held-out runs.
owned_files: [.planning/envelopes/L2-21-qa-fable.out.md]
checks:
  - Reproduce: real2 0.5173, real1 0.6081, orig 0.6305; `--dec sd_prior=0` gives real2 0.4833; pytest passes.
  - EM correctness: convergence, initialisation, per match vs per half, which frames enter (hidden only?), NaN/no-evidence handling, the log-odds shift math (prior ratio vs training prior "pi"), LOMO integrity of "pi".
  - Gate 0.6: was it truly fixed before the provided estimate was read (builder says yes)? Random-hiding max |shift| 0.49 vs real2 0.97-1.82 — is the margin adequate; what happens near it?
  - COMPLIANCE: the algorithm now estimates a parameter (the dead-ball prior on hidden frames) from the provided match's own input at run time, unsupervised and identically on every match. Is that "fitting/calibrating on the provided video" in the organisers' sense, or ordinary inference (like the attacking-direction vote from player positions)? Give a clear argument either way and what to disclose.
  - TRANSFER RED FLAG check: provided estimate 0.869 (shift +2.27) is above every real2 match (max 0.838 true 0.83); provided hidden D share 0.724 -> 0.897; no-evidence hidden frames (18%) become 99.7% D; D runs per half hour 24.4 -> 30.0; 7.5% of frames change. Using only input aggregates, judge whether this is plausible (e.g. are the provided match's hidden blocks really mostly stoppages: player speeds, block lengths, positions near lines, player counts) or an over-extrapolation. Consider recommending a cap on the estimate/shift (e.g. at the bench's max) — but any cap must be justified from external data, not from the provided match.
constraints:
  - RAM: check >= 3 GB free before every run (python ctypes GlobalMemoryStatusEx); foreground; one run at a time; < 10 min per command.
  - Do not edit source files. Do not commit.
deliverable: output envelope <= 350 words: verdicts, compliance argument, transfer judgment, findings with file:line, commands run.
```
