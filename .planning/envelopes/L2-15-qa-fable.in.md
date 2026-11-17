```yaml
id: L2-15-qa-fable
role: Fable cold QA (correctness, not style)
goal: Decide whether the uncommitted L2-15 change (touch_hold duel=2 rule; ball_impulse refactor) is correct and safe to make the default. Verdict PASS / FAIL with findings.
context_refs:
  - git diff (src/possess/decode.py, src/possess/shape.py, scripts/diagnose.py, scripts/tune_shape.py, tests/test_decode.py)
  - CLAUDE.md, .planning/LOOP.md (protocol), ADR-005..012 in .planning/DECISIONS.md
  - src/possess/team_control.py (existing duel rule this ports), src/possess/pipeline.py
rules_reminder: Agents MAY read the provided match's INPUT under data/challenge (ADR-001) for aggregates; NEVER open the public ground-truth CSV (third_party/mtah/evaluation/gt/*); never tune on the provided match. No held-out runs.
owned_files: [.planning/envelopes/L2-15-qa-fable.out.md]
checks:
  - Claimed "behaviour identical" refactor into ball_impulse: verify (old configs with duel=0 reproduce prior numbers, e.g. eval_L2-14 json values, if cheap).
  - duel=2 logic: edge cases (period boundaries, dead frames, hidden ball, both teams contested at start, tie distances, home/away symmetry).
  - Any leakage: does anything learned/selected use challenge data, public5, or held-out?
  - Reproducibility: `uv run python scripts/evaluate.py --variant X --bench real1 --no-public` gives 0.5935 and the orig bench 0.6205 with no extra flags; `uv run pytest` passes.
  - Selection-bias risk (~45 candidates on real1) and the transfer finding (change near-inert on provided match): is adopting still justified under LOOP.md rules?
constraints:
  - RAM: check >= 3 GB free before every run (python ctypes GlobalMemoryStatusEx); foreground; < 10 min per command; one run at a time.
  - Do not edit source files. Do not commit.
deliverable: output envelope <= 300 words: verdict, findings with file:line, commands run.
```
