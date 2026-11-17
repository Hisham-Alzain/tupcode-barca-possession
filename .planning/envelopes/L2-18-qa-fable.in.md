```yaml
id: L2-18-qa-fable
role: Fable cold QA (correctness, not style)
goal: Decide whether the uncommitted L2-18 change (src/possess/edges.py post-pass snapping hidden stoppage EXITS (>= 3 s, ending on a hidden frame) to the largest p90 player-speed step within +-2 s; DEC_SHAPE ed_exit=1, ed_w=1.0, ed_win=2.0; viterbi per-frame stoppage-cost options off by default; FEAT_V 4) is correct, leak-free and safe to make the default. Verdict PASS / FAIL with findings.
context_refs:
  - git diff + new files (src/possess/edges.py, decode.py, shape.py, scripts/diagnose.py, scripts/tune_shape.py, tests/test_edges.py, tests/test_decode.py, tests/test_shape.py), eval/results/tune_edges.json, transfer_edges*.json
  - .planning/envelopes/L2-18-stop-edges-fable.in.md and .out.md (builder's claims to verify)
  - CLAUDE.md, .planning/LOOP.md, .planning/TRIED.md, ADR-005..014
rules_reminder: Agents MAY read the provided match's INPUT under data/challenge (ADR-001) for aggregates; NEVER open the public ground-truth CSV (third_party/mtah/evaluation/gt/*); run evaluate.py with --no-public; never tune on the provided match. No held-out runs (all looks used).
owned_files: [.planning/envelopes/L2-18-qa-fable.out.md]
checks:
  - Default reproduces real1 0.6081 / orig 0.6305 with no extra flags; `--dec ed_exit=0` reproduces 0.6061; viterbi new options off are byte-identical; `uv run pytest` passes.
  - edges.py logic: half boundaries, hidden-frame definition (same as decoder's), stoppages touching period start/end, overlapping windows, snap never crossing into another stoppage or past the next possession spell, NaN p90, behaviour when edge_p90 missing (silently skipped — is that safe in the CLI path on the provided match? confirm the provided match actually gets the snap).
  - The ±2 s / 1 s values: are they truly MatchLens anchors (cite where) and not picked on the grid?
  - Leakage: nothing fitted; diagnose edges uses tune truth only.
  - Selection bias: ~120 candidates, CI90 lower bound +0.0002, gain mostly metrica1 — is adopting justified under LOOP.md (real1 improves, orig not down)? Give your judgment.
  - Transfer: provided-match hidden D share 0.711 -> 0.724 on top of L2-16's rise; exits move +1.08 s there (bench +0.48..+1.19). Red flag or consistent?
constraints:
  - RAM: check >= 3 GB free before every run (python ctypes GlobalMemoryStatusEx); foreground; one run at a time; < 10 min per command. Another RAM-light agent may be loading small npz files; that's fine.
  - Do not edit source files. Do not commit.
deliverable: output envelope <= 300 words: verdict, findings with file:line, commands run.
```
