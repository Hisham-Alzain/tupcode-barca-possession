```yaml
id: L3-A4-qa-fable
role: Fable cold QA (correctness + compliance + robustness)
worktree: D:\ITe\year5\graduation project\The project\bp-wt\a4 (branch agent/a4, commits 2864f12, 96cef95 on top of main 5fed209). Work THERE (cd for every command). Review `git diff 5fed209..HEAD`.
goal: Decide whether `vtx=2.0` (decode.touch_vertex / vertex_team_dist: on a low-passed ball track the contact point of a touch is rebuilt from the straight incoming and outgoing ball flight, and the touch is credited to the team nearest that point) is correct, leak-free, compliant and safe as the default. Verdict PASS / FAIL, adopt / adopt with changes / reject.
context_refs:
  - .planning/envelopes/L3-common.md (rules, decision rule ADR-016), L3-A4-short-possessions.in.md + .out.md (builder's claims)
  - ADR-016..019 in .planning/DECISIONS.md (real2 bench; smooth_touch regime switch and its compliance argument: ball acc p99 < 150 m/s2, any threshold 48-204 identical on all data), docs/RULES.md
rules: every data-loading command through `uv run python scripts/ramlock.py -- ...`; ALWAYS --no-public; NEVER --split heldout; NEVER open third_party/mtah/evaluation/gt/*; provided-match INPUT aggregates only. Do not edit source files; do not commit.
owned_files: [.planning/envelopes/L3-A4-qa-fable.out.md in the a4 worktree]
checks:
  - Reproduce: default real2 0.5618, real1 0.6068, orig 0.6283; `--dec vtx=0` real2 0.5326; pytest passes; real1/orig labels byte-identical with vtx on/off (mode must be off on raw tracks).
  - Geometry: line fits for incoming/outgoing flight (window lengths, NaN gaps, hidden frames, period boundaries, degenerate/parallel lines, slow ball, very short flights); vertex far from both players; interaction with duel=2, reappear, smooth_touch peak picking, clean_ball.
  - Robustness: the builder ran seeds 0-3; run 2 further real2 seeds (in memory) and a pooled paired 2-min block bootstrap; note metrica1 loses on seeds 1-3 — per-match pooled CI. Also sensitivity: the bench smooths with Gaussian sigma 7; check sigma 3 and 5 variants if the bench allows (builder of L2-20 did this for smooth_touch) — the gain must not depend on matching the simulator's own smoothing.
  - Bench-artefact risk: is the vertex reconstruction effectively inverting the simulator's Gaussian filter (a property of the bench, not of real provider smoothing)? Argue from the method (straight-line flight extrapolation is physics, not a filter inverse) and from the sensitivity runs.
  - Compliance: no constant chosen from the provided match; radius 2 m reused; it fires only in the low-passed regime (same disclosure as ADR-017).
  - Transfer: 3.2% of frames change on the provided input, possessions/half hour 120.1 -> 116.7 — consistent with the bench direction (counts fall there too)?
deliverable: out envelope <= 300 words: verdict, seed table, sensitivity, findings with file:line, commands run. Return the same as your final message.
```
