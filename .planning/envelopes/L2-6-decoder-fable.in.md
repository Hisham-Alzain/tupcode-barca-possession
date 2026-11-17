```yaml
id: L2-6-decoder-fable
role: Fable designer/builder (hard task: diagnose a failed idea, design a better one)
goal: Make whole-half decoding (plan P3) beat the frame-by-frame rule on the tune split, or show convincingly why it cannot with our evidence.
matchlens_start:
  files: [src/possess/stoppage_ml.py (ported MatchLens dead-ball), src/possess/team_control.py (team-level Vidal-Codina zones)]
  baseline: "tune mean spellF1 0.3722 (Metrica G1 0.414, G2 0.372, IDSSE 1 0.331) with the frame rule; decode.py best 0.3597 (eval/results/tune_decoder.json)"
context_refs:
  - .planning/LOOP.md (protocol, all iterations)
  - docs/possession-state-machine.html (current logic and failure modes)
  - docs/research/possession-research-plan.md and docs/research/literature/B_segmentation/survey.md
  - src/possess/{decode.py, team_control.py, pipeline.py, deadball2.py, features.py}
  - scripts/{evaluate.py, tune_decoder.py, bench.py (explain(): error breakdown by cause), oracle.py}
owned_files: [src/possess/decode.py, tests/test_decode.py, scripts/tune_decoder.py, .planning/envelopes/L2-6-decoder-fable.out.md]
hints_to_consider (not orders):
  - The frame rule wins mainly through k-frame retention + back-dating + restart flip. The decoder's emission is a flat +-w_ctrl per control frame and ignores retention strength, distance margin, and how long a team has had control; possession counts are too low (244 vs 511 on IDSSE), so it under-switches.
  - Possible directions: richer emissions (soft margin d_other - d_own, run length of control), duration-dependent (semi-Markov) switch costs, separate costs for switches inside ball gaps, using the frame rule's output as one more emission, per-match diagnosis with bench.explain().
constraints:
  - Tune ONLY on the tune split (metrica1, metrica2, idsse1). Never on the public 5-minute window or the challenge data; do not run scripts/bench.py --split heldout.
  - Memory: the laptop has ~4 GB free at best. Before every run check free RAM (powershell -NoProfile -Command "[math]::Round((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1MB,1)"); if < 3 GB, wait. Run everything in the FOREGROUND (background jobs get killed), each command under 10 minutes; keep grids small (<= 30 configs) and load features once per process (as tune_decoder.py does). Feature caches already exist in data/external/_bench/ (feat_*_s0_v1.npz); do not recompute them.
  - Run python as: uv run python ... from the repo root. Tests: uv run pytest.
  - Repo stays self-contained; do not touch files outside owned_files; do not change pipeline defaults (the orchestrator does that after QA). Do not commit.
deliverable: code + tests + output envelope (<= 300 words) with: diagnosis of why v1 lost (numbers), what you changed, tune mean and per-match spellF1 + possession counts for the best config, the exact evaluate.py command that reproduces it (e.g. uv run python scripts/evaluate.py --variant p3v2 --decoder viterbi --dec ...), and risks.
done_when: tune mean > 0.3722 with the reproducing command, OR a clear negative result with evidence.
budget: one session of focused work.
```
