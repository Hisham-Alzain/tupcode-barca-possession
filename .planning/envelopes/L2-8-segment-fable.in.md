```yaml
id: L2-8-segment-fable
role: Fable designer/builder (hard task)
goal: Find the biggest remaining error cause of the current default pipeline on the tune split and fix it. Prime suspect is under-segmentation (predicted vs true possessions 318/357, 287/330, 297/511): short true possessions merged into neighbours, especially on IDSSE (true mean 6.7 s). Target: tune mean (leave-one-match-out for learned parts) > 0.4932.
matchlens_start:
  files: [MatchLens deadball.py (ported: src/possess/stoppage_ml.py), MatchLens possession.py zones (team_control.py, decode.py)]
  baseline: "tune 0.4932 (m1 0.573, m2 0.482, i1 0.425), held-out 0.4789, default = decoder v2 + team shape (ADR-005, ADR-006)"
context_refs:
  - .planning/LOOP.md (protocol; held-out looks: 2 — you may NOT look at held-out)
  - docs/research/possession-research-plan.md (P4 restart team from the Laws, P6 set-piece / line-crossing stoppage start, P7 per-source radius), docs/research/literature/C_restarts/survey.md
  - src/possess/{pipeline.py, decode.py, shape.py, deadball2.py, team_control.py, features.py}
  - scripts/{evaluate.py, bench.py (explain(): miss/spurious causes), oracle.py, tune_shape.py, tune_decoder.py}
owned_files: [src/possess/decode.py, src/possess/shape.py, src/possess/deadball2.py, src/possess/pipeline.py (DEC/DEAD2 constants only), scripts/diagnose.py (new), scripts/tune_shape.py, tests/test_decode.py, tests/test_shape.py, tests/test_deadball2.py, .planning/envelopes/L2-8-segment-fable.out.md]
step_1_diagnose: Write scripts/diagnose.py that runs the CURRENT default on the tune matches (same path as evaluate.py, LOMO shape models) and prints (a) bench.explain() miss/spurious causes, (b) oracle-style ceilings for the current default (perfect stoppages / perfect team on visible / on hidden), (c) true-vs-predicted possession length histograms. Pick the biggest cause from the numbers, not from the suspect above.
step_2_fix: One hypothesis at a time; smallest change; small grids (<= 30 configs) chosen on tune only.
constraints:
  - Tune ONLY on metrica1, metrica2, idsse1. Never on public 5-minute window or challenge data; do not run --split heldout.
  - Learned parts reported leave-one-match-out. Any new learned model must pass the noise gate (sigma 0.10 m position noise must not drop it below the current default 0.4932 minus 0.005; and 7-12 players dropout reported).
  - Memory: check free RAM >= 3 GB as a SEPARATE command before every run (powershell -NoProfile -Command "[math]::Round((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1MB,1)"; if blocked, grep MemAvailable /proc/meminfo). Foreground only, each command < 10 minutes. Never recompute existing caches.
  - uv run python / uv run pytest. Repo stays self-contained. Do not commit. Do not change defaults outside DEC/DEAD2 constants; if you change a default, also make evaluate.py reproduce it with no flags.
deliverable: diagnosis table + fix + tests + output envelope (<= 300 words): biggest cause with numbers, what changed, tune LOMO per match + possession counts, noise-gate numbers if learned, reproducing command, risks.
done_when: tune > 0.4932 reproduced by evaluate.py, or a clear negative with the diagnosis.
```
