```yaml
id: L2-7-shape-fable
role: Fable designer/builder (hard task)
goal: Add ball-free evidence for which team has the ball while the ball is hidden (plan P5), as an extra emission in the Viterbi decoder, and/or fix the decoder's under-segmentation (IDSSE 243 predicted vs 511 true), so the tune mean beats 0.3976.
matchlens_start:
  files: [MatchLens attacking_direction.py (per-period attacking direction from GK x; ours: epts.AtdMatch._estimate_sides and the goalkeeper guess), MatchLens possession.py zones (decode.py emissions)]
  baseline: "tune 0.3976 (m1 0.452, m2 0.390, i1 0.351), held-out 0.3625, default pipeline (pipeline.DECODER / DEC), ADR-005"
context_refs:
  - .planning/LOOP.md (protocol; held-out looks: 1 — you may NOT look at held-out)
  - docs/research/swarm-findings.md (F2, F6, F7: relative team shape, count-robust features)
  - docs/research/literature/B_segmentation/survey.md (Ballradar, Perše, PathCRF)
  - src/possess/{decode.py, pipeline.py, features.py, team_control.py}, scripts/{evaluate.py, bench.py, tune_decoder.py, oracle.py}
owned_files: [src/possess/shape.py (new), src/possess/decode.py, src/possess/features.py, tests/test_shape.py (new), tests/test_decode.py, scripts/tune_shape.py (new), .planning/envelopes/L2-7-shape-fable.out.md]
ideas (not orders):
  - Team-shape features from player positions only, relative and count-normalised: home minus away centroid x signed by attacking direction, each team's x/y spread and convex-hull area, defensive-line heights, which team is nearer the LAST seen ball and moving toward it. Train a small model (logistic regression / gradient boosting from scikit-learn is NOT installed: prefer numpy logistic regression or add a pinned dependency to pyproject.toml + uv.lock only if needed) on the TUNE matches only, with leave-one-match-out to estimate generalisation; its log-odds become an emission on hidden frames (weight w_shape).
  - Or address under-switching directly: per-source analysis of why IDSSE true possessions are short (6.7 s) and the decoder misses them (bench.explain()).
constraints:
  - Tune/train ONLY on the tune split (metrica1, metrica2, idsse1). Never on the public 5-minute window or challenge data; do not run --split heldout.
  - A learned model must not see truth from the match it is evaluated on when you report numbers: report leave-one-match-out tune scores as the headline.
  - New per-frame features need one pass over each tune match: features.compute streams frames; cache new arrays separately (e.g. data/external/_bench/shape_<name>_s0_v1.npz) — do not rewrite existing feat_* caches. Challenge features for public5 are cached in data/challenge/_runs/; extend with a separate shape cache file if needed.
  - Memory: check free RAM >= 3 GB before every run (powershell -NoProfile -Command "[math]::Round((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1MB,1)" as a SEPARATE step, not chained). Foreground only, each command < 10 minutes, grids <= 30 configs.
  - `uv run python ...`, `uv run pytest`. Repo stays self-contained. Do not change pipeline defaults; do not commit.
deliverable: code + tests + output envelope (<= 300 words): approach, leave-one-match-out tune numbers per match + possession counts, the evaluate.py command reproducing the best config, risks.
done_when: tune mean (leave-one-match-out for learned parts) > 0.3976 with the reproducing command, or a clear negative result with evidence.
```
