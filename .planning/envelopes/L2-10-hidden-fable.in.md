```yaml
id: L2-10-hidden-fable
role: Fable designer/builder (hard task)
goal: Catch possession changes that happen while the ball is hidden (diagnosis L2-8: hidden-block turnover recall 0.08), so tune mean beats 0.5679.
matchlens_start:
  files: [MatchLens possession.py zones (decode/team_control), our shape.py team model (MatchLens attacking_direction idea)]
  baseline: "default tune 0.5679 (m1 0.610, m2 0.564, i1 0.530), held-out 0.5584 (3 looks). Run scripts/diagnose.py for the current breakdown first."
context_refs:
  - .planning/envelopes/L2-8-segment-fable.out.md, L2-7-shape-fable.out.md (team-shape model details), L2-9-ballclean-fable.out.md
  - docs/research/swarm-findings.md (F2: possession survives ~75% of gaps; F6: relative team shape; player-anchored fill found a team change in 30/147 gaps)
  - src/possess/{decode.py, shape.py, pipeline.py, features.py, ball_fill/player_anchored.py}, scripts/{evaluate.py, diagnose.py, tune_shape.py}
owned_files: [src/possess/decode.py, src/possess/shape.py, src/possess/pipeline.py (DEC constants only), scripts/diagnose.py, scripts/tune_shape.py, tests/test_decode.py, tests/test_shape.py, .planning/envelopes/L2-10-hidden-fable.out.md]
ideas (not orders):
  - Edge evidence: team nearest the last-seen ball vs the first-seen ball after the gap (player-anchored), with the switch placed where team-shape log-odds cross.
  - A gap-level classifier (same team before/after) from shape-model log-odds trajectory, attacking direction, ball exit/entry positions; LOMO on tune.
  - Lower switch cost inside hidden blocks when shape evidence is strong.
constraints:
  - Tune ONLY on metrica1, metrica2, idsse1; LOMO for learned parts; noise gate (sigma 0.10 m and 7-12 players) for any learned model; no held-out; never public5/challenge for choices.
  - RAM: separate-command check >= 3 GB before every run; foreground; < 10 min per command; no recomputation of existing caches; grids <= 30.
  - evaluate.py must reproduce any default you set with no flags. Self-contained repo. Do not commit.
deliverable: code + tests + output envelope (<= 300 words): diagnosis numbers, change, tune LOMO per match + counts, hidden-turnover recall before/after, noise gate, reproducing command, risks.
done_when: tune > 0.5679 reproduced by evaluate.py, or a clear negative.
```
