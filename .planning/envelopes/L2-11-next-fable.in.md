```yaml
id: L2-11-next-fable
role: Fable designer/builder (hard task)
goal: Re-diagnose the current default (ADR-005..009) on the tune split and fix the biggest remaining error cause, so tune mean beats 0.5845.
matchlens_start:
  files: [src/possess/stoppage_ml.py (MatchLens deadball port), MatchLens detectors/set_piece_detector.py ideas (Law 8/15/16/17 restart geometry; not vendored)]
  baseline: "default tune 0.5845 (m1 0.630, m2 0.585, i1 0.539), held-out 0.5762 (4 looks)"
context_refs:
  - scripts/diagnose.py (run it first on the current default), .planning/envelopes/L2-8..L2-10 *.out.md
  - docs/research/literature/C_restarts/survey.md (candidate fixes C1-C4: set-piece geometry as dead evidence held until the kick; line-crossing stoppage start; restart team from the Laws; player-only stoppage classifier)
  - src/possess/{deadball2.py, decode.py, shape.py, pipeline.py, stoppage_ml.py}
owned_files: [src/possess/deadball2.py, src/possess/decode.py, src/possess/shape.py, src/possess/restart.py (new), src/possess/pipeline.py (DEC/DEAD2 constants + one call if needed), scripts/diagnose.py, scripts/tune_shape.py, tests/test_deadball2.py, tests/test_decode.py, tests/test_shape.py, tests/test_restart.py (new), .planning/envelopes/L2-11-next-fable.out.md]
constraints:
  - Choose the target from diagnose.py numbers (stoppage errors, restart-team errors, or other).
  - Tune ONLY on metrica1, metrica2, idsse1; LOMO for learned parts; noise gate (sigma 0.10 m, 7-12 players) for learned parts; no held-out; never public5/challenge for choices. Beware Metrica's bench fill-line ball during stoppages (scripts/bench.py show_dead_ball): any ball-based stoppage feature must be checked on IDSSE (real ball) separately.
  - RAM: separate-command check >= 3 GB before every run; foreground; < 10 min per command; no recomputation of existing caches; grids <= 30.
  - evaluate.py must reproduce any default you set with no flags. Self-contained repo. Do not commit.
deliverable: diagnosis table + change + tests + output envelope (<= 300 words): tune LOMO per match + counts, IDSSE-only effect, noise gate if learned, reproducing command, risks.
done_when: tune > 0.5845 reproduced by evaluate.py, or a clear negative with the diagnosis.
```
