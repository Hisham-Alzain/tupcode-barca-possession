```yaml
id: L2-21-hidden-d-real2-fable
role: Fable designer/builder (hard task)
goal: On real2 the default (L2-20, commit 99e7547) under-predicts dead ball on hidden frames — predicted hidden D share 0.55-0.66 vs true 0.71-0.83. Find and fix the cause so that real2 tune beats 0.4833, real1 does not drop by more than 0.003 (0.6081), orig not more than 0.005 (0.6305).
decision_rule: real2 and real1 CO-DECIDE (Fable QA L2-20): adopt only if neither drops by more than 0.003 and at least one gains; orig drop <= 0.005; paired bootstrap on the gaining bench (2-min blocks, CI90 lower bound > 0), and a second real2 seed (in memory, no cache) must agree in sign.
tried_ref: .planning/TRIED.md — read first; T55 (dead_fast), T59 (exit snap), T25/T37 (stoppage model), ADR-010 (ball-still rules failed to transfer). State new_angle.
matchlens_start: src/possess/stoppage_ml.py, shape.py stoppage model, edges.py; MatchLens set-piece ideas.
context_refs:
  - .planning/envelopes/L2-20-real2-fable.out.md + L2-20-qa-fable.out.md, .planning/research/provided-vs-bench-gaps.md, L2-16/L2-18 *.out.md
  - .planning/LOOP.md, ADR-005..015, scripts/{bench.py (real2), evaluate.py, diagnose.py, audit_bench.py}, src/possess/*
hints: on real2 hiding is state-coupled (blocks start on a slow ball near a player, end on the restart kick) — so "ball hidden for long + players slow/re-forming" is itself stoppage evidence the bench never showed before; check whether the stoppage model's hidden-frame calibration (trained on randomly hidden frames) is the issue (e.g. LOMO refit of the stoppage model on real2-style hidden frames of the TUNE matches); check the hidden-block length as a feature/prior; check player-count drop (close-ups) as evidence. Watch the provided match: its hidden D share is already 0.724 at L2-18 — report it; a rise must be matched by real2's own rise toward truth.
rules_reminder: Agents MAY read the provided match's INPUT under data/challenge (ADR-001) for transfer-check aggregates only; NEVER the public ground-truth CSV (third_party/mtah/evaluation/gt/*); never tune or fit on the provided match; any regime switch must be justified independently of the provided match's statistics (see L2-20 QA compliance note).
owned_files: [src/possess/decode.py, src/possess/shape.py, src/possess/edges.py, src/possess/deadball2.py, src/possess/team_control.py, src/possess/pipeline.py (constants only), scripts/diagnose.py, scripts/tune_shape.py, tests/*, .planning/envelopes/L2-21-hidden-d-real2-fable.out.md]
constraints:
  - Tune ONLY on metrica1, metrica2, idsse1 (LOMO for anything learned; noise gate); --no-public always; NEVER --split heldout (all looks used).
  - RAM: check >= 3 GB free before every run (python ctypes GlobalMemoryStatusEx); foreground; one run at a time; < 10 min per command; new caches under new names; grids <= 30.
  - evaluate.py must reproduce any default you set with no extra flags. Do not commit. Do not edit .planning files other than your out envelope.
deliverable: out envelope <= 300 words: diagnosis, change, real2/real1/orig per match + counts, hidden D share predicted vs true on real2, bootstrap + second seed, transfer numbers (provided hidden D share, possessions per half hour), commands, risks.
done_when: a change meeting the decision rule, or a clear negative with the diagnosis.
```
