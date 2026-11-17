```yaml
id: L2-22-hidden-carry-fable
role: Fable designer/builder (hard task)
goal: Improve WHICH TEAM is labelled during and around hidden-ball blocks on real2 (state-coupled hiding: blocks start on a slow ball near a player, end on a kick). Hypothesis H2 of .planning/research/provided-vs-bench-gaps.md — possession at disappearance (team of the player within 1 m when the ball vanishes) carries through live hidden blocks; at reappearance, the team of the kicker/nearest player at the kick confirms or flips. Beat the default (L2-21, commit d653755): real2 0.5173, real1 0.6081, orig 0.6305.
decision_rule: real2 and real1 CO-DECIDE (ADR-016): adopt only if neither drops by more than 0.003 and at least one gains; orig drop <= 0.005; paired 2-min block bootstrap CI90 lower bound > 0 on the gaining bench; a second real2 seed (in memory) agrees in sign.
step_0: diagnose team errors on real2 hidden frames and in the 3 s around block edges (wrong team share, by block length and live/dead), and an oracle (true team at disappearance carried through live hidden frames) — stop if the oracle is < +0.01.
tried_ref: .planning/TRIED.md — team_control restart_flip/backdate (ADR-005), team context (ADR-009), duel (ADR-013), smooth_touch (ADR-017). State new_angle.
matchlens_start: src/possess/team_control.py (possession carry, restart_flip, backdate — MatchLens possession.py port), decode.touch_hold (MatchLens impulse).
context_refs:
  - .planning/research/provided-vs-bench-gaps.md, envelopes L2-20/L2-21 *.out.md and QA outs
  - .planning/LOOP.md, ADR-005..018, scripts/{bench.py (real2), evaluate.py, diagnose.py}, src/possess/*
rules_reminder: Agents MAY read the provided match's INPUT under data/challenge (ADR-001) for transfer-check aggregates only; NEVER the public ground-truth CSV (third_party/mtah/evaluation/gt/*); never tune or fit on the provided match; any regime switch must be justified independently of the provided match's statistics.
owned_files: [src/possess/decode.py, src/possess/shape.py, src/possess/team_control.py, src/possess/edges.py, src/possess/pipeline.py (constants only), scripts/diagnose.py, scripts/tune_shape.py, tests/*, .planning/envelopes/L2-22-hidden-carry-fable.out.md]
constraints:
  - Tune ONLY on metrica1, metrica2, idsse1 (LOMO for anything learned; noise gate); --no-public always; NEVER --split heldout.
  - RAM: check >= 3 GB free before every run (python ctypes GlobalMemoryStatusEx); foreground; one run at a time; < 10 min per command; new caches under new names; grids <= 30.
  - evaluate.py must reproduce any default you set with no extra flags. Do not commit. Do not edit .planning files other than your out envelope.
deliverable: out envelope <= 300 words: step-0 diagnosis + oracle, change, real2/real1/orig per match + counts, bootstrap + second seed, transfer (provided possessions per half hour, team switch rate at block edges), commands, risks.
done_when: a change meeting the decision rule, or a clear negative with the diagnosis.
```
