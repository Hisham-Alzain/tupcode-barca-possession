```yaml
id: L2-16-hidden-stop-fable
role: Fable designer/builder (hard task)
goal: Detect stoppages (D) inside hidden-ball blocks better, so real1 tune beats 0.5935 with orig >= 0.6155. The L2-10 oracle puts perfect hidden-block stoppages at about +0.04.
tried_ref: .planning/TRIED.md — read it first. Do not repeat a row without a stated new angle (ball-still rules failed to transfer: ADR-010; shape stoppage model exists: ADR-011; L2-13 stoppage weights). Write `new_angle:` in your out envelope.
matchlens_start: src/possess/stoppage_ml.py (ported MatchLens DeadBallDetector), shape.py stoppage model (player-only), MatchLens set-piece ideas (restart formations, Laws 8/15/16/17). State which you start from.
context_refs:
  - .planning/LOOP.md (protocol, transfer-check amendment, decide on real1, 2026-11-14 no-repeat amendment), ADR-005..013, envelopes L2-10, L2-13, L2-15 *.out.md
  - scripts/{diagnose.py, evaluate.py (--bench real1), transfer_check.py, oracle.py}
  - src/possess/*
rules_reminder: Agents MAY read the provided match's INPUT under data/challenge (organisers' clarification, ADR-001) for transfer checks, aggregates only; NEVER the public ground-truth CSV (third_party/mtah/evaluation/gt/*); never tune or fit on the provided match.
owned_files: [src/possess/decode.py, src/possess/shape.py, src/possess/deadball2.py, src/possess/team_control.py, src/possess/pipeline.py (constants only), scripts/diagnose.py, scripts/tune_shape.py, tests/*, .planning/envelopes/L2-16-hidden-stop-fable.out.md]
ideas (not orders): collective player kinematics during a hidden block (mean speed drop, players re-forming, a lone player at a line/corner/spot = restart); duration prior for stoppages; LOMO-learned player-only stoppage evidence restricted to hidden frames; restart side from formation.
constraints:
  - Tune ONLY on metrica1, metrica2, idsse1 (LOMO for learned parts; noise gate sigma 0.05/0.10 and 7-12 players); decide on real1, report orig; transfer check on the provided match's input (D share on hidden frames, possessions per half hour must move like the bench); no held-out.
  - RAM: check >= 3 GB free before every run (python ctypes GlobalMemoryStatusEx); foreground; one run at a time; < 10 min per command; new caches only under new names; grids <= 30.
  - evaluate.py must reproduce any default you set with no extra flags. Do not commit. Do not edit docs or .planning files other than your out envelope.
deliverable: diagnosis + change + tests + output envelope (<= 300 words): real1 and orig tune per match + counts, D recall/precision on hidden frames, transfer numbers, reproducing commands, risks, new_angle.
done_when: real1 > 0.5935 with orig >= 0.6155 and a passing transfer check, or a clear negative with the diagnosis.
```
