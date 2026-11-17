```yaml
id: L2-15-segment-fable
role: Fable designer/builder (hard task)
goal: Fix IDSSE-style under-segmentation (real1 tune idsse1 356 predicted vs 511 true possessions; IDSSE true mean 6.7 s) without hurting Metrica, so real1 tune beats 0.5834 with orig >= 0.6060. The provided match's ball-proximity profile resembles IDSSE (Fable QA L2-14), so IDSSE gains are the ones most likely to transfer.
context_refs:
  - .planning/LOOP.md (protocol: decide on real1, transfer check, held-out usage note), ADR-005..012, envelopes L2-8..L2-14 *.out.md
  - scripts/{diagnose.py, evaluate.py (--bench real1), transfer_check.py}
  - src/possess/*
rules_reminder: Agents MAY read the provided match's INPUT under data/challenge (organisers' clarification, ADR-001; the repo CLAUDE.md agrees) for transfer checks, aggregates only; NEVER the public ground-truth CSV; never tune on the provided match.
owned_files: [src/possess/decode.py, src/possess/shape.py, src/possess/team_control.py, src/possess/pipeline.py (constants only), scripts/diagnose.py, scripts/tune_shape.py, tests/*, .planning/envelopes/L2-15-segment-fable.out.md]
hints: diagnose which IDSSE possessions are missed (length, ball visible?, contested?, who is near); DFL possession flips on short passes/receptions (definition differences vs Metrica events) — check whether per-source behaviour differs and whether a source-agnostic change exists; touch-hold max duration; switch cost vs duration prior.
constraints:
  - Tune ONLY on metrica1, metrica2, idsse1 (LOMO for learned parts; noise gate); decide on real1, report orig; transfer check (possessions per half hour on the provided match vs bench must move similarly); no held-out.
  - RAM: separate-command check >= 3 GB before every run; foreground (if the harness backgrounds a long run, run nothing else alongside); < 10 min per command; new caches only under new names; grids <= 30.
  - evaluate.py must reproduce any default you set with no extra flags. Do not commit.
deliverable: diagnosis + change + tests + output envelope (<= 300 words): real1 and orig tune per match + counts, transfer numbers, reproducing commands, risks.
done_when: real1 > 0.5834 with orig >= 0.6060 and idsse1 up, or a clear negative with the diagnosis.
```
