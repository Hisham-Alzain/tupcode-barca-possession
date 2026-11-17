```yaml
id: L2-14-next-fable
role: Fable designer/builder (hard task)
goal: Re-diagnose the current default (ADR-005..009 + ADR-011) on the REALISTIC bench real1 and fix the biggest remaining error cause, so real1 tune beats 0.5731 without the original bench dropping more than 0.005 (0.6002).
context_refs:
  - .planning/LOOP.md (protocol incl. transfer-check amendment; decide on real1), ADR-005..011, envelopes L2-8..L2-13 *.out.md
  - scripts/{diagnose.py, evaluate.py (--bench real1), transfer_check.py, audit_bench.py, bench.py}
  - src/possess/*
rules_reminder: Agents MAY read the provided match's INPUT under data/challenge (organisers' clarification, ADR-001; CLAUDE.md agrees) for transfer checks, aggregates only; NEVER the public ground-truth CSV; never tune on the provided match.
owned_files: [src/possess/decode.py, src/possess/shape.py, src/possess/deadball2.py, src/possess/team_control.py, src/possess/pipeline.py (constants only), scripts/diagnose.py, scripts/tune_shape.py, tests/*, .planning/envelopes/L2-14-next-fable.out.md]
known_open: IDSSE under-segmentation on real1 (360/511 predicted possessions; true mean 6.7 s); restart-team errors; turnovers inside hidden blocks (recall ~0.18).
constraints:
  - Tune ONLY on metrica1, metrica2, idsse1 (LOMO for learned parts; noise gate for learned parts); decide on real1, report orig; transfer check for any new signal; no held-out.
  - RAM: separate-command check >= 3 GB before every run; foreground; < 10 min per command; new caches only under new names; grids <= 30.
  - evaluate.py must reproduce any default you set with no extra flags. Do not commit.
deliverable: diagnosis + change + tests + output envelope (<= 300 words): real1 and orig tune per match + counts, transfer-check numbers, reproducing commands, risks.
done_when: real1 > 0.5731 with orig >= 0.5952, or a clear negative with the diagnosis.
```
