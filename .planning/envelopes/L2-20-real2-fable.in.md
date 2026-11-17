```yaml
id: L2-20-real2-fable
role: Fable designer/builder (hard task)
goal: (A) Build a third bench variant `real2` that reproduces the provided match's INPUT conditions found in .planning/research/provided-vs-bench-gaps.md (smoothed ball, state-coupled hiding: blocks start on a slow ball near a player and end on a kick, long-gap tail, low player counts in hidden frames), using ONLY input aggregates of the provided match (never any truth). (B) Re-score the current default and the key past switches on real2 (touch rules ADR-007/013 on/off, duel, L2-16 dead_fast, L2-18 ed_exit) and report which ones still help. (C) Try ONE targeted fix for the largest real2-specific loss (e.g. H2: possession carried through a hidden block by the team near the ball at disappearance; restart taker at reappearance; or H1: wider impulse lag on a smoothed ball).
tried_ref: .planning/TRIED.md (read first; state new_angle). Precedent: L2-12 built real1 from the provided match's gap histogram (input aggregates) — same method, more dimensions.
matchlens_start: scripts/bench.py VARIANTS (real1), src/possess/decode.touch_hold (MatchLens impulse port), team_control (possession carry rules).
context_refs:
  - .planning/research/provided-vs-bench-gaps.md (THE input for this task), .planning/envelopes/L2-12-audit-fable.out.md, L2-15/16/18 *.out.md
  - .planning/LOOP.md (protocol), ADR-005..015
  - scripts/{bench.py, evaluate.py, diagnose.py, transfer_check.py, audit_bench.py}, src/possess/*
rules_reminder: Agents MAY read the provided match's INPUT under data/challenge (organisers' clarification, ADR-001) for AGGREGATES used to design a degradation; NEVER the public ground-truth CSV (third_party/mtah/evaluation/gt/*), never any label of the provided match, never tune or fit the ALGORITHM on the provided match. Write in the out envelope one paragraph on why matching bench degradation to input aggregates is evaluation design, not calibration of the algorithm (same as L2-12).
owned_files: [scripts/bench.py (add VARIANTS["real2"], keep real1 byte-identical), scripts/evaluate.py (only to accept --bench real2 if not automatic), scripts/audit_bench.py, src/possess/decode.py, src/possess/team_control.py, src/possess/shape.py, src/possess/pipeline.py (constants only), scripts/diagnose.py, scripts/tune_shape.py, tests/*, .planning/envelopes/L2-20-real2-fable.out.md]
decision rule for (C): a change is adoptable only if real2 tune improves AND real1 tune does not drop by more than 0.003 AND orig does not drop by more than 0.005; LOMO for anything learned; noise gate; transfer check (possessions per half hour, hidden D share on the provided match). real1 remains the official deciding bench unless the out envelope argues (with the audit numbers) that real2 should replace it — the orchestrator decides that, not you.
constraints:
  - Tune ONLY on metrica1, metrica2, idsse1; no held-out (all looks used; never run --split heldout); --no-public always.
  - Audit real2 against the provided input with audit_bench / the gap-analysis metrics: each targeted gap must be within or near the provided value; report the table.
  - RAM: check >= 3 GB free before every run (python ctypes GlobalMemoryStatusEx); foreground; one run at a time; < 10 min per command; new caches only under new names; grids <= 30.
  - evaluate.py must reproduce any default you set with no extra flags. Do not commit. Do not edit .planning files other than your out envelope.
deliverable: out envelope (<= 300 words): real2 audit table, default + switches on real2/real1/orig, the fix and its numbers, transfer numbers, reproducing commands, risks.
done_when: real2 built and audited + re-score table; fix adopted-candidate or a clear negative.
```
