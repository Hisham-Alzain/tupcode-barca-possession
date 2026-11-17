```yaml
id: L4-definitions-review-fable
role: Fable reviewer (read-only audit + plan). RAM-light: no benches, no pipeline runs.
user_request (2026-11-15): "I want my system to follow the organizers' definitions in every term. Review each organiser definition mentioned in the emails, Google Drive and the repository, review it against our system, state the differences and conduct a plan to handle it on our system."
sources (organiser text; quote verbatim with file:line):
  - third_party/mtah/CHALLENGE.md, OVERVIEW.md, DATA.md, EVALUATION.md, FAQ.md, RULES.md, SUBMISSION.md, README.md, evaluation/possession_metrics.py (metric definitions in code), evaluation/run_evaluation.py, starter/baseline.py
  - Emails: docs/RULES.md (verbatim transcript of the organisers' "Clarification of the rules" emails), .planning/DECISIONS.md ADR-001, .planning/organiser-question.md (our open question, not yet answered). Gmail is not connected and Google Drive has no hackathon files: say so; list anything that would need the original emails.
  - NEVER open third_party/mtah/evaluation/gt/* or any public truth CSV.
our system: src/possess/ (pipeline.py, decode.py, deadball2.py, stoppage_ml.py, edges.py, team_control.py, shape.py, cleanup.py, epts.py), docs/TECHNICAL.md, .planning/DECISIONS.md, external truth builders src/possess/metrica_csv.py + src/possess/idsse_cache.py + scripts/bench.py / scripts/evaluate.py (what our benches call H / A / D and a possession).
goal:
  1. Inventory EVERY organiser term/definition: ball in play / dead (D), who has possession while live, change of possession (controlled; touch without control; deflection not intercepted; aerial duel without control), contested / loose ball (stays with last team), restart / starting possessor, stoppage start/end timing, X / graded frames / halves, frame & time alignment, H/A orientation, "a possession = maximal run", how GT is derived (corrected tracking, mechanical, no smoothing), metric (spellF1, IoU matching, macro F1, what is free/penalised e.g. going quiet through occlusion), submission format, and any rule constraints.
  2. For each: what our pipeline does (file:line) AND what our external truth / bench assumes. Classify: MATCH / PARTIAL / DIFFERENT / UNKNOWN, with the expected scoring impact (high/med/low).
  3. Plan: per difference, the concrete change (pipeline and/or external-truth relabelling), how to validate it on external data within our gates (ADR-016: real2 + real1 co-decide, orig drop <= 0.005, >= 3 seeds), priority order given the deadline 2026-11-17 23:59 CET. Note: the "controlled possession / deflection" difference is already being worked on by envelope L4-control-definition-fable (worktree bp-wt/a5) — include it in the table, do not redesign it.
rules: do not edit src/, do not commit, no data-loading commands (reading code/docs only). Read CLAUDE.md and docs/RULES.md first.
owned_files: [docs/organiser-definitions-review.md, .planning/envelopes/L4-definitions-review-fable.out.md]
deliverable: docs/organiser-definitions-review.md (full table + plan, any length) and out envelope <= 300 words (top differences by impact + plan order). Return the out envelope as your final message.
```
