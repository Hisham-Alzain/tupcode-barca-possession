```yaml
id: L4-definitions-review-fable
status: done (read-only; no bench, no pipeline run, no GT opened, nothing committed)
deliverable: docs/organiser-definitions-review.md (42 terms, verbatim quotes with file:line, verdicts, plan)
```

**Differences, by impact**

1. **Controlled possession** (CHALLENGE.md:55-80): DIFFERENT, high. Pipeline and external truth both count touches, deflections and uncontrolled headers. Already with `L4-control-definition-fable`.
2. **Second-half pre-kick-off** (FAQ.md:38-40: frames 110,808–113,741 are graded and all `D`): DIFFERENT, verified. Draft `2026-11-15a` writes `H` 1,455 / `A` 44 / `D` 1,435 there, six impossible possessions. No bench contains this case. Estimated +0.004 spellF1 (not measured), plus macroF1.
3. **Restart possessor** (CHALLENGE.md:66): PARTIAL. Soft other-team prior only; Smart Tagging unused; external truth may carry the old team briefly after a restart (not measured).
4. **Gates**: ADR-016's "orig drop <= 0.005" is on the old truth, which rewards what the organisers exclude. Needs an ADR before item 1 merges.
5. **TECHNICAL.md** rules section and labelled-data table: PARTIAL (SUBMISSION.md:68-71).
6. **Hosted-LLM clause** (FAQ.md:109-112) vs the agents email: UNKNOWN; needs the original email and terms.

MATCH: labels, H/A orientation, `X`, graded halves, frame alignment, row count, scorer (vendored copy byte-identical), maximal-run possession, reproducibility by inspection, data not tracked.

**Plan order**

1. `D` before each half's kick-off, located from the Smart Tagging `Kick Off (Start)` tag (input, not the constant 113,742). Benches must stay byte-identical; record as a definition-alignment ADR.
2. Control work continues; add the gate ADR, a restart-sliver count and the IDSSE aerial check to it.
3. Send `.planning/organiser-question.md` today, with the kick-off and LLM-clause questions added.
4. Rewrite TECHNICAL.md §6 in organiser terms.
5. Clean-clone regeneration, validate, upload.

**Not available:** Gmail, Drive; original emails, submission-instructions email, terms.

**One check beyond docs/code:** label counts in our own draft CSV around the kick-offs.
