```yaml
id: L6-naive-definitions-review-fable
status: done (read-only; no pipeline run, no ground truth opened, nothing uploaded, no commit)
deliverable: docs/naive-definitions-review.md
matchlens_start: n/a (review only)
```

**Verdicts (39 items):** MATCH 19 · PARTIAL 13 · MISMATCH 1 · NOT COVERED 3 · AMBIGUOUS-IN-DOC 3.

**Draft 15d against the document's printed numbers (our output only):** 199,319 rows, 0 X on graded frames; half 1 starts A at 11,185; half 2 first live frame 113,741 (document 113,742). 371 possessions, mean 9.4 s. Public window: 21 possessions against 17, about 821 D frames against about 1,465 (derived from "17 possessions, mean 14.2 s"). Reporting only, not a basis for choosing.

**Top gaps by expected score impact (my judgement, not measured):**
1. Control is too permissive: `ctl_s=1.0` against "a few seconds"; a single touch holds up to 4 s; `confirm_s` exists but is off; no pass proxy.
2. Restart team: `c_same=1.0` and `restart_flip=True` rest on "other team restarts 78–90%", measured under the old every-deflection truth.
3. Dead-ball extent and edges come from proxies.
4. Set-piece and goal tags unused (only Kick Off (Start) is read).
5. MISMATCH: the document forbids giving competition material to hosted LLMs; RULES.md relies on an email I could not see.

**Doc/code contradictions:** TECHNICAL and README say Smart Tagging is unused (`kickoff.py` reads it); TECHNICAL and the explainer say mid-gap switch placement (`gap_pos=-1.0`); TECHNICAL §6 lacks ctl_s, pre-kick-off D, reappear, vtx; touch radius 1 m against 2 m; "17 vs 17" is now 21; results table missing four steps; placeholders remain.

**Plan (all checkable on external data except tags and compliance):** bench `ctl_s` 2–3 and `confirm_s` on the controlled truth chosen beforehand; re-measure the restart share on controlled truth, then `c_same=0`, `restart_flip=False` if it falls; set-piece tags only as a soft prior (cannot be validated externally, risky this late); update the three docs; get the LLM clause confirmed in writing.

**Caveats:** the docx reads like a second-hand summary; H/A mapping and model internals not verified.
