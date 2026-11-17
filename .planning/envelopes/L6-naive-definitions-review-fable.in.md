```yaml
id: L6-naive-definitions-review-fable
role: Fable NAIVE reviewer (cold, read-only on code; no prior context on purpose)
user_request (2026-11-16): "spawn an agent that is a naive reviewer that will review our system definitions against this document, which is the Barca hackathon definitions: C:\Users\ASUS\Downloads\Barcalona Hackathon.docx"
stance: You know nothing about this project except what you read. Do NOT read earlier reviews first (docs/organiser-definitions-review.md, .planning/DECISIONS.md, TRIED.md, research notes) - form your own verdict from the document and the code. You may read them only AFTER your verdict is written, to add a short "agrees / disagrees with earlier review" section.
inputs:
  - reference: C:\Users\ASUS\Downloads\Barcalona Hackathon.docx (read locally, e.g. `uv run --with python-docx python -c ...` or unzip word/document.xml). Never upload it or any project file anywhere.
  - our system: src/possess/ (pipeline.py, decode.py, shape.py, deadball2.py, ballclean.py, ctl_truth.py, kickoff.py, edges.py, features.py and what they import), docs/TECHNICAL.md, docs/reading-possession.html (explainer), README.md. Current submission draft: submissions/draft_2026-11-15d.csv (you may count labels; it is our output, not truth).
task:
  1. Extract EVERY definition, rule, label meaning and edge case in the document (verbatim quote + location). Include H/A/D/X meanings, possession start/end, control vs touch, deflections, duels, dead ball, restarts, halves/periods, frame indexing, scoring, any timing tolerances.
  2. For each, find what our code actually does (file:line, flag/default value), from the code itself, not from docs. Verdict: MATCH / PARTIAL / MISMATCH / NOT COVERED / AMBIGUOUS-IN-DOC. Explain like a football-literate outsider: what a frame would be labelled by us vs by the document in a concrete example.
  3. Also check our docs (TECHNICAL.md, explainer) say the same as the code; list contradictions.
  4. Plan: for each non-MATCH, the smallest change that would align us, its risk, and whether it can be validated on external data (data/external) without tuning on the challenge match.
rules: read CLAUDE.md and docs/RULES.md first. Read-only on src/, data/, submissions/. NEVER open third_party/mtah/evaluation/gt/* or any truth CSV for the provided match; never run --split heldout; never write under data/. No pipeline runs needed; if you must load data use `uv run python scripts/ramlock.py -- <cmd>` (exit 75 = busy: wait, retry). Do not commit, do not push.
owned_files: [docs/naive-definitions-review.md, .planning/envelopes/L6-naive-definitions-review-fable.out.md]
deliverable: docs/naive-definitions-review.md (full table + plan) and an out envelope <= 300 words (counts per verdict, top mismatches by expected score impact, doc/code contradictions, plan). Return the out envelope as your final message.
```
