# BACKLOG — parked, not built

- HSMM duration models (PLAN step 8, only if time)
- Bench: include idsse_cache.VERSION in derived cache names (case/dist/cen/pred) so a parser change invalidates them (QA L3).
- Bench: tests for get_case IDSSE branch and ball-N gaps inside a half (QA L3).
- Bench realism: hide ~28% of the ball on true dead-ball frames too (challenge keeps the ball on ~72% of D frames; IDSSE 100%, Metrica fill 100%).
- P5 re-QA LOW: fall back to DEC_BASE (not DEC with zero shape evidence) for a half whose direction vote share < 0.65, with a CLI warning; weight direction votes by player count; GK rule can drop a deep defender when the keeper is untracked; pass fps through pipeline; bench.py --pipeline possess ignores positions (use evaluate.py); CLI recomputes features vs evaluate's cache (verify identical CSV before submission).
