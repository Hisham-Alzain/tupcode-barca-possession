```yaml
id: L2-15-qa-fable
verdict: PASS (correct, safe as default; not a claimed win)
held_out: not run. public ground truth: not opened (all runs --no-public). No source edits, no commit.
side_effect: evaluate.py wrote eval/results/eval_L2-15_qa_real1.json, eval_L2-14_qa_real1.json, eval_L2-15_qa_orig.json (outside owned_files; keep or delete)
```
**Reproduced (bit-identical to the builder's JSON, per match and mean).**
- default, real1: 0.5934780677030255 (0.659 / 0.598 / 0.523)
- default, orig: 0.6204894941281591
- `--dec duel=0`, real1: 0.583372418121196 = eval_L2-14_real1.json, so the refactor is inert
- `uv run pytest`: 111 pass

**Refactor.** `decode.py:88` `ball_impulse` is the old block moved verbatim. Old (HEAD) vs new `touch_hold` on 1800 random cases (NaN gaps, NaN distances, ties, lag 1-3, duel 0/1): 0 mismatches.

**duel=2 (`decode.py:148,156`).** Correct. Contested touch skips the reset and falls to the countdown, so the hold is never renewed. Hidden frame still clears. Contested at start stays 0. Ties irrelevant. Home/away swap gives exact negation (300 random cases). duel=2 output equals duel=0 wherever duel=0 is non-zero: it only adds hold frames, so downside is bounded. Halves are decoded separately; dead frames are overridden by hard stoppages.

**Leakage.** None. Nothing fitted; `diagnose.py segment` uses tune truth only; transfer uses challenge input only.

**Findings (none blocking).**
1. No paired bootstrap for L2-15. +0.010 after ~45 candidates is about the size of L2-14's borderline gain. Supporting it: 6 of 6 matches up over both benches, a three-valued rule with a MatchLens prior, not a tuned constant.
2. Transfer: contested touches 0.16 / min on the provided match vs 1.6-3.0 on the bench; possessions unchanged. Expect 0 on the real match. Under LOOP.md this is adoptable as default but does not justify the last held-out look on its own; bundle it with the next change.
3. Builder's own risk stands: with 0.1-0.2 m ball noise duel=2 is worse than duel=0. Provided ball sigma is 0.000, so it does not apply here.
4. `diagnose.py:204` calls `ball_impulse` without `touch_hold`'s length guard (n < 7 breaks `np.convolve`); diagnostics only.

**Commands.** `uv run pytest -q`; `evaluate.py --variant L2-15_qa --bench real1 --no-public`; `--variant L2-14_qa --bench real1 --no-public --dec duel=0`; `--variant L2-15_qa_orig --no-public`; RAM 3.9-4.7 GB free before each.
