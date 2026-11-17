```yaml
id: L2-18-qa-fable
verdict: PASS (safe as default; gain is marginal, two low-severity notes)
```
**Reproduced.** Default real1 0.6081 (0.670 / 0.615 / 0.540), orig 0.6305; `--dec ed_exit=0` real1 0.6061, orig 0.6283. `uv run pytest`: 140 pass. New `viterbi` with edges off == HEAD `viterbi` on all 199,320 provided-match frames.

**edges.py logic: correct.** Fuzz, 3000 random cases (NaN p90, period breaks, X, exit+entry, +-4 s): 0 failures; run count and order kept, X untouched, no move across a period or past a neighbour's middle (`edges.py:122-131`). NaN p90 skips (`:165`). Hidden mask = the decoder's `feats["visible"]` (`decode.py:356`).

**Provided match gets the snap** (`pipeline.py:76-78`): 36 moves of 83 D runs >= 3 s, 487 runs before and after, p90 NaN 0.03%. The silent skip only applies without `pos`, where DEC_BASE has no `ed_exit`.

**Findings.**
1. Low, `edges.py:127,131`: only the edge frame must be hidden; the moved span is not. Provided match: 138 of 1107 changed frames have a visible ball (7 moves).
2. Low, `edges.py:12-13`, `shape.py:492-493`: 1.0 / 2.0 equal `stoppage_ml.py:97,107`, but those are a stationary-ball duration and a displacement window, an analogy. "Not tuned" is overstated: the grid ran ed_w {1, 2} x ed_win {2, 4}. Not cherry-picked: all exit-only cells 0.6075-0.6084, the chosen one is not the best.
3. Note, transfer: 32 moves later, 4 earlier; +1040 D frames, -67; hidden D share 0.711 -> 0.724 confirmed. Moved exits median +1.36 s (builder's +1.08 s is another statistic). Top of the bench range, same direction: consistent, not a red flag, but bench exits already end late on m1 / i1 (+0.40 / +0.16 s).

**Leakage: none.** Nothing fitted; diagnose reads challenge input caches only (`diagnose.py:340-342, 608-610`), no public truth.

**Selection bias.** CI90 lower +0.0002, gain m1-heavy: within noise. LOOP rule met; adopting is defensible, expect ~0.

**Commands.** `evaluate.py --variant L2-18-qa[-off] [--bench real1] --no-public [--dec ed_exit=0]` (wrote 4 `eval/results/eval_L2-18-qa*.json`); scratchpad fuzz script. RAM >= 4.98 GB free before each.
