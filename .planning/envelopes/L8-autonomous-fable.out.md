```yaml
id: L8-autonomous-fable
status: CHECKPOINT 12 (2026-11-17 08:00 CET = UTC+1; earlier "real clock" stamps were UTC+3) - ONE CANDIDATE PASSES THE GATE AS A BUNDLE: T141 `w_bd=1.0, w_rev=2.0`.
branch: agent/a8 = main fedadd8 (w_bt default) + opt-in flags w_bd, w_rev / gap_cp (react.py ported from agent/a7 47ebbe8), ed_still. Defaults byte-identical, pytest 206 green, never pushed.
log: .planning/research/autonomous-L8.md (rows 79-85, TRIED drafts T138-T141, "Round 7: start here")
matchlens_start: events/deadball.py, kinematics.py (T137); MatchLens has no ball-free team inference (possession.py checked), react.py is new (L5).
```
**T141.** `w_bd` = T137 (visible-ball stoppage model). `w_rev` = T89 (L5): a hidden H<->A switch costs less where the all-player flow reverses towards the new team. Baseline = default with leave-one-match-out w_bt.

d ctl2 [CI90], leave-one-match-out:

| | real2 s0-s3 | real1 | orig |
|---|---|---|---|
| bd + rev | +0.0052 / +0.0005 / +0.0068 / +0.0071, median +0.0060 | +0.0254 [+0.0168, +0.0345] | +0.0166 [+0.0093, +0.0243] |
| + ed_still=1.0 | +0.0061 / +0.0010 / +0.0071 / +0.0080 | +0.0269 | +0.0211 |
| bd + rev, transfer | +0.0055 (s0) | +0.0179 | +0.0022 |

Old truth, bd + rev: +0.0027 / +0.0012 / +0.0070 / +0.0039, +0.0210, +0.0154.
**Gate:** real2 median +0.0060, no drops, real1 and orig CI90 > 0.
**Caveats:** marginal: neighbours (w_rev 1 / 3, w_bd 0.5 / 1.5) gain on every bench but their real2 medians are +0.0041..+0.0048 (row 86); neither part passes alone; real2 CI90 > 0 on 2 of 4 seeds; w_rev constants were picked in L5 on real1.
**Provided input (no truth):** 0.68 % of frames change; w_rev alone 0.028 % (little hidden live play there), so the submission moves by what T137 moves; possessions 324 -> 326.
**QA should:** held-out split, public 5 min; `a8b_r7.py bundle [excl=1]`, `prov`.

**Diagnosis (rows 80, 83):** boundary-date oracle +0.043..+0.052; a perfect visible-frame stoppage model gains only +0.006..+0.013 on real2, so T137 alone cannot meet the real2 clause.
**Rejected:** T138, T139, T140; row 84 neutral.
