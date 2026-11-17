```yaml
id: L2-8-segment-fable
status: partial   # win measured on tune; default NOT switched: wiring needs 2 lines outside owned scope
```
**Biggest cause (scripts/diagnose.py, default 0.4932):** short possessions. 588 of 1198 true possessions are < 4 s (IDSSE 293 of 511); recall 0.11 (< 2 s) and 0.29 (2-4 s) vs 0.75+ above 10 s. Turnover recall within 1 s: 0.51 ball visible, 0.08 hidden. Oracle ceilings: team on visible 0.688 (IDSSE 0.425 -> 0.714), team on hidden 0.653, stoppages 0.598. The missed short spells are one-touch deflections / tackles / clearances: the other team is nearer for only a few frames, so the distance emission never pays the switch cost.

**Fix:** `decode.touch_hold` (port of MatchLens impulse.py:impulse_verdict + possession.py constants cos 0.85, dv 1.5 m/s, SG 7/2; team level, no IDs): ball impulse within 1 m of one team => that team holds until the next touch (max 4 s). New emission `w_touch`. Hold agrees with truth on 94-99% of covered visible live frames. Thresholds rise with the ball noise measured from the data (no labels).

**Tune (LOMO shape models), w_touch=0.7:** 0.5657 (m1 0.603 322/357, m2 0.564 286/330, i1 0.530 369/511). Visible turnover recall 0.51 -> 0.70; < 2 s recall 0.11 -> 0.25. 27 configs; plateau 0.565-0.577 (r_touch 0.75 + c_switch 6 = 0.5773, not chosen: IDSSE prefers 1.0, differences ~noise).

**Noise:** ball sigma 0.05 m 0.5664; ball + players 0.10 m 0.5627; + 7-12 players 0.5279 (default without touch there: 0.4555). Without adaptive thresholds ball 0.10 m gave 0.4956.

**Reproduce:** `uv run python scripts/diagnose.py w_touch=0.7 [ball_noise=0.1 noise=0.1 drop=1]`; `uv run pytest` (80 pass).

**To make it the default (not done, pipeline.label is not mine):** in `pipeline.label` add `"ball_x": f["ball_x"], "ball_y": f["ball_y"]` to `feats`; add `w_touch=0.7` to `DEC_BASE`. Then evaluate.py reproduces with no flags. evaluate.py was not run (it reads the public window).

**Risks:** held-out not looked at. Bench noise is white; a challenge ball with interpolated or jumpy tracks is untested. Hidden-ball turnovers (recall 0.08) are now the largest remaining cause.
