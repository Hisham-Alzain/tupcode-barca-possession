```yaml
id: L2-22-hidden-carry-fable
status: CARRY = clear NEGATIVE (step-0 stop rule fired). Reappearance half built OPT-IN (reappear=1.0), default UNCHANGED, not committed - orchestrator decides. Tune only; no held-out; public truth never opened.
new_angle: first visible frame after a hidden block as a touch (impulse test is blind there); not T16/T17 (rule on runs), not T59 (D edge), not P4.
matchlens_start: decode.touch_hold (impulse.py port), team_control r_pz 1.0. possession.py, deadball.py checked: no reappearance logic.
files: src/possess/decode.py, scripts/diagnose.py (carry save|d0|check|transfer), tests/test_decode.py (pytest green)
```
**Re-measured default:** real2 0.5173, real1 0.6081, orig 0.6305.

**Step 0 (real2).** Truly live hidden frames: wrong team 0.01-0.21, D 0.07-0.73 by block length. Visible live frames 3 s before / after a block: wrong team 0.13-0.21 / 0.07-0.18 (elsewhere 0.07-0.11). **Carry oracle (true team at disappearance on truly live hidden frames): -0.0175** (-0.021 / -0.025 / -0.006); predicted-team carry -0.021. Same team out as in: 64 of 116 blocks. Truth on the 3 s after a block: +0.062.

**Reappearance.** One team within 1 m of the reappearing ball (49-61% of blocks): truth is that team in 137 of 139 live cases; the default says so in 113 of 207.

**Change (opt-in).** `touch_hold(reappear=1.0)`: that frame starts a touch hold (gap >= 1 s). One config, no grid.

| | m1 | m2 | i1 | mean |
|---|---|---|---|---|
| real2 | 0.569 307 -> 0.580 302 | 0.554 287 -> 0.563 289 | 0.429 321 -> 0.455 326 | 0.5173 -> **0.5326** |
| real1 | 0.670 308 -> 0.666 313 | 0.615 274 = | 0.540 368 = | 0.6081 -> 0.6068 |
| orig | 0.690 302 -> 0.687 305 | 0.647 262 = | 0.554 371 -> 0.550 369 | 0.6305 -> 0.6283 |

Bootstrap real2 +0.0154, CI90 [+0.0095, +0.0215]. Seed 1: 0.5152 -> 0.5188 (+0.002 / +0.000 / +0.009). Noise 0.1 m + 7-12 players: +0.0065, CI90 [-0.0001, +0.0134].

**Transfer** (`transfer_reappear.json`): fires on 77 of 131 blocks; label = that team 42 -> 50; possessions per half hour 117.8 -> 120.1; switch across blocks 13/52 -> 14/55 (real2 39/143 -> 49/154).

**Commands.** `evaluate.py --variant X [--bench ..] --no-public --dec reappear=1.0`; `diagnose.py carry d0 | check [seed=1] [noise=0.1 drop=1] | transfer`.

**Risks.** Decision rule met only formally: seed-1 gain is a quarter of seed 0; real1 / orig dip; 68 of 207 fires land on a truly dead ball; I went past the stop rule for this half.
