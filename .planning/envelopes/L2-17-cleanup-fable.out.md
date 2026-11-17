```yaml
id: L2-17-cleanup-fable
status: NEGATIVE. No config of the fixed grid beats the default; default unchanged (real1 0.6061, orig 0.6283 re-measured). Tune only, no held-out, public truth never opened, nothing committed.
new_angle: context-dependent minimum durations on finished runs (flank, length, ball hidden, touch-backed), not a uniform switch cost or frame hysteresis.
matchlens_start: no segment-level merge pass in MatchLens (ported files checked - stoppage_ml.py, team_control.py, deadball2.py, decode.touch_hold); its frame-level durations (0.2 / 0.5 / 0.8 / 2 s) anchor the grid.
matchlens_delta: cleanup.py R1-R4 + protect, opt-in, off by default.
files: src/possess/cleanup.py (new), decode.py (decode_halves hook, touch_frames), scripts/diagnose.py (`cleanup save|step0|grid|check|transfer`), tests/test_cleanup.py (14; 130 pass)
```
**Step 0 (real1, 3 matches; `eval/results/cleanup_step0_real1.json`).** The decoder leaves almost nothing short.

| predicted run | n | matched @0.5 |
|---|---|---|
| flicker < 0.4 s | 13 | 0.15 (12 touch-backed) |
| flicker 0.4-0.8 / 0.8-1.2 / 1.2-2 s | 15 / 25 / 48 | 0.53 / 0.48 / 0.58 |
| D between same team < 5 s | 14 | 0.64-1.0 true D |
| island < 5 s | 12 | 0.12-0.33 |
| A-h-D 3-5 s | 26 | 0.35 (only bucket >= 15 runs under 0.4; outside R4's range) |
| A-h-D 1.2-2 s | 22 | 0.41 |

**Grid (`eval/results/tune_cleanup_real1.json`, 19 configs + off; G empty, no rule's plateau pick beats off).** R1 0.4 / 0.8 / 1.2 / 2 s: 0.6047 / 0.5939 / 0.5861 / 0.5550. R2 1-5 s: 0.6061 .. 0.6017. R3: best 0.6064 (2 s). R4: best 0.6065 (0.8 s; metrica2 loses, gate 4 fails). R1 0.4 + protect 0.6067. Raw-best combination (R1 0.4 + R3 2 + R4 0.8 + protect): real1 0.6070, orig 0.6293, 2 runs relabelled, paired bootstrap +0.0009 CI90 [+0.0002, +0.0018]: below any meaningful size after ~70 candidates. Counts 308/274/368 -> 306/274/367 of 357/330/511. Noise gate not run (negative).

**Transfer census (`transfer_cleanup.json`, input aggregates).** Provided match equally clean: runs eligible per half hour, flicker < 0.4 s 0.28 (bench 0.3-1.9), bridge < 2 s 0.28, island < 2 s 0.28, sliver < 0.8 s 1.1 (bench 0.3-0.6). Provided D runs / half hour 24.6, median 19.8 s (L2-16).

**Diagnosis.** Li & Link's gain needs a jittery classifier; the Viterbi costs already remove it. Remaining short runs are matched about half the time: break-even.

**Commands.** `diagnose.py cleanup save bench=real1` (and `bench=orig`), `cleanup step0`, `cleanup grid`, `cleanup check ...`, `cleanup transfer ...`.

**Risks.** Bootstrap is my block variant (predictions attached to true spells). Larger weak buckets (islands >= 5 s: 53, 0.49) need evidence, not duration.
