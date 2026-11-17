```yaml
id: L2-9-ballclean-fable
status: win   # default switched (decode.viterbi ball_clean=1); not committed; hidden-block hypothesis not started
matchlens_delta: MatchLens ball tracker coasting + kinematics speed gate, made offline (a teleport is a jump that comes back); stale repeats: nothing in MatchLens
```
**Built:** `src/possess/ballclean.py:clean_ball`, called in `decode.viterbi` on the ball given to `touch_hold` only (distances, `visible`, caches untouched; `pipeline.py` not edited). Removes (a) runs of <= 3 frames that leave the track faster than 35 m/s and come back, (b) exact repeats of <= 10 frames entered and left by a moving ball, (c) readings > 3 m off a 105 x 68 pitch. Holes <= 10 frames are bridged linearly, longer ones become hidden.

**Tune mean (LOMO), cleaner off -> on** (faults on the touch ball only, seed 2):

| condition | off | on |
|---|---|---|
| clean | 0.5657 | **0.5673** (m1 0.609, m2 0.567, i1 0.527) |
| teleport 1 per 10 s | 0.5570 | 0.5673 |
| teleport 1 per 2 s | 0.5278 | 0.5665 |
| stale 1 per 10 s | 0.5553 | 0.5648 |
| stale 1 per 2 s | 0.5311 | 0.5622 |
| ball noise 0.05 m | 0.5664 | 0.5644 |
| noise + teleport 1 per 2 s | 0.5294 | 0.5653 |
| noise + stale 1 per 2 s | 0.5331 | 0.5612 |
| teleport + stale, both 1 per 2 s | not run | 0.5610 |

**One measured design choice:** removing long freezes cost IDSSE 0.530 -> 0.512 (16,600 frames; they are stoppages whose exit jump is the restart touch), so freezes > 10 frames are kept. Constants were set beforehand, no grid.

**Reproduce:** `uv run python scripts/diagnose.py quick=1 [teleport=0.5] [stale=0.5] [ball_noise=0.05] [ball_clean=0]`; `uv run python scripts/evaluate.py --variant L2-9-ballclean --no-public` -> 0.5673 with no flags (`eval/results/eval_L2-9-ballclean.json`); `uv run pytest` (91 pass).

**Risks:** faults are injected into the touch ball only; in real data they also corrupt the distances and the stoppage detector, which stay uncleaned. Long stale blocks (challenge x = -51.8 m) are kept by design. The off-pitch rule assumes a centred 105 x 68 pitch. IDSSE alone is -0.003 clean and -0.010 under noise 0.05. Held-out and public window not looked at. One run was auto-backgrounded by the harness at 180 s; nothing ran beside it.

**Not done:** hidden-block turnovers (recall 0.08).

## v2 after QA (Fable QA of 4784917: adopt after fixes) — not committed

**Fixes**
1. Real kicks removed (MEDIUM): a teleport run is now removed only if every frame lies >= `dev_min` (1.5 m) off the line between the frames before and after it AND no player is within `r_touch` (`clean_ball(..., d_near, r_touch, fps)`, passed by `decode.viterbi`). Clean data: removed alive frames idsse1 243 -> 58, all of them short stale runs.
2. Jump followed by hidden frames (LOW): the edge rule now needs <= 3 frames, >= 1.5 m beyond reach and no player near. A genuine jump of more than that with nobody near, hidden within 3 frames, is still removed.
3. Off pitch (LOW): `fps` passed through. The rule does remove alive frames: metrica1 46 of 1031 (> 3 m out, none with a player within 1 m; 21 at 4 m, 9 at 5 m), metrica2 and idsse1 none. Kept at 3 m.
4. Harness: `diagnose.py` takes `tp_lo/tp_hi` (jump size), `st_lo/st_hi` (freeze length) and recomputes the nearest-player distances on faulted frames from the faulted ball (`real_d=0` = v1 harness).

**Clean tune (LOMO):** off 0.5657 (0.6032 / 0.5643 / 0.5295) -> v2 **0.5680** (m1 0.6103, m2 0.5643, i1 0.5295). idsse1 equals the baseline exactly (the "0.530" is 0.5295). `evaluate.py --variant L2-9-ballclean-v2 --no-public` -> 0.5680, no flags.

**Faults, tune mean, cleaner off -> v2** (distances faulted too, seed 2)

| condition | off | v2 |
|---|---|---|
| teleport 5-10 m, 1 per 10 s | 0.5566 | 0.5697 |
| teleport 5-10 m, 1 per 2 s | 0.5229 | 0.5605 |
| jump 1.4-5 m, 1 per 2 s | 0.5270 | 0.5532 |
| jump 0.5-1.4 m, 1 per 2 s | 0.5347 | 0.5371 (not handled) |
| stale 3-8 frames, 1 per 10 s | 0.5492 | 0.5632 |
| stale 3-8 frames, 1 per 2 s | 0.5261 | 0.5569 |
| stale 11-50 frames, 1 per 10 s | 0.5359 | 0.5421 |
| stale 11-50 frames, 1 per 2 s | 0.4473 | 0.4668 |
| teleport + stale, 1 per 2 s each | 0.4967 | 0.5460 |
| noise 0.05 m | 0.5664 | 0.5671 |
| noise + teleport 1 per 2 s | 0.5248 | 0.5635 |

**Parameters chosen by these runs (8 configs):** `dev_min` 2.0 -> 1.5 (jump 1.4-5 m: 0.5477 -> 0.5532; clean 0.5659 -> 0.5680, idsse1 equal; 1.0 gave 0.5576 but idsse1 clean 0.529, rejected). `stale_max` 10 -> 50 frames, bridged (long 1 per 10 s 0.5361 -> 0.5421, 1 per 2 s 0.4475 -> 0.4668; clean equal). Freezes > 2 s stay untouched (v1 measurement: removing all cost idsse1 0.530 -> 0.512).

**Risks:** what is left after cleaning is mostly the distance emission and stoppage detector reading the faulted ball (long freezes: 0.4668 vs 0.5680 clean) — the touch ball alone cannot fix it; next step is to clean the ball before `features`/distances. Jumps under 1.4 m are not detected. `stale_max` 50 matches the top of the injected range. On metrica1 the cleaned touch output differs on 146 alive frames (raw right on 72, cleaned on 52; 45 of them are the off-pitch frames). `uv run pytest`: 96 pass.
