# Provided match vs realistic bench (real1): input gaps the bench cannot see

2026-11-14. Input-only distribution comparison (ADR-001): provided input (`features_v1.npz`, `positions_v1.npz`) and the default's labels on it (`draft_2026-11-14h.csv`) against the three tune matches on real1 (`feat_*_s0_real1.npz`, `edges_*_real1_v1.npz`, `idsse_idsse1_v1.npz`). Public truth never opened; nothing tuned or chosen. Bench columns are m1r / m2r / i1r (metrica1, metrica2, idsse1). Touch rows come from `eval/results/audit_transfer_real1.json` and `transfer_hidden_stop.json`. Scripts (session scratchpad, not in the repo): `gaps.py`, `gaps2.py`, `gaps3.py`. Player counts and direction votes for the bench are IDSSE only (Metrica positions are not in the small caches).

## Ranked gaps

| # | metric | provided | bench real1 (m1r / m2r / i1r) | likely effect on spellF1 |
|---|---|---|---|---|
| 1 | Ball direction kinks (> 45 deg, > 3 m/s) per visible min | 8.1 | 47.7 / 52.9 / 62.8 | Touch detector starved: see rows 1b-1d |
| 1b | Ball acceleration p90 / p99 (m/s2) | 18 / 37 | 73 / 293, 79 / 283, 107 / 569 | Provided ball is a smoothed track: a touch is spread over several frames, the lag-2 impulse test (0.08 s) misses it |
| 1c | Speed change over 0.4 s, p90 (m/s) | 3.2 | 7.2 / 7.7 / 7.8 | Same |
| 1d | Touch onsets per min / touch-hold share of visible frames | 3.9 / 0.27 | 7.9 / 0.67, 7.7 / 0.69, 9.0 / 0.72 | ADR-007 (+0.047 real1) and ADR-013 act on about a third of the frames they act on in the bench: bench score overstates the real one; visible frames fall back to distance emission |
| 2 | Ball speed at the last visible frame before a hidden block / 0.2 s after it reappears (median m/s) | 2.1 / 12.3 | 5.4 / 5.5, 7.1 / 5.8, 5.3 / 5.1 | Real blocks start on a slow ball and end on a kick; bench blocks are random cuts. Restart frame is readable from the input, bench cannot reward it |
| 2b | A player within 1 m of the ball at reappearance | 0.60 | 0.49 / 0.38 / 0.33 | The reappearing ball is at the restart taker's feet in 60% of blocks |
| 2c | Ball reappears < 5 m from where it vanished; median displacement over the block | 0.27; 11 m | 0.07 / 0.10 / 0.13; 32 / 24 / 28 m | Blocks are stoppages, not play that moved on |
| 2d | Fast-player speed (p90) median, visible / hidden (m/s) | 3.34 / 1.57 | 2.69 / 2.59, 2.66 / 2.72, 2.92 / 2.71 | Known (L2-13 dhide, ADR-014). Hidden = mostly stopped on the provided match, independent on the bench |
| 2e | Slow runs (p90 < 2 m/s, >= 5 s): mean hidden share inside / fully visible | 0.75 / 0.11 | 0.47 / 0.29, 0.39 / 0.36, 0.42 / 0.34 | Visible stoppages are rare on the provided match (p90 < 2 on 19% of visible frames vs 34-37%): ADR-011's visible-frame gain mostly does not transfer |
| 2f | Predicted D share of block frames (median per block) | 0.77 | 0.32 / 0.17 / 0.25 (truth 0.31 / 0.40 / 0.40) | The default calls most real blocks dead; if wrong, whole possessions vanish with no bench warning |
| 3 | Hidden frames in gaps > 60 s; longest gap | 0.19; 221 s | 0.06 / 0 / 0; 71 / 59 / 59 s | 4 gaps hold 19% of hidden frames (predicted D 0.87, 15.6 players tracked): the bench never tests them |
| 3b | Hidden frames in gaps 40-60 s; gap median (>= 5 s) | 0.17; 11.6 s | 0.31 / 0.35 / 0.34; 16.8 / 16.6 / 16.6 s | Bench over-weights 40-60 s blocks and under-weights 5-10 s ones (53 of 131 real gaps, predicted D 0.29 there) |
| 3c | Hidden share per 5-min window, min / max / sd | 0.12 / 0.80 / 0.18 | 0.22-0.23 / 0.61-0.74 / 0.10-0.13 | Real hiding is bursty; whole 5-min stretches are 80% hidden |
| 3d | Visible run median; runs < 5 s | 13.8 s; 0.23 | 22.4 / 20.6 / 19.0 s; 0.15 / 0.17 / 0.14 | Shorter visible stretches: fewer touches to anchor a possession between blocks |
| 4 | Tracked players: mean visible / hidden; changes per s | 21.4 / 19.2; 0.33 | 22 / 22; 0.004 (IDSSE) | Bench noise gate drops players at random (T28, T52), never coupled to state |
| 4b | Hidden frames with < 16 players tracked (visible) | 0.14 (0.01) | 0 | Close-up camera: p90 < 2 on 81-84% of those frames. A stoppage cue the bench cannot show, and a bias risk for the p90 column (few players, unstable quantile) |
| 4c | A team with < 6 players, hidden frames | 0.13 | 0 | Shape models fall back there (no-evidence share 0.09 vs 0) |
| 5 | Home team nearer the ball (visible) / predicted H share of possession, visible / hidden | 0.56 / 0.61 / 0.51 | 0.51 / 0.51 / 0.54, 0.50 / 0.49 / 0.51, 0.51 / 0.45 / 0.50 | Provided match is lopsided; hidden-frame team model returns to 50/50. Bench matches are balanced, so the cost is invisible (hidden H/A frames = 12% of the match) |
| 6 | Visible ball outside the pitch / within 2 m of a line or outside | 0.04 / 0.09 | 0.15 / 0.20, 0.16 / 0.22, 0.20 / 0.29 | real1 keeps the dead ball visible off the pitch; the provided input mostly loses it. Line-based evidence earns more on the bench |
| 7 | Predicted direct H<->A switches placed in hidden frames; possession spells fully hidden | 0.10; 0.06 | 0.25 / 0.20 / 0.21; 0.13 / 0.13 / 0.14 (truth 0.39-0.44; 0.27-0.30) | Follows from 2f: fewer hidden possessions predicted. Right if blocks are stoppages, costly if not |
| 8 | Predicted D starts within 1 s of a hide onset | 0.22 | 0.05 / 0.04 / 0.03 | D edges already lock to block edges on the provided match (L2-18 territory) |

No gap found (inside the bench range): possessions per half hour 112 (87-113), spell median 5.6 s (5.2-8.8), share of spells < 4 s 0.36 (0.26-0.38), D runs per half hour 24.6 (25-27), D run median 19.8 s (16-21), D frames in runs > 60 s 0.31 (0.20-0.28), short gaps < 1 s 4.8 per half hour (4.0-4.6), attack-direction vote share 0.985 / 0.966 (IDSSE 1.00 / 0.97), nearest player <= 1 m 0.29 (0.28-0.41), ball > 15 m/s 0.08 (0.10-0.11), D followed by the same team 0.56 (0.47-0.52).

Bench-visible side note (not a transfer gap): after a predicted D run the same team returns in 47-52% of cases, truth 10-22%. Those D runs are 80-95% true D, so the error is the team before the stoppage (a short last-touch spell merged away), not the restart team.

## Top 3 testable hypotheses

### H1. The real ball is smoothed: touch detection needs a wider impulse window
Evidence: rows 1-1d. Speeds match the bench (frame speed p99 24.6 vs 26-28 m/s) but kinks are 6-8 times rarer and acceleration p99 is 8-15 times lower. Proximity is not the cause (player within 1 m: 0.285 vs IDSSE 0.278). Not in TRIED: T41 / T52 add noise, nothing smooths.
Test on external data only:
1. New bench variant `smooth1` in the ball-fault stage: low-pass the visible ball inside each visible run (Gaussian or Savitzky-Golay), filter width fixed by matching three input aggregates of the provided ball: acceleration p90 ~18 and p99 ~37 m/s2, kinks ~8 per visible min. No labels involved.
2. Score the default on `smooth1` with touch on and off (w_touch 0.7 / 0). Expected: the +0.047 touch gain shrinks. That number is the size of the transfer loss.
3. On `smooth1` only, scan `ball_impulse` lag (2 -> 4, 6, 8 frames) and dv_thr / cos_thr, or a curvature-over-0.4 s test. Accept if it recovers the gain on `smooth1` without losing on real1 and the noise gate. Transfer check: touch-hold share on the provided match should rise from 0.27 toward the bench's 0.67-0.72.

### H2. Hiding is tied to stoppages and the reappearing ball is the restart kick
Evidence: rows 2-2f, 6, 7. New beyond dhide (L2-13) and edge snapping (L2-18): the exit signature (12.3 m/s ball, player within 1 m in 60% of blocks, ball back within 5 m in 27%) and the fact that visible stoppages are rare.
Test on external data only:
1. Bench variant `stophide1`: replace the independent `hide_mask` by state-coupled hiding. Hide each true stoppage >= 3 s from 0.5-2 s after the stop until the first frame after the restart where the ball exceeds ~8 m/s (so the ball reappears in flight), then add random live blocks until the hidden share is 0.41. Fix the stoppage hide rate by matching input aggregates only: p90 < 2 m/s on ~19% of visible frames, slow-run hidden share ~0.75, entry / exit ball speed ~2 / ~12 m/s.
2. Score the current default and each adopted step (ADR-011, 013, 014) on `stophide1`: which gains survive when visible stoppages are rare.
3. New evidence to test there: at the exit of a block that holds predicted D, end D at the reappear frame and give the restart to the team of the player within 1 m at that frame (touch-style emission, not a rule on finished runs, so not T58). Re-test P1 / P4 restart team on `stophide1`: the L2-14 oracle (+0.002) was measured where D exits were visible.

### H3. Camera close-ups and long blackouts: player count as stoppage evidence, real gap lengths in the bench
Evidence: rows 3-4c. 14% of hidden frames have < 16 tracked players (1% of visible), the count falls to 15.6 in gaps > 60 s, and the bench has neither these gaps nor any player loss tied to state.
Test on external data only:
1. `hide_mask` draws from the provided gap-length list itself (131 values, input only) instead of the binned histogram: restores the > 60 s tail (19% of hidden frames) and the 5-10 s mass. Report bench spellF1 by gap-length bucket.
2. Player-loss degradation: inside hidden blocks longer than 20 s keep only players within a 30-40 m window for part of the block, so that mean count (19.2 hidden) and the share of frames under 16 players (0.14) match; add count flicker 0.33 changes per s. Check the default (p90 column of ADR-014 first) holds its score.
3. Only if 2 holds: add tracked-player count (and its 5 s drop) as a column of the LOMO stoppage model, trained on the degraded bench, with the noise gate.

## Limits
- Predicted-label rows describe the default's output, not truth: they show where a mistake would hide, not that one exists.
- One provided match against three bench matches; no confidence intervals. Rows 1 and 2 are far outside the bench range; rows 5 and 3d are close to it.
- real1's kink rate is partly its own injected faults (about one jump per minute), but the original bench also has 20-40 touch onsets per min against 6.5.
