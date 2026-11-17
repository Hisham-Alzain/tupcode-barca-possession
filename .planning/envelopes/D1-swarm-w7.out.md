**Window and what happens**
150 s second-half slice (750 rows, 5 Hz). Ball missing on 46% of rows, in 5 blocks of 6–25 s. Play is mostly build-ups, long balls, goal kicks and throw-ins, with Athletic pressing high in Barcelona's third.

**Findings**
1. Missing-ball blocks are slow: team centroid speed averages 0.76 m/s (H) and 0.84 m/s (A) with ball missing (medians 0.35/0.46), versus 1.9/2.3 m/s with ball visible.
2. Athletic compacts when the ball is missing: width 35.8 m vs 41.6 m visible (-14%), length 58 m vs 52 m, centroid about 4 m deeper. Barcelona changes little (width 49.6 vs 52.0 m).
3. Missing time clusters in set pieces and long balls: Athletic goal kick 99 of 175 rows missing, Barcelona goal kick 81 of 175, Athletic long-ball tag 32 of 53.
4. Stoppage blocks end at the touchline. Ball positions at block edges sit at |y| 31–38 m; one block ends 5 frames before an off-pitch ball, and another ends 20 frames before a throw-in tag.
5. Nearest team at block start and block end agrees in 2 of 4 measurable gaps. The two flips came with ball displacements of 3.6 m and 12.7 m.
6. Slow visible ball (≤1 m/s) is held by the nearest side: H nearest 80% of rows, at 1.8 m vs A 6.0 m. At 1–5 m/s the nearest side is random (H 35%).
7. Tags agree with geometry. Build-up and long-ball tags put the ball in the taking team's own third (toward-attack share 0.00–0.07). Box-entry tags put it in the attacking third (0.88–0.97). Paired tags on the same frames are consistent when "defending in attacking third" is read as pressing.

**Signal ranking for possession while the ball is missing**
1. Last-seen ball owner carried forward: only available signal, but it flipped in 2 of 4 gaps.
2. Athletic width and depth compression (-14% width, +4 m deeper): suggests a block, but Barcelona does not show it.
3. Centroid advance during a gap (Barcelona +4 m toward attack while Athletic -6 m in one gap): single example, not reliable.
4. Team speed drop (about 0.8 m/s): separates missing from open play, not possession.

**Dead-ball signature**
Ball at or beyond the touchline (|y| ≥ 30 m) at a block edge, with both teams' median centroid speed ≤0.5 m/s during the block. Off-pitch readings are rare (2 rows). No static ball at all: no run of 3+ rows below 0.5 m/s, so stoppages are invisible in the ball data.

**Surprises / data problems**
- Barcelona defends +x (goal-kick ball at x≈45 m). Athletic's direction (-x) is inferred from its build-up geometry only; its goal-kick ball was never deep.
- Ball off-pitch readings reach 4 m beyond the lines.
- Team counts run 9–12 (Barcelona mean 11.3), so tracking is not clean.
- Simultaneous tags for both teams conflict in type (e.g. Barcelona build-up with Barcelona long ball), not in side.

**Confidence (0–100%)**
Possession while the ball is missing: 20%. Tag consistency: 65%. Dead-ball signature: 35%.
