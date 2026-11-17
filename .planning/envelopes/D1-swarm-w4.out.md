# D1-swarm w4 (w4_h1_late, frames 78000-81749)

**Window and what happens** (2–3 lines)
150 s, 750 rows at 5 Hz. Ball visible on 44% of rows; five gaps (5.0, 5.2, 8.8, 19.6, 44.8 s) hold 417 missing rows (about 83 s, 55% of the window). Around frames 79165-79270 the ball sits beyond the touchline (y to -42 m). A 45 s gap (frames 80055-81170) overlaps the Barcelona set piece tag (80913-81788).

**Findings**
1. The 45 s gap is a whole-block shift toward +x. Home centroid x goes 11.6 -> 29.4 -> 24.8 m and away 24.5 -> 42.6 -> 42.1 m. Away's back line moves from x 3.3 to 37.4 m at mid-gap, and home's front reaches x 51.6 m.
2. The 20 s gap (78200-78685) goes the other way: both centroids move about 8-17 m toward -x (home 6.1 -> -9.4, away 17.8 -> 1.2) while the block compresses.
3. Shape diverges by possession state. Home x-length is 48 m with ball visible and 68 m missing; away is 65 m visible and 39 m missing. Away's x-spread drops 17.1 -> 10.6 m while home's rises 13.7 -> 18.5 m.
4. Nearest team to the last-seen ball matches the nearest team to the first-seen ball in all four gaps with both ends observed (A-A, H-H, H-H, A-A). One 5 s gap moves the ball about 30 m (x -0.1 -> 29.8), roughly 6 m/s average.
5. With ball visible, the nearest home player is a median 3.3 m away and the nearest away player 8.7 m. Ball x correlates 0.59 with home centroid x and 0.51 with away.
6. Ball speed between 0.2 s rows: median 5.2 m/s, p90 15 m/s, max 33 m/s. Centroid speed is 1.39 m/s visible and 0.96 m/s missing.
7. Off-pitch episode (79140-79290): both centroids drift about 6 m toward +x over 7.5 s (home -31 -> -25, away -17 -> -10). No regroup or stop is visible.

**Signal ranking for possession while the ball is missing** (best first)
1. Nearest team to the last-seen ball. It agreed at every testable gap boundary, but the sample is four gaps.
2. Divergence of team x-extent (ball-holder stretches, defender compacts). Strong in the 45 s gap, but its direction depends on the gap.
3. Block centroid position and direction of drift. Both teams move together, so direction alone is weak.
4. Tags. Ten of thirteen windows agree with nearest-player majority, after reading defending tags as the opposite side.

**Dead-ball signature**
Not clean. The only off-pitch run (23 rows, 4.6 s) shows no stoppage in team motion. Short static runs (0.4-1.4 s, speed under 0.5 m/s) occur at (25, 25), (-16, -29) and (-38, 10). None lasts long enough to read as a dead ball.

**Surprises / data problems**
- Ball y reaches -42 m, beyond the ±34 m touchline, for 23 rows. Unclear if this is out of play or tracking error.
- Max ball speed of 33 m/s and p90 of 15 m/s at 0.2 s steps suggest jumps.
- Barcelona "BALL IN THE BOX" (80925-81308) has the nearest player away on 100% of visible rows. "PLAYERS IN THE BOX" and "BALL IN FINAL THIRD" for Barcelona show home nearest only 40-41%.
- Player counts vary (home 8-12, away 9-12).
- Missing share here is 55%, above the 41% global figure, so this window is harder than average.

**Confidence (0–100%)**
30%. Nearest-player heuristics are supported by four gaps and thirteen tag windows. The shape signals are one 150 s slice with no labels, so they may not generalise.
