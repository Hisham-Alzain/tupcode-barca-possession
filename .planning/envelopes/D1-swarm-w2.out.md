**Window and what happens**
w2_public_b, t 597–747 s, 750 rows at 5 Hz. gt: A 399 rows (53%), D 266 (35%), H 85 (11%). 12 label runs: short A→H→A blips (0.6 s, 1.2 s) inside A spells, D stoppages of 14–20 s, and a final H spell of 11.8 s.

**Findings**
1. Ball missing on 146/750 rows (19%), in only 3 gaps of 26, 46 and 74 rows (about 5–15 s). Missing share by label: A 18%, D 28%, H 0%. Each gap has the same label on both sides (A|A, A|A, D|D), but n=3.
2. Nearest-player distance separates live play from dead ball. Median nearest distance: A 1.3 m, H 1.6 m, D 2.9 m. p90: A 5.2 m, H 6.6 m, D 9.7 m. Of 41 visible frames with nearest player >8 m, 32 are D (about 17% of visible D frames, versus 1.6% of live frames).
3. Dead-ball onset is preceded by a rise in the distance from the ball to the nearest player of the team that held it. Examples: H→D at 695.2 s rises 5.5→11.9→13.9 m over 3 rows; A→D at 721.2 s rises 2.3→7.5→16.8 m over 3 rows.
4. Ball speed (median, visible frames): A 6.0 m/s, H 4.5 m/s, D 2.6 m/s. No label has static frames (<0.3 m/s), so "ball static" is not a D signal here.
5. Live changes are contested. At A→H (614.0 s) and H→A (614.6 s), both teams are 1–3 m from the ball and the sign of (dH − dA) flips within about 1 s. Nothing jumps at the change.
6. In-possession centroid shift (own-team centroid x, m): H 22 m in H spells vs 11 m in A spells; A −0.6 m in A spells vs −6.9 m in H spells. Direction is inferred from the sign of H's x. This is confounded by ball x, so weak.
7. Team centroid speed during gaps is about half of visible speed: home 0.40 vs 1.05 m/s, away 0.45 vs 0.73 m/s.
8. Tags, mostly consistent under an attacking/defending reading. BUILD UP (Barcelona) is 91% H; PROGRESSION (Atletic) is 88% A; SET PIECES (Barcelona goal kick) is 81% H; DEFENDING MIDDLE THIRD (Barcelona) is 88% A. Atletic set pieces and CREATING CHANCES are mostly D (76%, 75%).

**Signal ranking for possession while the ball is missing** (best first)
1. Label persistence across the gap (3/3 same label). Only a sample of 3, so unproven.
2. Team-centroid speed drop. Gaps show about half the visible speed; it is a hint of dead time, not of who has the ball.
3. Own-team centroid shift toward the opponent goal (item 6). Confounded by ball position.
4. Not measured: nearest team to the last known ball position before a gap.

**Dead-ball signature**
Nearest-player distance rises to more than 8 m (item 2–3). Ball |x| median is 41 m (near goal lines, as for H). 10% of D visible frames are off-pitch (|x|>52.5 or |y|>34), against 0% of live frames. D runs are long (14–20 s) and the ball is on the pitch and still moving at 2–3 m/s (for example 635.6–655.8 s, ball about (−22…−26, −27.6), nobody within 13 m). So D is not only "ball off pitch".

**Surprises / data problems**
- The 41% missing figure from the brief does not hold for this window (19%), and gaps here are 5–15 s, not 30 s.
- H blips of 0.6 s and 1.2 s between A spells (614.0 s, 630.8 s) sit where both teams are within 1–3 m of the ball. These may be gt flicker or uncontrolled touches that should not count as changes. Check against the rule before trusting them.
- H has zero missing frames and only 85 rows, so nothing about the H side of gaps can be learned here.
- Defending tags name the team without the ball, so tag team and gt letter need mapping before comparison.
- The window has 11 changes and 1 gap-sequence; thresholds need checking on other windows.

**Confidence (0–100%)**
30%
