# D1-swarm w1 report (w1_public_a)

**Window and what happens**
750 rows at 5 Hz (150 s): 604 H, 119 A, 27 D, 10 label runs. Only 31 rows lack the ball, one 29-row gap (rows 330-358) that spans an A to D to H stoppage. Barcelona (home) defends the -x goal and attacks +x (home's deepest player is at x < -40 in 378 rows and never at x > 40; away is mirrored).

**Findings**
1. Nearest-player team agrees with gt on 79% of visible interior rows (526/668); 85% when the ball is within 3 m of a player. Disagreements cluster near label changes.
2. Nearest-team flips tend to come before gt changes: of 6 measurable H/A boundaries, a sustained flip led by 1 to 8 rows (0.2 to 1.6 s) in 5 cases, lagged 1 row in one, and was not found in one. Small n.
3. Median ball speed: 5.6 m/s in H rows, 8.3 in A, 10.3 in D (D has only 2 visible rows). Boundary rows are not distinct (median 4.6 m/s vs 6.0 interior).
4. Home shape separates by possession: median convex hull 1810 m^2 in H rows vs 1256 in A rows (+44%); home y-spread 18.3 m vs 12.1 m. Away hull barely changes (926 vs 983). Centroid x does not separate (home cx -7.7 H vs -8.1 A).
5. In the 29-row gap, both teams' centroid speed falls from about 3.7 and 2.7 m/s to under 0.3 m/s within 1.6 s, then recovers to about 0.9 m/s before restart. The last visible ball was moving about 6 m/s toward the y = -20 touchline.

**Signal ranking for possession while the ball is missing**
1. Last visible ball position and direction, extrapolated to the touchline: strongest local cue, but seen once here.
2. Home hull and y-spread: largest possession difference (+44%, +50%), measured on visible rows only, so unverified while the ball is missing.
3. Nearest team at the last visible row: 79% interior accuracy, usable only as a prior.
4. Team centroid speed collapse: separates dead ball, not H vs A.
5. Centroid x: no separation, not useful.

**Dead-ball signature**
Ball absent about 5.8 s, starting 0.8 s after the last visible ball. Both teams' speed drops to 0.2-0.6 m/s, then rises slowly. Restart when the ball reappears near (-25, -23) with a player within 0.1 m. D ends 0.4 s after reappearance. No ball coordinate ever leaves the pitch (max |x| 44.2, max |y| 33.6), so D is inferred from the gap and stillness, not from position.

**Surprises / data problems**
- Tags conflict with gt. "BALL IN FINAL THIRD" and "BALL IN THE BOX" tagged Atletic Club show gt = H in 75-81% of labelled rows. Barcelona "PROGRESSION" tags match gt 85-100%. "DEFENDING" tags are consistent only if Team means the defending side. Tag Team semantics vary by tag type.
- Player-count changes (10/11/12) cause centroid spikes: median 10.5 m/s on count-change steps vs 1.7 otherwise (n=21). Use medians or count-robust stats.
- Only 31 of 750 rows (4%) lack the ball, not 41%. This window cannot teach the swarm the long-gap behaviour beyond one event.

**Confidence (0-100%)**
Attack direction and nearest-team proxy: 70%. Shape rules and gap behaviour: 30% (one gap, 6 transitions).
