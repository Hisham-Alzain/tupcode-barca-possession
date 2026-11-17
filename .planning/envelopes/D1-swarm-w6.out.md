# D1-swarm w6_h2_mid report

## Window and what happens
Frames 150000-153749 (t 6000-6150 s, 750 rows at 5 Hz), 2nd half. Barcelona (H) attacks toward -x (inferred from tags: median ball x during Barcelona-attacking tags is -38 m); Athletic (A) defends the -52.5 goal here. Ball missing on 50% of rows, in four gaps: 45.6 s (t 6096-6141), 15.6 s (6004-6020), 11.2 s (6074-6085), 2.6 s.

## Findings
1. Ball-visible frames: H nearest player to ball in 250/375 (67%); median nearest-player distance H 2.9 m, A 5.3 m. Ball within 2 m of any player in 177/375 (47%).
2. Ball speed median 5.4 m/s, p90 14.6 m/s, max 39.6 m/s (33 samples >15 m/s). The max is a likely tracking jump.
3. Shape while ball missing vs visible: H own-goal depth 61 m vs 73 m, H width 34 m vs 40 m; A depth 30 m vs 17 m, A width 29 m vs 30 m. Both blocks compress toward mid-pitch when the ball is hidden.
4. Team centroid speed halves when the ball is missing: about 1.3 m/s (visible) to 0.6 m/s (missing) for both teams. This is the strongest shape-based cue here.
5. Across gaps, the two centroids move in opposite directions (11 s gap: H +19 m toward A goal, A -19 m; 46 s gap: H -27 m, A +30 m). Only 3 events, so weak evidence for possession direction.
6. Visible-to-visible ball jumps across gaps: 0.5 m (15.6 s gap, same corner spot), 50 m (11.2 s), 12 m (45.6 s).
7. Tag coverage: 605/750 frames tagged. All 145 untagged frames fall inside the 45.6 s gap (t 6110-6138).

## Signal ranking for possession while the ball is missing
1. Last visible ball position + nearest-team to it (H nearest in 3/3 gaps) - only cue that ties the ball to a team; weakly reliable.
2. Centroid speed drop (about 2x) - flags a stoppage/dead-ball-like regime, not who has the ball.
3. Opposing centroid shifts across the gap (item 5) - direction hint, n=3.
4. Block depth - differs by about 12 m between states, but confounded by where the ball was.
Team identity from shape alone was not separable in this window.

## Dead-ball signature
- Ball off-pitch segments at the goal line and touchline (t 6002.6, 6029.6, 6073.6, 6087.6), 37 frames in total.
- Ball parked at a fixed coordinate (-51.8, -7.3) for about 3.4 s (t 6092.4-6095.8) before the 46 s gap; a stale value.
- Corner-area pinning at (-51, -33) across the 15.6 s gap, then the same point after: a corner-type stoppage.
- Both centroid speeds drop to about 0.6 m/s.

## Surprises / data problems
- Tags disagree with ball positions: BALL IN THE BOX tags have the visible ball inside the box in only 111/233 samples (48%); FINAL THIRD 275/331 (83%); PROGRESSION 92/130 (71%).
- Ball x repeated at -51.8 in 19 samples and at -43.4/-42.2 in 4 each: clamping or stale values.
- Ball x reaches -56.7 (beyond goal line) in visible rows.
- Player counts: H 9-12 (median 11), A 7-12 (median 11).
- Missing fraction here (50%) is above the 41% dataset average.

## Confidence (0-100%)
55%. Stoppage and dead-ball cues are clear, but possession direction rests on three gaps and no labels for this window.
