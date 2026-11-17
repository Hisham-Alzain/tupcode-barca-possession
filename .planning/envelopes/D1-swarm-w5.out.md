# D1-swarm w5_h2_early report

**Window and what happens**
150 s, 750 rows at 5 Hz, second half. Ball visible on 68% of rows; four missing blocks of 7.6 to 15.2 s hold 99% of missing time. Barcelona (home) attacks toward -x and Athletic (away) toward +x. Play is open with fast passing and a few short off-pitch stoppages.

**Findings**
1. Direction is consistent with tags. Away players sit beyond x=-40 in 74% of frames and home beyond x=+40 in 64%. Barcelona BALL IN THE BOX has ball x median about -34 m; Athletic FINAL THIRD about +30 to +37 m.
2. Absolute team centroids move with the ball-missing blocks, but both teams move almost identically: home-vs-away centroid shift differs by only 0.6 to 2.9 m (block shifts of -14 to +18 m). A common-mode translation dominates.
3. Team shape while ball is missing: centroid speed 1.1 m/s vs 1.7 to 1.8 m/s when visible. Spread-x (12.5 vs 12.6 m home; 16.2 vs 17.8 m away) and spread-y (about 12 m) barely change. Teams slow down but do not compress.
4. Visible ball: median speed 5.5 m/s, p90 13.7 m/s, 30% of frames above 8 m/s, static (<0.3 m/s) on only 3%. Nearest-player distance median 2.5 m; 56% of frames have a player within 3 m, so nearest player is a weak owner proxy.
5. Nearest team across missing blocks: last-seen to reappearance went H to A in two blocks, A to A in one, and is undefined in one that runs to the window end. Too few to rank.
6. Tags: Athletic-tagged final-third and box windows have away nearest 74 to 85% of visible frames (agrees). Barcelona-tagged final-third and box windows have home nearest only 27 to 35% (disagrees, though those balls sit deep in Athletic's half where defenders cluster).

**Signal ranking for possession while the ball is missing**
1. Nearest team at block edges: the most direct owner signal, but n=3 here.
2. Home-minus-away centroid x (not raw centroid): removes the common-mode shift; untested here.
3. Absolute team centroid x: rejected for now, dominated by common shift.
4. Spread (x and y): no detectable change; not useful.

**Dead-ball signature**
Three off-pitch ball runs of 3.4 to 3.8 s (55 frames, about 11 s total). Both teams' median speed drops from about 1.0 to 1.2 m/s to 0.5 to 0.6 m/s, and home spread-x shrinks by 1.5 to 2.5 m. One static-ball run of 1.4 s near the goal line on the away-defended end, with both teams at about 1 m/s. Long missing blocks are not clearly dead-ball.

**Surprises / data problems**
- Common-mode shift: both centroids move by the same amount (differences up to 3 m) in every missing block; this looks like scene or tracking drift, not team shape.
- Only four block-edge samples carry the possession question.
- Tag windows are long (up to about 35 s) and extend past the window edges. SET PIECES (throw-in) tags sit over 78 to 90% visible-ball frames, so tags are not ball-precise.
- Home count 9 to 11, away 10 to 11; no goalkeeper identities.

**Confidence (0–100%)**
35%
