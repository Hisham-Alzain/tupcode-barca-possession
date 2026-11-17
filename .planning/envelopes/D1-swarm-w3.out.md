**Window and what happens**
w3_h1_mid, frames 45000-48749 (about 1800-1950 s), 750 samples, first half. Barcelona (H) attacks toward +x. The ball is visible in 27% of samples, in four gaps of about 54, 30, 21 and 4 s; the last gap runs to the window end.

**Findings**
1. Gaps hold 96% of missing samples. Visible ball speed: median 4.3 m/s, p90 13.4, max 21. No run of 1 s or more with the ball near-static, so no visible dead-ball stop.
2. Tracking dropouts coincide with the gaps. 47% of missing-ball rows have 8 or fewer Barcelona players, against 0% of visible rows (Athletic: 21% vs 0%). Team-centroid features in gaps are unreliable for about half the rows.
3. When both teams have 9+ tracked players, Athletic is compact in gaps and visible alike (x-spread about 5.6 m, y-spread about 7 to 8 m). Barcelona's x-spread is about 20 m, probably goalkeeper or outlier inclusion rather than a real shape change.
4. Nearest-player proxy (ball within 3 m of a player, 169 rows, 141 Barcelona-nearest): Barcelona-minus-Athletic centroid x gives AUC 0.73, Barcelona y-spread 0.77, both x-spreads about 0.47 (no separation).
5. In the long gap, Athletic's centroid sits 30-42 m toward its goal while Barcelona's rises from about 16 to 28 m. Athletic's x-spread falls from about 12 m to about 4 m over the same span. Consistent with a Barcelona attack against a deep Athletic block, not proven.
6. Ball-return ambiguity: at one gap's start the ball barely moves (about 2 m) but the nearest team flips from Athletic to Barcelona. Either a contested ball or proxy noise.
7. Team centroid speed inside gaps is about 0.5 m/s for both teams (1.1 to 1.6 m/s for Barcelona in the two longest), so no collective stop is visible.

**Signal ranking for possession while the ball is missing**
1. Centroid x difference (H minus A): best separation, though weak (AUC 0.73).
2. Barcelona y-spread: AUC 0.77, but the label itself is noisy.
3. Athletic compactness when deep (x-spread under about 6 m): descriptive, not a separator.
4. Last visible ball position and nearest team: useful at gap edges only, and it flips when the ball barely moves.

**Dead-ball signature**
None seen. The throw-in tag (team N/A, frames 45000-45151) has the ball at y -12 to -24, 10-20 m from the touchline at 34 m, and moving. The free-kick tag shows no static ball.

**Surprises / data problems**
- Barcelona drops to 2-3 tracked players for roughly 15-20 s inside the longest gap.
- The "7+ players in the box" tag for Barcelona is not supported: at most 7 Barcelona players in the attacking box, in 2 of 401 rows (median 1). Athletic reaches 11 in that box, which points to a team-label swap or tracking error.
- "Ball in the box" tags: the ball is in the box (x > 36, |y| < 20) in only 19-30% of visible rows.
- Counts go up to 12 Athletic players, so tracking is over-inclusive at times.

**Confidence (0-100%)**
30%. Gap-period team features are confounded by player-count dropouts, and the possession proxy is noisy, so no reliable rule emerges from this window.
