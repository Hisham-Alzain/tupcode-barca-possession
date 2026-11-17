# Swarm findings: what the raw tracking says about possession (2026-11-11)

7 Haiku agents, one 2.5-minute window each (5 Hz), brief `.planning/envelopes/D1-swarm-brief.md`, reports `D1-swarm-w1..w7.out.md`. w1–w2 = public window with true labels; w3–w7 = unlabelled, spread over both halves. Allowed by ADR-001. Everything below is a **hypothesis to verify with code on the full match and to tune on external data**, never a rule for specific frames. Agent confidence ranged 20–55%.

## 1. Confirmed across windows (≥ 3 windows agree)
| # | Finding | Evidence | Consequence |
|---|---|---|---|
| F1 | **A missing ball is not a dead ball.** | w2 (labelled): the 3 gaps of 5–15 s have the same live label on both sides; only 28% of D rows lack the ball, so 72% of dead-ball frames *have* a ball. w3/w4/w6: whole blocks shift 15–30 m during 20–45 s gaps (an attack in progress). | MatchLens `deadball.py` "ghost > 0.5 s ⇒ dead" is wrong for this data. Remove it. This is the single biggest fix. |
| F2 | **Possession usually survives a gap.** | Nearest team at last-seen vs first-seen ball agrees: w4 4/4, w6 3/3, w2 3/3 (true labels), w7 2/4, w5 1/3. ≈ 13/17 ≈ 75%. Player-anchored fill found a team change in 30/147 gaps (20%). | Default: carry the last owner through a gap (an HMM with a sticky prior does this naturally). Spend the evidence budget on the ~20–25% of gaps where it changes. |
| F3 | **Team speed halves while the ball is missing.** | All 7 windows: centroid speed ≈ 0.4–0.8 m/s missing vs 1.0–1.9 m/s visible. | Ambiguous on its own (slow build-up *or* stoppage). Use it with F4, not as a dead-ball trigger. |
| F4 | **Dead ball looks like: ball far from everyone + near/over a line + collective slowdown.** | w2 (labelled): nearest-player distance median 2.9 m in D vs 1.3–1.6 m live; nearest > 8 m ⇒ 78% D. w2/w5/w6/w7: ball at |y| ≥ 30 m or past the goal line at block edges; corner-spot pinning; speed drop to ≈ 0.5 m/s. Only ~10% of D frames have the ball actually off-pitch. | Dead-ball detector = combination: max distance to nearest player, distance to touch/goal line, team speed collapse, set-piece tags. Not "ball absent", not "ball off pitch" alone. |
| F5 | **Nearest team to a visible ball is a decent but noisy owner.** | w1 (labelled): agrees with truth 79% (85% within 3 m). Flips *lead* true changes by 0.2–1.6 s. w5/w7: at ball speeds 1–5 m/s the nearest side is near random; slow ball (≤ 1 m/s) is held by the nearest side 80%. | Use nearest team weighted by ball speed and distance, plus retention (stays on it ≥ ~1 s) before switching. Matches the "controlled possession" rule. |
| F6 | **Use relative team features, not absolute.** | w5: both centroids drift together by 14–18 m during gaps (common-mode). w3: home-minus-away centroid x separates owner with AUC 0.73; Barcelona y-spread 0.77. w1 (labelled): home convex hull +44%, y-spread +50% when H has the ball; away shape barely changes. | Shape features: H–A centroid difference toward each team's attack, hull and y-spread *of the team in question*, line heights. Normalise by player count. |
| F7 | **Tracking degrades exactly where the ball is lost.** | w3: 47% of gap rows have ≤ 8 Barcelona players (0% when the ball is visible). Counts 7–12 per team in every window; count changes produce fake centroid jumps of ~10 m/s (w1). | Shape features must be robust: medians, count-normalised, ignore frames with < 8 players, smooth over seconds. |
| F8 | **Tag "Team" means different things by tag type.** | BUILD UP / PROGRESSION match the attacking team 85–100% (w1, w2). DEFENDING-* name the *defending* team. BALL IN THE BOX / FINAL THIRD are unreliable (ball in box only 19–48% of rows; w1 Athletic box tags have truth = H 75–81%). Goal-kick and long-ball tags coincide with ball-missing blocks (w7). | Map per tag type: attacking codes → that team has it; DEFENDING → the other team; box/final-third → low weight. Set-piece tags mark restarts (D end). |

## 2. Data problems to handle in code
- Ball positions beyond the lines by 4–8 m; stale repeated values (x = −51.8 m 19 times in w6); jumps up to 33–40 m/s at 0.2 s steps. → ball cleaning step (PLAN step 2): drop teleports, flag clamped values.
- True labels contain sub-second possessions (0.6 s and 1.2 s H inside A, w2). The scorer counts them; a sticky model will miss some. Accept that cost.
- The public window is not representative: 4% (w1) and 19% (w2) ball missing vs 41% in the match. One more reason never to judge versions on it.
- Attack direction is recoverable per half (deepest player / goal-kick ball position) — matches our adapter's side estimate.

## 3. What this says actually solves it (ranked)
1. **Fix dead-ball logic (F1, F4).** Missing ≠ dead; dead = far-from-players + line proximity + slowdown + set-piece tags. Expected to remove most of the 30-vs-17 over-splitting.
2. **Sticky segmentation (F2).** HMM/Viterbi over {H, A, D} with high self-transition; ball gaps inherit the last owner unless evidence says otherwise.
3. **Change detection inside gaps.** The ~20–25% of gaps where the owner changes: player-anchored team at both ends (gap_info A ≠ B), relative shape flip (F6), tags (F8).
4. **Visible-ball owner with retention (F5).** Nearest team weighted by speed and distance, switch only after control is kept ≈ 1 s.
5. Ball gap interpolation: irrelevant for scoring (the 10 fill variants agree), only needed for touch geometry in short gaps.

## 4. Next verification (code, not agents)
- Recompute F2, F3, F4, F6 on the full match automatically, and on Metrica/IDSSE with true labels and long-block ball hiding.
- Tune every threshold above on external data only.
