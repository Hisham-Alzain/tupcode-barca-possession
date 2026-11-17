# D1-swarm · reverse-engineer possession from raw tracking (shared brief)

Allowed by ADR-001 (organisers: "Yes you can use agents to read raw data"). Still forbidden: hand labels, tuning on the public window, frame-specific rules. Your job is understanding, not labelling.

## The problem
Barcelona (home, H) vs Athletic (away, A), 25 fps tactical camera. Every frame needs one label:
H = Barcelona has the ball, A = Athletic has it, D = ball out of play. A possession is a maximal run of H or of A; D always ends a possession. Possession changes only on **controlled** possession by the other team (a deflection, a block or a won header without control does not count; a loose ball stays with the last team). The score (spellF1) counts whole possessions: splitting one possession into two, or missing a stoppage, is what costs points.

The ball position is missing on 41% of the live frames, and almost all of it in long blocks: 131 gaps of 5 s or more hold 99.9% of the missing frames (typical gap 10–30 s, some over 30 s). Short gaps are rare. Players have no identities, only a team.

## Your data
One CSV under `data/challenge/_agent_slices/` (path in your task), 5 rows per second (every 5th frame), 2.5 minutes.
- Header lines starting `#`: window range, and the Smart Tagging phases (automatic, can be wrong) overlapping it, e.g. `# tag frames 11185-11900 BUILD UP {'Team': 'Barcelona', ...}`.
- Columns: `frame, t_s, gt, ball_x, ball_y, home, away`.
  - `gt` = true label H/A/D (only in public windows w1, w2; empty elsewhere).
  - `ball_x, ball_y` metres, pitch centred, x in [-52.5, 52.5] (goal lines), y in [-34, 34] (touchlines). Empty = ball not seen.
  - `home`, `away` = player positions `x:y|x:y|...` sorted by x. Usually ~10–11 per team; counts vary because tracking is uncorrected.

## What to do
Work through the numbers yourself (compute, don't guess). Look for **mathematical regularities** that would let a program decide H/A/D, especially while the ball is missing:
1. Ball visible: how do ball–player distances, ball speed and direction changes behave around a change of possession and around D?
2. Ball missing: what do the two teams' shapes do? Centroid x relative to each team's own goal, spread (std of x and y, convex-hull-like width/length), compactness, team velocity, the defensive line height, who is nearer the last ball position, how the teams move when the ball comes back.
3. D (dead ball): how does it look? Ball off-pitch, ball static, everyone slowing down, players regrouping, tags.
4. Tags: do they agree with what you see?

Report only general patterns (thresholds as rough magnitudes with units), never "at frame N do X".

## Output (≤ 350 words) to the file named in your task, these headings exactly
- **Window and what happens** (2–3 lines)
- **Findings** (numbered, each with the numbers that support it)
- **Signal ranking for possession while the ball is missing** (best first, one line why)
- **Dead-ball signature**
- **Surprises / data problems**
- **Confidence (0–100%)**
Reply with only the output file path.
