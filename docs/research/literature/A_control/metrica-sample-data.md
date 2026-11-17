# Metrica Sports sample data (tracking + events)

- Source: https://github.com/metrica-sports/sample-data (Games 1-2 CSV, Game 3 EPTS FIFA format + JSON events)

## What is known (search text)
- Tracking files do not carry a ball-owning-team column (databallpy sets Metrica ball_possession to None and infers it from events).
- In Metrica data the ball position is typically EXACTLY equal to the position of the player in possession (noted in a learned-model paper). The closest player to the ball also changes often without a real possession change.
- Event types include passes, recoveries, ball lost, challenges (aerial, ground); the exact mapping Metrica uses to a per-frame "owning team" in their corrected tracking product: NOT found.

## Relevance to us
- On Metrica, "ball inside 1.0 m of a team" is nearly an oracle when the ball is visible (ball glued to carrier); thresholds tuned there will be too tight on real optical tracking (IDSSE/TRACAB, challenge data). Tune zone radius on IDSSE or with position noise.
- Truth derived from events: switches happen at the event frame of the controlling action (recovery / reception), so a predicted switch dated at confirmation (k frames later) is systematically late.
