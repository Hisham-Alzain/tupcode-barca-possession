# Event-provider possession definitions: Opta / Stats Perform, StatsBomb (Hudl), SPADL / atomic-SPADL

## Opta / Stats Perform
- URL: https://www.statsperform.com/resource/sequences-and-the-premier-league ; theanalyst.com/articles/opta-football-stats-definitions
- Rule: "a possession is ended by the opposition gaining control of the ball". Possession = one or more consecutive sequences of the same team; sequences end on defensive actions, stoppages, shots (so a stoppage does NOT end a possession if the same team restarts). Older Opta wording: a possession starts when a player controls the ball and ends when they no longer do.
- Clearances / aerial duels / deflections: explicit rule NOT found; by the control rule a block or deflection without control does not switch.

## StatsBomb (Hudl)
- URL: https://www.hudl.com/releases/statsbomb (2025-09-22 release note)
- `possession` / `possession_team` fields; the possession model was revised in 2025 ("breaking play earlier or grouping isolated touches"). Exact rule text NOT found.

## SPADL / atomic-SPADL (Decroos et al., KU Leuven; socceraction)
- URL: https://socceraction.readthedocs.io/en/stable/documentation/spadl/atomic_spadl.html
- No explicit possession-sequence rule found. Atomic-SPADL splits a pass into give + receive and inserts an `interception` action if the other team gets the ball or `out` if it leaves play. Atomic-VAEP feature: whether the team with possession two actions ago still has it.

## Relevance to us
- All event definitions agree with our truth: change only on controlled possession by the other team; loose balls, blocks and aerial duels without control keep the last team; same-team restarts after a stoppage keep the possession.
- Implication for the state machine: a duel or "nobody in zone" frame must keep the previous team (our team rule already does this; MatchLens did not -> failure paths 3, 4).
- Implication for restarts (cluster C): if the team that had the ball also restarts, truth may treat it as one possession across the stoppage (Opta) or two (our scorer splits on D). Check how the truth builder handles it.
