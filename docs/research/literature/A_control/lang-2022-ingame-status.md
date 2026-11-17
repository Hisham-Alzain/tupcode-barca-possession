# Predicting the in-game status in soccer with machine learning using spatiotemporal player tracking data

- Authors: Steffen Lang, Raphael Wild, Alexander Isenko, Daniel Link
- Year / venue: 2022, Scientific Reports
- URL / DOI: https://doi.org/10.1038/s41598-022-19948-1 ; PMC9522646

## Key method
- Player positions ONLY (no ball) -> frame-level in-game status (ball in play vs stoppage).
- 102 Bundesliga matches with manually annotated status; logistic regression, decision trees, random forests, AdaBoost compared.

## Reported results
- Up to 92-93% frame accuracy on 47 unseen matches; AdaBoost best, 81% precision for stoppages > 2 s.
- Team possession without ball: not addressed (not found).

## Relevance to us
- Mostly cluster C (dead ball), but it matters for A: dead-ball errors are the biggest lever after iteration 2 (oracle 0.356 -> 0.557) and stoppages are where the team label resets. Cheap tree models over player-only features work; the same feature pipeline can serve a player-only team-possession classifier for ball gaps.
- Defines the "Episode" concept used by Link's group (stable possession >= 3 s, see li-link-2026-phases.md).
