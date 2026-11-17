# Automatic event detection in football using tracking data

- Authors: Ferran Vidal-Codina, Nicolas Evans, Bahaeddine El Fakir, Johsan Billingham
- Year / venue: 2022, Sports Engineering 25, art. 18 (arXiv 2202.00804, v3 31 Aug 2022)
- URL / DOI: https://arxiv.org/abs/2202.00804 ; https://doi.org/10.1007/s12283-022-00381-6

## Key method
Two stages, deterministic decision trees.
1. Possession model, per frame, per player. Each frame gets one of four ball-control states: dead ball, no possession, possession, duel.
   - Possession zone (PZ): circle around each player; player has control when the ball is inside it. Radius is provider-dependent: 0.5 m for provider A, 1 m for providers B and C (search text).
   - Duel zone: circle around the ball of radius >= PZ radius; if at least two opponents are inside it, the frame is a duel. Duel overrides possession.
   - Gain: inside a run of control frames for one player, a gain is recorded only if the ball changes direction by more than eps_theta OR speed by more than eps_v ("ball changes trajectory and speed" -> gain; constant -> no gain, the ball flew past/over the player).
   - Loss: ball leaves the player's PZ, ball displacement exceeds a threshold, and the player is absent from the next control frame.
2. Event model: changes of possession + laws of the game -> passes, receptions, interceptions, shots, set pieces.

Numeric values of eps_theta, eps_v, the displacement threshold and duel radius: NOT found in search text (our port uses 0.85 dot product, 1.5 m/s, 0.5 m, 2.3 m; these could not be verified against the paper).

## Reported results
+90% detection rate in most event categories across tournaments and providers (abstract). Possession-level accuracy per frame: not found.

## Relevance to us
- This is exactly MatchLens. The touch-evidence gain test is the cause of failure paths 1 and 2 (Candidate P with no team) in docs/possession-state-machine.html. The paper evaluates events, not a per-frame team label, so it never had to label the frames between first control and proven touch: abstaining there is harmless for event detection but costly for spellF1.
- PZ radius is provider-dependent in the paper itself. Supports re-tuning our 1.0 m zone per data source (IDSSE switches too rarely: 286 vs 511 possessions).
- The touch test was designed against a ball flying over a player. Our bench can use it as a veto only (do not switch while the ball passes at speed with no change), not as a gate before naming a team.
