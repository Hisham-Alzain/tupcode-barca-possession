# PathCRF: Ball-Free Soccer Event Detection via Possession Path Inference from Player Trajectories

- Authors: Hyunsung Kim et al. (KAIST / Fitogether)
- Year / venue: 2026, KDD 2026 (arXiv 2602.12080)
- URL / DOI: https://arxiv.org/abs/2602.12080 ; https://doi.org/10.1145/3770855.3818152 ; code github.com/hyunsungkim-ds/pathcrf

## Key method
- No ball input at all. Players form a fully connected dynamic graph; possession state at each time step = one edge (from-player -> to-player; self-edge = carrying). 26^2 = 676 candidate edges.
- Neural backbone (set-based encoder over players; exact architecture not verified) emits per-step edge scores; a Dynamic Masked CRF restricts which edge transitions are allowed; Viterbi decoding gives the possession path; an event is declared whenever the selected edge changes.
- Data: Sportec Open DFL dataset (= IDSSE, 7 matches).

## Reported results
- Edge accuracy 69.64% (676 classes); event F1 75.69%.
- Frame-level TEAM possession accuracy 92.40% without any ball data; errors on both teams cancel over time (home share 62.13% truth vs 62.64% predicted).

## Relevance to us
- The strongest evidence that team possession is recoverable from player positions alone: directly addresses failure path 5 (ball missing on 41% of frames, 5-30 s blocks) and the "turnover hidden inside a gap is dated when the ball returns" problem.
- Viterbi with a transition mask is the structured version of our k-frame retention: we can get the same benefit with a 3-state (H/A/D) HMM where emissions come from ball-zone evidence when the ball is visible and from a player-only classifier when it is not.
- Caveat: our challenge data has team labels but no IDs; PathCRF uses per-player identities only to define edges, so a team-level reduction (2 states + dead) is natural. Trained on IDSSE, which is in our bench (beware leakage between tune/held-out if reusing their weights).
