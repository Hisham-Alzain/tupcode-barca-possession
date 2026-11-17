# Using Temporal Convolutional Networks to estimate ball possession in soccer games

- Authors: Borghesi, Costa, Morra, Lamberti (Politecnico di Torino)
- Year / venue: 2023, Expert Systems with Applications 223, 119780
- URL / DOI: https://doi.org/10.1016/j.eswa.2023.119780 ; accepted version iris.polito.it (eswa_tcn_authors_accepted_version.pdf)

## Key method
- Per-frame 3-way game state: dead / home possession / away possession, from tracking data.
- Two designs: single ternary head vs two binary heads (dead-vs-alive, then which team). Best: two networks with self-attention.
- A TCN builds an independent trajectory embedding per object (players, ball); permutation-invariant fusion (element-wise sum, self-attention, 2D conv).
- Notes in the paper (search text): many situations make the owning team very hard to define; possession treated as two steps (alive? then which team?).
- Exact input features: NOT found.

## Reported results
- Possession accuracy 86.2% (+7.3 points over state of the art); dead/alive accuracy 89.2% (+33.2 points).

## Relevance to us
- Same problem framing as ours (H/A/D per frame). Two-stage (dead first, then team) matches our architecture (deadball2 + team_control).
- 86% frame accuracy is a realistic ceiling reference for learned models; our visible-frame team accuracy is ~68-72% for MatchLens.
- A learned temporal model handles retention and deflections implicitly from context; smallest version for us: gradient boosting over windowed team-level features + HMM smoothing (no deep training needed in the remaining time).
- Earlier group work (Morra et al., "Slicing and dicing soccer"; SoccER) is rule-based: atomic events from positional data plus temporal/logical rules -> complex events.
