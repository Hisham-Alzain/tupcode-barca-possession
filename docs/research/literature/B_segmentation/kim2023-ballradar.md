# Ball Trajectory Inference from Multi-Agent Sports Contexts Using Set Transformer and Hierarchical Bi-LSTM

- Authors: Hyunsung Kim, Han-Jun Choi, Chang Jo Kim, Jinsung Yoon, Sang-Ki Ko
- Year / venue: 2023, KDD 2023
- URL / DOI: https://arxiv.org/abs/2306.08206 , doi 10.1145/3580305.3599779 ; code reportedly github.com/pientist/ballradar (repo listed in one search result; a second search could not confirm it, so unverified)

## Key method
Infers the ball trajectory from player trajectories only. Set Transformer encoders give permutation-invariant/equivariant team context per frame; a hierarchical Bi-LSTM first predicts per-player ball-possession probabilities, then the ball position conditioned on them. Adds a "reality loss" penalising physically implausible ball motion, plus post-processing. Finding: LSTM beat Transformer variants (latest player positions are the most informative context).

## Data needed
Player tracking with player identity per team slot and ball tracks for supervision. Reported training data (repo description, from search text): 15 K League matches of GPS tracking + 3 Metrica sample games. Player IDs are used as set elements, but the set encoder is permutation invariant, so slot identity across frames is less critical than in an LSTM-per-player design (not verified).

## Reported results (from search snippets / secondary summaries, not checked against the paper's tables)
- Ball position error ~3.65 m (H-LSTM-RL best variant).
- Possession accuracy: 64.7% for the individual player, **85.8% for the attacking team** (players only, no ball).
- With 20% of the true ball given (imputation): error 1.41 m, possession accuracy 91.2%.
- Metrica-only breakdown: not found.

## Relevance to us
High, as a reference ceiling: from players alone a learned model names the team in possession ~86% of frames. Our hidden-ball carry-forward is right on ~75% of gaps (swarm F2), so a players-only team classifier could beat carry-forward inside long gaps, but only if decoded with a sticky segmentation (86% per-frame accuracy with independent errors would fragment possessions badly under spellF1). Training the full model in 2 days on our 5 matches: risky (they used 18 matches; our IDSSE/Metrica lack consistent IDs after we drop them). A lighter version (hand features + GBM or small BiGRU, see survey C2/C5) is the realistic transfer.
