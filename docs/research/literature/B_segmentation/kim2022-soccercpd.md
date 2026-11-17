# SoccerCPD: Formation and Role Change-Point Detection in Soccer Matches Using Spatiotemporal Tracking Data

- Authors: Hyunsung Kim, Bit Kim, Dongwook Chung, Jinsung Yoon, Sang-Ki Ko
- Year / venue: KDD 2022, pp. 3146-3156
- URL / DOI: https://arxiv.org/abs/2206.10926 , doi 10.1145/3534678.3539150 ; code github.com/pientist/soccercpd

## Key method
Frame-wise role assignment (Bialkowski-style), then two change-point steps: formation CPD on role-adjacency matrices (discrete g-segmentation, average scan statistic) and role CPD on role permutations. Separates tactical changes from temporary ones.

## Data needed
Player tracking with IDs (to follow role permutations); expert annotations only for evaluation.

## Reported results
Detects annotated tactical change points accurately (numbers not found in search text).

## Relevance to us
Low. Change points here are minutes apart (formation changes), not possession changes (seconds). The useful idea is a non-parametric graph-based scan statistic on team-shape graphs, which could in principle date a hidden-ball turnover, but simpler single change-point search on a 1-D shape signal does that job (see killick2012-pelt.md). Needs IDs; we drop them.
