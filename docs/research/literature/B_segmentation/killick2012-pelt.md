# Optimal Detection of Changepoints with a Linear Computational Cost (PELT)

- Authors: Rebecca Killick, Paul Fearnhead, Idris A. Eckley
- Year / venue: 2012, JASA 107(500):1590-1598 (arXiv 1101.1438)
- URL: https://arxiv.org/abs/1101.1438 ; Python: `ruptures` (Pelt), R: `changepoint`

## Key method
Exact minimisation of sum(segment costs) + lambda x (number of change points), with pruning that keeps expected cost O(n). Penalty usually BIC/SIC-like.

## Data needed
Any 1-D/multi-D signal; no labels (lambda tuned).

## Reported results
Orders of magnitude faster than Optimal Partitioning, more accurate than Binary Segmentation.

## Relevance to us
Medium. Penalised segmentation with a per-change cost lambda is the same objective as Viterbi with a switch penalty, but PELT gives boundaries, not labels. Useful narrowly: **dating a turnover inside a ball gap** — run a single-change-point search (or PELT with high lambda) on a gap-internal signal such as home-minus-away centroid x in attack direction, team spread, team speed; place the switch at the best change point instead of at the frame the ball reappears. In a full H/A/D Viterbi this comes for free if gap frames get shape-based emissions, so prefer that route.
