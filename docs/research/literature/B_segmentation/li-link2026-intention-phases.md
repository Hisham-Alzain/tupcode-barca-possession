# Intention Driven Identification of In-Possession Match Phases in Association Football through Temporal Graph Learning

- Authors: Yuesen Li, Daniel Link (TU Munich)
- Year / venue: 2026, arXiv preprint 2606.09289 (not peer reviewed)
- URL: https://arxiv.org/abs/2606.09289

## Key method
Temporal Graph Attention Network: frame-level player-interaction graphs (team-membership indicator per node) + contextual features + Transformer temporal model; hierarchical labels (3 intentions, 6 phases: build-up, progression, counter-attack, maintenance, sustained threat, finishing); post-processing of the frame predictions into segments.

## Data needed
7 Bundesliga matches, TRACAB 25 Hz, frame-level phase annotations.

## Reported results
Frame-level macro F1: 0.87 (intention), 0.76 (invasion phases), 0.79 (scoring phases). Sequence-level IoU-based F1 rose **0.68 -> 0.79 (intentions) and 0.61 -> 0.71 (phases) from post-processing alone**. "Sequence modelling was the main driver of segmentation quality."

## Relevance to us
High as evidence, low as code. The most direct quantitative statement found that **temporal post-processing/segmentation adds ~+0.10 segment-level F1 on top of the same frame classifier** — the same kind of metric as our spellF1. Their 7-match training set is close to our 5 matches, so a learned sequence model on our data is not absurd, but the T-GAN itself is too much for 5 days. Its phases presuppose the in-possession team (they label phases of the team in possession), so it is not a possession detector.
