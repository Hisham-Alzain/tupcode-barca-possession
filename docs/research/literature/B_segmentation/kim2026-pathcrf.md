# PathCRF: Ball-Free Soccer Event Detection via Possession Path Inference from Player Trajectories

- Authors: Hyunsung Kim, Kunhee Lee, Sangwoo Seo, Sang-Ki Ko, Jinsung Yoon, Chanyoung Park
- Year / venue: 2026, KDD 2026 (arXiv Feb 2026)
- URL / DOI: https://arxiv.org/abs/2602.12080 , doi 10.1145/3770855.3818152 ; code https://github.com/hyunsungkim-ds/pathcrf

## Key method
Player trajectories as a fully connected dynamic graph; at every frame exactly one edge (possessor -> next receiver, or similar) is the "possession state". A neural backbone scores edges, a **Dynamic Masked CRF** decodes the edge sequence: logically impossible edge-to-edge transitions are masked with a large negative score, and the probability of *staying* in the current state is made context-dependent (dynamic). Viterbi gives the single best sequence; an event is emitted whenever the selected edge changes. Trained with CRF negative log-likelihood (forward algorithm).

## Data needed
Player tracking + event data (possession path labels), no ball at inference. Dataset size: not found in search text (same group's earlier work used K League GPS + Metrica).

## Reported results (abstract)
- Correct possession edge among 676 candidates: 69.64%.
- Event detection F1: 75.69%.
- Ablation: plain classifier + Viterbi constrained decoding is a baseline; the dynamic (context-dependent) stay probability gives a realistic event count (1,961) and the best F1. Exact ablation numbers: not found.

## Relevance to us
**Very high, as a design pattern.** It is exactly "whole-match segmentation where hand rules become evidence": per-frame scores + transition constraints + Viterbi, with a learned, context-dependent self-transition. The key lesson for spellF1: a fixed switch penalty over- or under-segments (our IDSSE under-switching, 286 vs 511 possessions, is this symptom); making the stay cost depend on context (ball near the other team, ball fast, gap start/end) fixes the event count. We can implement the team-level version (3 states H/A/D instead of 676 edges) without the neural backbone, using our existing evidence as emissions — 1 day. Training their full model on 5 matches in 2 days: possible code-wise (public repo) but labels for edges need events; IDSSE has events, Metrica has events, so feasible but risky.
