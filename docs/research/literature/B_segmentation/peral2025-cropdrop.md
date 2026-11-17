# Multi-Modal Soccer Scene Analysis with Masked Pre-Training

- Authors: Marc Peral, Guillem Capellera, Luis Ferraz, Antonio Rubio, Antonio Agudo
- Year / venue: arXiv Dec 2025; WACV 2026 (pp. 3536-3545)
- URL: https://arxiv.org/abs/2512.19528

## Key method
Transformer over player trajectories + player type + per-player image crops, for ball trajectory inference, ball state classification and ball possessor identification without ball positions. "CropDrop": masked pre-training that hides a player's visual crops over the whole sequence, so the model does not over-rely on images. Best: CropDrop on the visual modality only.

## Data needed
Large-scale proprietary professional match dataset (Kognia) with video crops. Size: not found.

## Reported results (from a search snippet of the comparison table; not verified in the PDF)
- Ball state accuracy 85.85% (vs TranSPORTmer 79.77%), possessor identification 89.56% (vs 83.79%), ADE 1.17, MaxErr 4.77 (units presumably m).
- Same group (VISAPP 2025, "Temporally Accurate Events Detection Through Ball Possessor Recognition in Soccer"): most events located within 0.6 s.

## Relevance to us
Low for implementation: needs video crops (we have none) and a large training set. Useful as an upper reference: players-only TranSPORTmer reaches ~80% state accuracy; images add ~6 points. Confirms that ball-state (incl. out of play) can be read from players at ~80% per frame, which is the order of magnitude a players-only dead-ball / team classifier could reach for us.
