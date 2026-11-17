# TranSPORTmer: A Holistic Approach to Trajectory Understanding in Multi-Agent Sports

- Authors: Guillem Capellera, Luis Ferraz, Antonio Rubio, Antonio Agudo, Francesc Moreno-Noguer (Kognia Sports Intelligence / IRI CSIC-UPC) — author list beyond Capellera from memory of the group, not verified in search text
- Year / venue: 2024, ACCV 2024
- URL: https://arxiv.org/abs/2410.17785 ; https://openaccess.thecvf.com/content/ACCV2024/html/Capellera_TranSPORTmer_A_Holistic_Approach_to_Trajectory_Understanding_in_Multi-Agent_Sports_ACCV_2024_paper.html

## Key method
One transformer for forecasting, imputation, ball inference and per-frame state classification. Set Attention Blocks alternate over time and over agents (equivariant). An input mask hides missing / to-predict observations, so the same model handles "ball partly missing". An extra "CLS agent" outputs a per-frame game state: pass, possession, uncontrolled, out of play. Coarse-to-fine ball inference (+25% over prior SOTA on ball inference, per abstract). No ball tracking or player velocities needed at inference.

## Data needed
Soccer and basketball tracking with ball and per-frame state labels. Soccer dataset size and training hardware: not found in search text.

## Reported results
- Per-frame state classification with ball unobserved: ~80.8% accuracy (original paper, via secondary summary); 79.77% on the "Soccer Match Dataset" test split as reported by the follow-up paper (2512.19528); possessor identification 83.79% in that follow-up's table.
- "Uncontrolled" state is the hardest (subjective annotation).
- Per-class / Metrica breakdown: not found.

## Relevance to us
Closest published analogue of our task (game state per frame, ball partly or fully hidden, includes an out-of-play class). Two transferable ideas: (1) masking-based training, i.e. train on frames where the ball is artificially hidden in long blocks, exactly what our bench does; (2) treating "out of play" as a class predicted jointly with possession. Not trainable in 2 days on our 5 matches at full size (transformer, unknown data volume; no public code found). ~80% per-frame state accuracy also shows per-frame classifiers alone still need temporal decoding for spellF1.
