# Predicting the in-game status in soccer with machine learning using spatiotemporal player tracking data

- **Authors:** S. Lang, R. Wild, A. Isenko, D. Link (TU Munich)
- **Year / venue:** 2022, Scientific Reports 12
- **URL / DOI:** https://doi.org/10.1038/s41598-022-19948-1 · PMC9522646 · PDF https://mediatum.ub.tum.de/doc/1831567/document.pdf
- **Verified from:** search snippets (abstract-level). Exact feature list and window lengths **not found**.

## Key method
- Predicts in play vs interrupted frame by frame from **player positions only, with no ball**. The hypothesis is that player speed, acceleration and orientation depend on the ball status.
- Models: logistic regression, decision tree, random forest, AdaBoost. Data: 102 Bundesliga matches with **manually annotated** in-game status. Trained on 45 matches and tested on unseen matches.
- Baseline: random guessing at the in-play share. Their "knowledge gain" measure compares against it.

## Reported results
- Up to **92% frame accuracy** on unseen matches.
- AdaBoost (best model): **81% precision** for stoppages longer than 2 s. The abstract text found does not give recall.
- Very short in-play or stoppage sequences are the hardest.
- Total distance computed with the predicted status: −102 ± 273 m (1.3%).
- Cites Gudmundsson & Wolle (F1 0.94 for ball-out detection, using ball and players).

## Relevance to us
- Works exactly where we are blind: **41% of our frames have no ball**, and deadball2 needs the ball for its far-from-players and near-line evidence. A players-only classifier gives D evidence on ball-gap frames.
- 81% precision is about our current 78%, so its value is the **recall it adds on frames without a ball**, not precision.
- Can be trained on the Metrica G1/G2 + IDSSE-1 tune split (truth D) with the swarm features (team speed collapse F3, spread).
