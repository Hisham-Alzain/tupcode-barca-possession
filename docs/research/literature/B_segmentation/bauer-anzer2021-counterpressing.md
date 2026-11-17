# Data-driven detection of counterpressing in professional football

- Authors: Pascal Bauer, Gabriel Anzer
- Year / venue: 2021, Data Mining and Knowledge Discovery 35:2009-2049
- URL / DOI: https://link.springer.com/article/10.1007/s10618-021-00763-7

## Key method
Supervised: 134 hand-crafted features from positional + event data around defensive transitions (distances to ball carrier, pressing players, team compactness, etc.), XGBoost classifier.

## Data needed
>20,000 manually labelled defensive transitions from 97 Bundesliga matches; positional AND event data (event data marks the turnover).

## Reported results
AUC 87.4% on held-out labelled transitions; applied to six Bundesliga seasons.

## Relevance to us
Low-medium. It classifies *what happens after* a known turnover, which presupposes the possession timeline we are trying to infer. Transferable pieces: (1) pressing features (number of opponents within 5-10 m of the ball's last position / the presumed carrier, defenders' velocity towards the ball) as evidence that a turnover just happened and the other team is pressing, useful as emissions near gap starts; (2) the recipe "hand features + gradient boosting" trains in minutes on CPU, the right tool class for our data size. Data volume (97 matches) is far beyond ours, so expect less.
