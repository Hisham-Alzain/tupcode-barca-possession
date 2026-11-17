# Toward Automatically Labeling Situations in Soccer

- Authors: Dennis Fassmeyer, Gabriel Anzer, Pascal Bauer, Ulf Brefeld
- Year / venue: 2021, Frontiers in Sports and Active Living 3:725431
- URL: https://www.frontiersin.org/articles/10.3389/fspor.2021.725431 ; PDF https://pubdata.leuphana.de/bitstream/20.500.14123/1615/1/Fassmeyer_Toward_Automatically_Labeling_Situations_in_Soccer.pdf

## Key method
Variational autoencoders trained on large unlabelled tracking data give a representation of a situation; a large-margin classifier (SVM-like) on top needs only a few labelled examples. Four VAE architectures compared.

## Data needed
Large unlabelled DFB tracking corpus + few labels per situation type.

## Reported results
>90% AUC for corner kicks, crosses and counterattacks.

## Relevance to us
Low-medium. Corner-kick detection from tracking at >90% AUC supports reading set pieces (dead ball + restart team) from player configuration, which is cluster C. The semi-supervised VAE route is not feasible for us in 5 days (needs a large unlabelled corpus and GPU time); the takeaway is that set-piece configurations are distinctive enough that even simple features should catch many of them.
