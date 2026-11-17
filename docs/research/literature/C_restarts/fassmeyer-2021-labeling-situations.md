# Toward Automatically Labeling Situations in Soccer

- **Authors:** D. Fassmeyer, G. Anzer, P. Bauer, U. Brefeld
- **Year / venue:** 2021, Frontiers in Sports and Active Living 3:725431
- **URL / DOI:** https://doi.org/10.3389/fspor.2021.725431
- **Verified from:** search snippets (abstract).

## Key method
- Variational autoencoders (4 architectures) learn features from **unlabelled** tracking data. A large-margin classifier is then trained on a few labelled examples. Tasks: corner kicks, crosses, counterattacks. Labels for corners and crosses come from event data. Data from the DFB.

## Reported results
- **AUC > 0.90** on every task. Per-task numbers: not found.

## Relevance to us
- This is the Anzer & Bauer line of work on set-piece detection. It shows that corners can be detected from tracking alone, but it detects windows and gives no exact start or end frames, and needs a VAE. Too heavy for the remaining time. Low priority.
- Related Anzer & Bauer work on synchronising shots with tracking data is mentioned by Biermann et al. 2023 (Sci Rep, doi 10.1038/s41598-023-39616-2). It is not about stoppages.
