# Automatic detection of tactical states in football based on player and ball position

- Authors: Frederic Rothe, Maaike Van Roy, Guilherme Pinheiro, Jesse Davis, Martin Lames
- Year / venue: 2026, journal with DOI prefix 10.1177/2215... (likely Journal of Sports Analytics — not verified)
- URL / DOI: https://doi.org/10.1177/22150218261432789

## Key method (abstract only)
Rule-based expert system of tactical states covering in-possession and out-of-possession phases; then a detector from positional data: player configurations are **clustered within each ball zone**, and a translation table maps clusters to expert states. Table learned from **three training games**, applied to new matches.

## Data needed
Player + ball positions; 3 labelled games for the translation table. A later arXiv paper says the expert system was validated against human annotation on 12 Bundesliga matches (not verified).

## Reported results
F1 > 0.9 for most states; beats three baselines. Per-state numbers: not found.

## Relevance to us
Medium. Shows that with only 3 training games, configuration clustering conditioned on ball zone recovers in/out-of-possession states well — but it *uses the ball zone*, which we lack in gaps. Variant for us: condition on last-seen ball zone at gap start. The "3 games is enough for a lookup-style model" result is encouraging for our 5-match training set.
