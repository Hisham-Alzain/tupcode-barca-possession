# An integrated dataset of spatiotemporal and event data in elite soccer (IDSSE)

- Authors: Manuel Bassek, Robert Rein, Hendrik Weber, Daniel Memmert
- Year / venue: 2025, Scientific Data 12, 195
- URL / DOI: https://doi.org/10.1038/s41597-025-04505-y ; PMC11787359 ; data figshare 10.6084/m9.figshare.28196177 ; repo github.com/spoho-datascience/idsse-data
- Contents: 7 Bundesliga 1/2 matches, official DFL metadata + event data + TRACAB position data at 25 Hz, CC-BY 4.0.

## How BallPossession / BallStatus are defined
- BallPossession: 1 = home team in possession, 2 = away team in possession. The paper says only that "ball possession is defined by the player in control of the ball".
- BallStatus: 0 = ball inactive, 1 = ball active.
- The derivation algorithm (rules, thresholds, manual vs automatic) is NOT published in the search text. Not found.
- Practitioner note (single informal GitHub issue, isaacboque/cota-cero-mtah2027 #18, unverified): the DFL possession flag has no published algorithm and STAYS SET during dead-ball periods; that team filtered possession changes to live frames lasting >= 1 s.
- Third-party users of IDSSE (FOOTPASS / others) use their own heuristic "closest player within 1.5 m is the carrier", not the DFL field.

## Reported results
Dataset paper; no possession-accuracy numbers.

## Relevance to us
- Our IDSSE truth = BallStatus + BallPossession. Because the flag persists through stoppages and its switching rule is unknown, short flag flips may exist that no control rule would produce; worth measuring the distribution of IDSSE possession lengths (511 truth possessions) before tuning to it.
- The 1.5 m carrier radius used by others on this exact data suggests our 1.0 m zone (tuned on Metrica, where the ball is often glued to the carrier) is too tight for TRACAB ball coordinates; consistent with IDSSE under-switching (286 vs 511).
