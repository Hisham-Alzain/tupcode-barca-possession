# Semi-automated last touch detection for out-of-bounds possession decisions in football

- Authors: Wang, Mills, Billingham, Robertson, Hosoi
- Year / venue: 2025, Sports Engineering 28, art. 36
- URL / DOI: https://doi.org/10.1007/s12283-025-00518-3 ; shura.shu.ac.uk/36072

## Key method
- Stage 1: a touch-probability model on ball tracking data yields a decision frame (instant of last touch).
- Stage 2: rule-based or supervised attribution of the touching player.
- Data: 2022 FIFA World Cup, connected ball (Kinexon IMU) available; operator may verify.
- Deflection-specific handling: not found.

## Reported results
Up to 82.5% attribution accuracy on duels; > 90% for aerial duels.

## Relevance to us
- Even with an instrumented ball and full tracking, attributing a touch in a duel is ~82% correct; from our noisy, ID-free, often ball-less data a touch-based gate cannot be reliable. Argues against making touch evidence a precondition for a gain (MatchLens failure paths 1-2) and for using it only as soft evidence.
- Relevant to restarts (cluster C): last touch before out-of-bounds decides who restarts - the restart team is the opponent of the last-touch team.
