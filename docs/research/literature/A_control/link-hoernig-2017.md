# Individual ball possession in soccer

- Authors: Daniel Link, Martin Hoernig
- Year / venue: 2017, PLOS ONE 12(7): e0179953
- URL / DOI: https://doi.org/10.1371/journal.pone.0179953 ; https://pmc.ncbi.nlm.nih.gov/articles/PMC5503225
- Precursor: Hoernig, Link, Herrmann, Radig, Lames 2016, "Detection of individual ball possession in soccer" (conference; precision/recall 80-92% on flawless raw data, per search text).

## Key method (from abstract and snippets)
- Hierarchy: Individual Ball Action (IBA) -> Individual Ball Control (IBC = IBA with control present) -> Individual Ball Possession (IBP). Team Ball Possession (TBP) = an unbroken sequence of IBP phases of one team.
- Convention: while the ball is in play, one team ALWAYS has team possession and one player always has individual possession. When individual possession switches between two players, the pass is attributed to the tactical intent of the first player (possession does not go to "nobody" during a loose ball).
- IBA start: the moment a player is able to act on the ball (and had no IBA before). IBA end: when the player is unable to interact with the ball for one second.
- "Naive physics" model: possession at time t if player-ball distance d < threshold T_p (value NOT found). A variant adding a prediction of the player's future position did not improve concordance.
- A machine-learning model estimates how long the ball is in a player's sphere of influence from player-ball distance, direction and speed of motion, and ball acceleration.
- A Bayesian network classifies the degree of control from: IBA duration, ball velocity/acceleration statistics, mean ball-to-player distance, counts of opponents within 0.5 m, 1 m, ..., 5 m. Trained on second halves.
- Related work in a 2026 paper describes Link & Hoernig as also having a rule-based team-level approach inferring possession changes from ball acceleration patterns (unverified detail).

## Reported results
Evaluation on 60 Bundesliga matches (season stated as 2012/13 or 2013/14 by different sources), 69,667 IBA intervals. Precision/recall values: NOT found in search text.

## Relevance to us
- Directly supports "never leave a live frame unlabeled; a loose ball stays with the last team" (fixes failure paths 3 and 4: Pending loss timeout, Duel without holder).
- The 1-second "unable to interact" end rule is a principled alternative to our 3 s Pending timeout and k-frame counting.
- Control vs touch: separates mere action (touch) from control (duration in sphere of influence + low opponent pressure). A feature set we can compute without player IDs: team-level opponent counts in rings around the ball.
