# Automatic event detection in football using tracking data

- **Authors:** F. Vidal-Codina, N. Evans, B. El Fakir, J. Billingham
- **Year / venue:** 2022, Sports Engineering 25 (article 18); preprint arXiv 2202.00804
- **URL / DOI:** https://doi.org/10.1007/s12283-022-00381-6 · https://arxiv.org/abs/2202.00804 · open PDF https://dspace.mit.edu/handle/1721.1/145347
- **Verified from:** search-result snippets only (full text not readable here; WebFetch blocked).

## Key method
- Two-step deterministic decision tree, no learning: (1) possession step (player-ball distance zones, the same paper MatchLens `possession.py` ports, R_pz / R_dz); (2) event step applying the Laws of the Game.
- The match is first split into **in-play vs dead-ball intervals**. The dead-ball flag is taken **from the data**: a boolean in-play flag (Tracab) or the absence of ball data (Track160, Hawk-Eye). The paper does not detect stoppages itself. It says that missing in-play/dead information is critical for set-piece detection.
- During each dead-ball interval, **set-piece triggers** record where the players and ball are (the "player configurations"). At the **first in-play frame** the ball must be within R_pz of the set-piece executor, and the triggers must still hold on the last dead frame. Flowchart for set pieces = Fig. 5. Exact trigger geometry and thresholds: **not found** in search text.
- Dead-ball event = attribute of the event just before the dead interval: out for corner / goal kick / throw-in, foul, penalty awarded, goal. Set piece = corner, goal kick, throw-in, free kick, penalty, kick-off.
- Three common tracking-error types are handled with mitigation rules. The supplementary material reports that these rules raised the mean set-piece F1 from **85% to 95%** (reported in one search summary, not checked against the supplement).

## Reported results
- Over 90% detection rate in most event categories, across tournaments and tracking providers. Benchmarked against manual tags, and human analysts disagree with each other by 10–15%.
- Confusions: free kicks near the corner flag tagged as corners, near the touchline as throw-ins, in the goal area as goal kicks. The authors suggest using ball z (for throw-ins), referee pose or video.

## Relevance to us
- The restart **executor** rule (ball within R_pz of one player at the first live frame) gives the restart team directly: the executor's team. We already have R_pz = 1.0 m from loop 2.
- The set-piece triggers are geometry we can turn around and use as **extra dead-ball evidence**: a player configuration that matches a set piece means the ball is dead.
- Caveat: the paper gets its stoppages for free from the provider flag. We have no flag, so its accuracy numbers do not carry over to our stoppage detection.
