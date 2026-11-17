# Predicting Ball Ownership in Basketball from a Monocular View Using Only Player Trajectories (+ a-HCRF forecasting)

- Authors: Xinyu Wei, Long Sha, Patrick Lucey, Peter Carr, Sridha Sridharan, Iain Matthews
- Year / venue: 2015, ICCV Workshops
- URL: https://openaccess.thecvf.com/content_iccv_2015_workshops/w21/html/Wei_Predicting_Ball_Ownership_ICCV_2015_paper.html
- Related: Wei, Lucey, Vidas, Morgan, Sridharan 2014, "Forecasting Events using an Augmented Hidden Conditional Random Field", ACCV (soccer: predicts which player owns the ball after an interval).

## Key method
- Ball owner predicted from player trajectories only; multi-agent data "permuted" into a compact role-ordered feature (roles instead of identities).
- a-HCRF: local observation features + hidden states to forecast future ball owner.

## Reported results
"Accurately predicts the ball owner" (abstract); numbers not found.

## Relevance to us
- Earliest evidence for player-only ownership inference; role ordering is a way to build fixed-length features without player IDs (we have team labels only). Supports a player-only team classifier for ball gaps.
