# Identifying Team Style in Soccer using Formations Learned from Spatiotemporal Tracking Data / Large-Scale Analysis of Soccer Matches / Discovering Team Structures in Soccer from Spatiotemporal Data

- Authors: Alina Bialkowski, Patrick Lucey, Peter Carr, Yisong Yue, Sridha Sridharan, Iain Matthews
- Year / venue: ICDM Workshops 2014; ICDM 2014; IEEE TKDE 2016
- URL: https://la.disneyresearch.com/wp-content/uploads/Identifying-Team-Style-in-Soccer-using-Formations-from-Spatiotemporal-Tracking-Data-Paper.pdf ; TKDE: Caltech record https://resolver.caltech.edu/CaltechAUTHORS:20170721-142537932

## Key method
Role assignment: each frame, players are assigned to formation roles by minimising entropy of role-specific occupancy maps (Hungarian assignment to a learned template, EM-style). Gives a per-frame "formation-aligned" team description without player IDs.

## Data needed
A season of Prozone tracking (TKDE: >21.5 M frames). Role assignment itself is unsupervised and can be fitted per match.

## Reported results
Team identity recognised from formation descriptors far better than from event stats. Possession-specific results: not found in search text (in the ICDM 2014 paper formations were reportedly computed separately for in- and out-of-possession phases; not verified).

## Relevance to us
Medium-low. Role assignment is an ID-free way to get stable per-role features (e.g. full-back height, centre-back line), which could make shape features robust to the player-count drops we see in gaps (swarm F7). But it is a feature-engineering step with no direct possession result; simpler count-normalised robust statistics (median line heights, spread) are cheaper. Not a priority for 5 days.
