# A trajectory-based analysis of coordinated team activity in a basketball game (and earlier CVBASE06 segmentation work)

- Authors: Matej Perše, Matej Kristan, Stanislav Kovačič, Goran Vučkovič, Janez Perš
- Year / venue: CVIU 2009 (online 2008); CVBASE 2006 workshop
- URL: https://vision.fe.uni-lj.si/docs/matejp/CVIU_perse_2008.pdf ; https://vision.fe.uni-lj.si/RESEARCH/SportA/segmentation.html

## Key method
Segments a game into **offense / defense / time-out** from **player trajectories only** (no ball). Per-frame features of team configuration are modelled with Gaussian mixtures fitted by EM (one per phase); temporal continuity enforced with a smoothing kernel (Gaussian / nonlinear) over the frame-wise posteriors.

## Data needed
Player trajectories with team; a small amount of labelled play to fit the mixtures (exact size: not found).

## Reported results (lab page, from search snippet)
Segmentation accuracy 85-95% in basketball, 90-97% in handball. Handles exclusions, substitutions, injuries.

## Relevance to us
**High for hidden-ball frames and stoppages.** It is the closest precedent for "who is attacking + is play stopped" from team shape alone, with a deliberately simple model (GMM + temporal smoothing) that can be fitted in minutes on CPU. Caveat: basketball/handball have stronger offense/defense geometry (half-court) than football; football transitions are less clean, so expect lower accuracy. Directly supports survey candidate C2 (shape-based emission for ball-gap frames) and the "time-out" class supports using team-shape for D.
