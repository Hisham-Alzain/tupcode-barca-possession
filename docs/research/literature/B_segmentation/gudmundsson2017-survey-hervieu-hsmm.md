# Spatio-Temporal Analysis of Team Sports — A Survey (incl. Hervieu et al. hierarchical semi-Markov model)

- Authors: Joachim Gudmundsson, Michael Horton
- Year / venue: ACM Computing Surveys 2017 (arXiv 1602.06994, 2016)
- URL: https://arxiv.org/abs/1602.06994
- Cited work: Hervieu & Bouthemy (handball), hierarchical parallel semi-Markov model — original paper not opened; details only from the survey snippet.

## Key method (as described in the survey)
Phase segmentation of team-sport matches is a recurring task. Hervieu et al.: upper level is a semi-Markov model with one state per phase label; **state durations modelled by Gaussian mixtures**; lower level parallel HMMs over player trajectories. Survey also lists Perše et al. basketball segmentation.

## Data needed
Labelled player trajectories (handball). Size: not found.

## Reported results
Not found in search text.

## Relevance to us
Medium. The only sports precedent found for an **explicit-duration (HSMM)** phase model. For us an HSMM over {H, A, D} would encode that D lasts ~5-60 s and possessions have a heavy-tailed length; but HSMM Viterbi costs O(T x D_max) per state, which at 25 fps and D_max ~ 60 s is ~1,500x slower than HMM Viterbi; downsampling to 5 Hz makes it feasible (27,000 frames per match x 300 durations x 3 states ~ 2.4e7 steps, seconds in vectorised numpy). Recommendation: start with sticky HMM (Fox), move to HSMM only if duration errors dominate.
