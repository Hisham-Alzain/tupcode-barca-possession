# A sticky HDP-HMM with application to speaker diarization

- Authors: Emily B. Fox, Erik B. Sudderth, Michael I. Jordan, Alan S. Willsky
- Year / venue: 2011, Annals of Applied Statistics (arXiv 0905.2592, 2009)
- URL: https://arxiv.org/abs/0905.2592

## Key method
Adds a self-transition bias kappa to each state's transition prior. Without it, the HMM switches rapidly among states (over-segmentation) because a first-order HMM models state persistence poorly. A later NeurIPS paper notes kappa ~ 100 gives long segments while avoiding the cost of semi-Markov (explicit-duration) models.

## Data needed
Unsupervised; nothing sport-specific.

## Reported results
State-of-the-art NIST speaker diarization at the time; plain HDP-HMM was not.

## Relevance to us
Speaker diarization is structurally our problem: who "holds the floor" (H/A), silence (D), unobserved stretches, scored by segment matching. Takeaway: a single self-transition bias, tuned so that the expected segment length matches truth (possessions ~10 s at our frame rate, so p_stay ~ 1 - 1/(10 s x fps)), is the cheap substitute for an HSMM. A geometric duration is a poor model of possession length (many very short ones + long ones), but under Viterbi the switch penalty is what matters, and it is one scalar to tune on the tune split.
