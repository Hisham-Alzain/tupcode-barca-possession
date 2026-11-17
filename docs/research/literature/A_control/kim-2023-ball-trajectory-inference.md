# Ball Trajectory Inference from Multi-Agent Sports Contexts Using Set Transformer and Hierarchical Bi-LSTM

- Authors: Hyunsung Kim, Han-Jun Choi, Chang Jo Kim, Jinsung Yoon, Sang-Ki Ko
- Year / venue: 2023, KDD 2023 (arXiv 2306.08206)
- URL: https://arxiv.org/abs/2306.08206

## Key method
- Input: player trajectories only. Set Transformers give permutation-invariant/equivariant representations of the multi-agent context; a hierarchical Bi-LSTM first predicts player ball possession as an intermediate target, then the ball trajectory.
- Reality loss + postprocessing keep the trajectory physically plausible.
- Listed applications: missing trajectory imputation, semi-automated pass annotation, possession-wise running metrics.

## Reported results
"Natural and accurate trajectories as well as admissible player ball possession" (abstract); numeric values not found in search text.

## Relevance to us
- Same message as PathCRF: possession is the intermediate variable from which the ball is inferred, so we should infer possession directly during gaps rather than interpolating the ball first (cf. docs/research/ball-interpolation.md).
- Bi-directional (offline) modelling is allowed for us: the whole match is labelled after the fact, so gap decisions can use frames after the gap.
