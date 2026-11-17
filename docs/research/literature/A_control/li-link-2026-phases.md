# Intention Driven Identification of In-Possession Match Phases in Association Football through Temporal Graph Learning

- Authors: Yuesen Li, Daniel Link
- Year / venue: 2026, arXiv preprint 2606.09289 (not peer reviewed)
- URL: https://arxiv.org/abs/2606.09289

## Key method
- Phases are recognised only inside a "stable ball possession"; stability = an Episode (Link & Lang): a stable possession lasting at least 3 s.
- Temporal Graph Attention Network for intention, then phase models; rule-based correction module; sliding-window filter. 7 Bundesliga matches, TRACAB 25 Hz.
- Related-work section credits Link & Hoernig 2017 with a rule-based approach inferring possession changes from ball acceleration patterns (unverified).

## Reported results
Frame macro-F1 0.87 (intention); post-processing raised sequence-level IoU-based F1 from 0.61 to 0.71 (phases) and 0.68 to 0.79 (intentions). Ablation: sequence modelling, not graph structure, drove segmentation quality.

## Relevance to us
- Their metric is close to spellF1 (segments matched by temporal IoU). Their biggest gain came from post-processing: sliding-window filter + rules on segments, +0.10 F1. Directly suggests a segment-level clean-up pass (merge or drop short possessions) after our frame rule.
- The 3 s Episode threshold is a published minimum for a "real" possession; useful prior for a minimum-duration constraint (HSMM) - but check our truth: Metrica/IDSSE truth possessions shorter than 3 s exist and must not be erased blindly.
