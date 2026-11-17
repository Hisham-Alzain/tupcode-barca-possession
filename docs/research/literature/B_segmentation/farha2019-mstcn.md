# MS-TCN: Multi-Stage Temporal Convolutional Network for Action Segmentation

- Authors: Yazan Abu Farha, Juergen Gall (MS-TCN++: Li, Abu Farha, Liu, Cheng, Gall)
- Year / venue: CVPR 2019; MS-TCN++ TPAMI 2020
- URL: https://arxiv.org/abs/1903.01945 ; https://arxiv.org/abs/2006.09220

## Key method
Stacked stages of dilated temporal convolutions, each refining the previous stage's frame-wise class probabilities. Loss = cross-entropy + truncated MSE "smoothing loss" on frame-to-frame log-probability differences, which reduces over-segmentation more than a KL term.

## Data needed
Frame-wise labelled sequences (they used 50Salads, GTEA, Breakfast; tens to thousands of videos). No sports tracking application found.

## Reported results
SOTA frame accuracy, edit score and segmental F1@{10,25,50} on the three datasets at publication. Segmental F1@IoU is close in spirit to our spellF1.

## Relevance to us
Medium-low within 5 days. If we train any sequence model on per-frame features (Metrica + IDSSE, ~5 matches x 90 min), a small single-stage TCN with the smoothing loss is cheap on CPU and directly targets over-segmentation. But a rule/GBM emission model + Viterbi achieves the same anti-fragmentation effect with no training risk; MS-TCN is a fallback, not a first step. Sports application: not found.
