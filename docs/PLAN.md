# Plan

## Task
199,319 frames → one of H / A / D / X per frame. ~159k graded (the two halves). Metric: spellF1 — predicted possessions matched one-to-one to truth by temporal overlap. Splitting kills the score. Baseline 0.24.
X inside a possession is free (scorer forward-fills the previous label); always label hand-overs.

## Pipeline (`src/possess/`)
1. **load** — unify tracking (fractional coords, per-file frame counting), match info (track → team), Smart Tagging, homography into one per-frame table.
2. **ball** — remove one-frame teleports; interpolate gaps ≤ ~0.5 s; longer gaps = unknown (NOT dead).
3. **touches** — direction/speed impulses (reuse `matchlens_ref/impulse.py`), nearest team; touch ≠ possession: require retention (team stays on it / next touch same team) to separate deflections from turnovers.
4. **shape** — team-level features needing no IDs: spread, compactness, centroid relative to ball/goal, pressing line height, team velocity toward ball. Classifier trained on external matches only.
5. **dead ball** — ball out of bounds, collective slowdown, ball static at restart spots, kickoff formation, Smart Tagging set pieces.
6. **tags** — Smart Tagging phases (with team) as soft evidence.
7. **fuse** — per-frame P(H), P(A), P(D); weight touches high when ball visible, shape+tags when not.
8. **segment** — Viterbi over {H, A, D} with switch penalties (HSMM if time: duration models). Whole-match decoding.
9. **finish** — snap possession starts to winning touch, D ends to restart touch; pre-2nd-half-kickoff = D; non-graded parts = X; write 199,319-row CSV.

## Evaluation
1. **Public 5 min** — format sanity only (offset, H/A swap, frame count). Not for comparing versions or thresholds. Not for debugging against video.
2. **External matches (the real validation set)** — Metrica G1/G2 (+G3), IDSSE ×2 (+more). Convert to H/A/D per frame; degrade to resemble challenge input (drop IDs, fragment tracks, position noise, hide ball ~41% biased to crowds/duels/long balls, optional noisy fake tags). Degradation calibrated on challenge *input* statistics only. Score with organisers' spellF1 code. Tune split vs held-out split; bootstrap CIs; ties → simpler version.
3. **Full-match no-key checks** (run locally by a human) — possession count (few hundred), length distribution, D vs tagged set pieces, Barça share, half-1 vs half-2 consistency.

## Schedule (deadline 2026-11-17 23:59 CET)
- D1: external labels + degradation + scorer running.
- D2–3: ball/touches/dead-ball/tags + Viterbi → first end-to-end submission.
- D4–5: shape model; tune switch costs & evidence weights on external.
- D6: full-match sanity checks.
- D7: buffer, submit.
