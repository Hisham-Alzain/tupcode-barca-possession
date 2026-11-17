# Possession fix: research synthesis and plan (2026-11-12)

Inputs: `literature/A_control/survey.md` (15 sources), `literature/B_segmentation/survey.md` (17), `literature/C_restarts/survey.md` (13), the state-machine page (`docs/possession-state-machine.html`), bench oracles (LOOP.md). Method adapted from Orchestra Research's autoresearch / ideation / rigor-review skills (read, not installed).

## Rigor notes before trusting anything
- Every literature number comes from search snippets (page fetching is blocked here); thresholds inside papers were mostly **not found**. All "predictions" below are planning guesses until the bench measures them.
- No provider publishes its per-frame owning-team rule (DFL/IDSSE, Metrica, SkillCorner, Second Spectrum, TRACAB, Hawk-Eye: not found). Our Vidal-Codina / Link-Hoernig thresholds are unchecked against the papers.
- PathCRF is reported as trained on IDSSE, which is our held-out split: use the idea, never the weights.
- Public repos from other teams on this same challenge appeared in search results; not read, not used.
- Verified by us: (a) `team_control.py` commits a switch only at the k-th control frame and never back-dates it; (b) **after a stoppage the other team owns the ball 82% / 78% / 90% of the time** (Metrica 1, Metrica 2, IDSSE 1 truth), while our pipeline keeps the pre-stoppage team.

## Where each failure from the state machine gets its fix
| Failure (state-machine page) | Fix | Source |
|---|---|---|
| Restart keeps the old team (measured: wrong ~85%) | Prior "other team restarts", then geometry: goal kick → defending team, corner → attacking team, throw-in/free kick → team standing at the still ball, kick-off → team that conceded | Our measurement; IFAB Laws 8, 13, 15–17; Vidal-Codina restart taker |
| Switch dated 4+ frames late | Back-date a committed switch to the challenger's first control frame | Event conventions (Opta/Metrica); Vidal-Codina back-dates losses |
| Frame-by-frame flicker and late/early switches | Whole-match Viterbi over {H, A, D}: zone evidence + dead-ball evidence as emissions, switch cost, minimum durations, D→other-team transition prior | PathCRF (KDD 2026), sticky HMM (Fox 2011), Li & Link 2026 (segment F1 0.61/0.68 → 0.71/0.79 with segment clean-up) |
| Too few switches on IDSSE (286 vs 511) | Zone radius per source / context-dependent switch cost | Vidal-Codina (0.5 or 1 m by provider); PathCRF |
| Turnover hidden in a 5–30 s ball gap | Team-shape classifier (relative, count-normalised features, gradient boosting on CPU) as emission on hidden frames | Perše et al.; Ballradar (85.8% from players only); PathCRF (92.4% per frame) |
| Missed / late stoppages | Set-piece geometry as dead evidence, held until the kick; start D at the line crossing; player-only stoppage classifier on hidden frames | Vidal-Codina; IFAB; Mills et al. 2026; Lang et al. 2022 (92% frame accuracy) |
| Contact vs distance | Control only when the ball is slow relative to the nearest player | Link & Hoernig 2017 |

## Plan, in build order (each step = one loop iteration on the bench, tune split only)
| # | Hypothesis | Prediction (tune spellF1, now 0.375) | Cost |
|---|---|---|---|
| P1 | Restart prior: after D, start the other team | + clearly; restart-team accuracy ~15% → ~85% | 10 lines |
| P2 | Back-date switches to the first control frame | +0.003 to +0.01 | 5 lines |
| P3 | Viterbi H/A/D decoder replacing the frame rules (P1, P2 become transition/emission terms) | +0.02 to +0.05; fewer fragments | 0.5–1 day |
| P4 | Restart team from geometry (Laws) as emission at D→live | +0.01 to +0.03 over P1 | 0.5 day |
| P5 | Team-shape classifier for hidden-ball frames, into P3 | +0.02 to +0.05 (closes 10–30% of the hidden gap) | 1–1.5 days |
| P6 | Stoppage evidence: set-piece geometry, line-crossing start | +0.03 to +0.04 | 0.5 day |
| P7 | Per-source zone radius / context switch cost | IDSSE count toward 511 | grid only |
| — | Learned sequence model (Ballradar-lite) | only if P3–P6 stall; overfit risk on 5 matches | 2 days |

Deadline 17 Nov: P1–P3 first (one day), then P5 and P6 in parallel, validated submission kept current from 16 Nov.
