# Does the method generalise? (L24, cold)

Question (user): "check if anything can be generalised; if it is not generalised enough, it is overfit." The L21 audit (`docs/overfit-audit.md`) answered seed selection and model memorisation. This one tests the axes it could not: data source, degradation condition, label convention, and learned models trained on the other source.

BASE = `shape.DEC_SHAPE_L4`, tags off. FINAL = today's default. Steps are cumulative (T95, T100, T112+T113, w_bt, w_bd+ed_still, set-piece), same leave-one-match-out learned models and simulated set-piece tags as L21. Score: spellF1, mean of the three tune matches unless a source is named.

**Verdict: keep FINAL.** FINAL beats BASE in all 7 runs, on all 3 matches, on all 4 truths (21 of 21 match cells at T=2; smallest single cell on any truth +0.002). Under shifted conditions the gain stays +0.044 to +0.067 (standard: +0.055). Three changes are general, three depend on a condition, none is shown to be harmful.

Scripts `scripts/gen_audit_stress.py`, `scripts/gen_audit_agg.py`; numbers in `eval/results/gen_audit_*.json` (`gen_audit_summary.json` holds every table). Held-out matches and the public truth were not touched.

## 1. Sources: Metrica (2 matches) vs IDSSE (1 match)

From the L21 fresh-seed benches (7 benches, ctl T=2, bench types weighted equally). "neg" = match-by-bench cells below zero.

| Step | Metrica | neg | IDSSE 1 | neg |
|---|---|---|---|---|
| T95 | +0.0060 | 0/6 | +0.0023 | 1/3 |
| T100 | +0.0159 | 0/6 | +0.0116 | 0/3 |
| T112+T113 | +0.0104 | 0/6 | +0.0067 | 0/3 |
| w_bt | +0.0064 | 0/6 | -0.0000 | 2/3 |
| w_bd+ed_still | +0.0155 | 0/6 | +0.0099 | 0/3 |
| set-piece | +0.0022 | 1/6 | +0.0032 | 0/3 |
| **FINAL - BASE** | **+0.0564** | 0/6 | **+0.0338** | 0/3 |

The total is present in both sources; IDSSE gets 60% of the Metrica gain. Five steps gain in both. **w_bt is carried by Metrica**: zero on IDSSE 1 on average (+0.010 on real2, -0.005 on real1 and orig). IDSSE is one match, so every IDSSE figure here has n = 1.

## 2. Conditions: unseen degradations

real2 bench, seed 5, rebuilt in memory with one parameter shifted (inputs cached as `data/external/_bench/gen_<tag>_*`). Control = the same seed with the standard parameters (reproduces L21 to four decimals). ctl T=2; CI90 = paired 2-minute block bootstrap of FINAL - BASE.

| Condition | hidden | BASE | FINAL | FINAL - BASE [CI90] | T95 | T100 | T112+T113 | w_bt | w_bd | set-piece |
|---|---|---|---|---|---|---|---|---|---|---|
| standard | 41% | 0.629 | 0.684 | +0.055 [+0.039, +0.071] | +0.014 | +0.013 | +0.011 | +0.011 | +0.007 | -0.000 |
| ball missing 20% | 21% | 0.612 | 0.679 | +0.067 [+0.049, +0.085] | +0.011 | +0.013 | +0.019 | +0.009 | +0.014 | -0.000 |
| ball missing 60% | 61% | 0.581 | 0.637 | +0.056 [+0.041, +0.071] | +0.008 | +0.014 | +0.026 | +0.002 | +0.002 | +0.004 |
| hidden blocks 1-3 s only | 42% | 0.596 | 0.651 | +0.055 [+0.038, +0.075] | +0.017 | +0.014 | +0.005 | +0.007 | +0.011 | +0.001 |
| raw ball, jitter x2 | 41% | 0.675 | 0.719 | +0.044 [+0.028, +0.059] | +0.007 | +0.014 | +0.008 | +0.007 | +0.011 | -0.002 |

Smallest single-match FINAL - BASE: +0.047 / +0.033 / +0.038 / +0.035 (20% / 60% / short / raw). FINAL wins everywhere.

Steps under shift:
- **T100** is the most stable step: +0.013 to +0.014 in every condition.
- **T112+T113** grows with the hidden share (+0.019 at 20%, +0.026 at 60%) and shrinks when blocks are short (+0.005): its size depends on how stoppages are hidden, its sign does not.
- **w_bd+ed_still** nearly vanishes at 60% hidden (+0.002): it works on the visible ball only.
- **w_bt** stays positive on the three-match mean (+0.002 to +0.011) but flips by source: IDSSE 1 -0.001 (20%), -0.002 (short); Metrica -0.004 (raw ball). 7 of 21 match cells are negative across the 7 runs (worst -0.008).
- **set-piece** is the only step with a negative mean under a shift: -0.002 with a raw ball (Metrica -0.003), -0.0002 at 20% missing. 5 of 21 match cells negative (worst -0.008), mean over runs +0.001.

## 3. Label conventions

FINAL - BASE on each truth (old = provider labels; ctl T = controlled-possession relabel).

| Run | old | T=1 | T=2 | T=3 |
|---|---|---|---|---|
| real2 s5 | +0.034 | +0.040 | +0.055 | +0.067 |
| real1 s2 | +0.049 | +0.053 | +0.057 | +0.064 |
| orig s2 | +0.042 | +0.045 | +0.046 | +0.052 |
| missing 20% | +0.037 | +0.040 | +0.067 | +0.082 |
| missing 60% | +0.041 | +0.046 | +0.056 | +0.065 |
| short blocks | +0.031 | +0.040 | +0.055 | +0.072 |
| raw, jitter x2 | +0.032 | +0.039 | +0.044 | +0.047 |
| **mean** | **+0.038** | **+0.043** | **+0.054** | **+0.064** |

FINAL does not win only at T=2: about 70% of the T=2 gain remains on the old truth and 80% at T=1. Smallest single match on any truth: +0.002 (short blocks, old truth).

Per step, mean over the 7 runs (runs below zero in brackets):

| Step | old | T=1 | T=2 | T=3 |
|---|---|---|---|---|
| T95 | +0.0004 (3) | +0.0033 (0) | +0.0088 (0) | +0.0136 (0) |
| T100 | +0.0131 (0) | +0.0142 (0) | +0.0150 (0) | +0.0145 (0) |
| T112+T113 | +0.0128 (0) | +0.0129 (0) | +0.0113 (0) | +0.0118 (0) |
| w_bt | +0.0011 (2) | +0.0012 (2) | +0.0059 (0) | +0.0117 (0) |
| w_bd+ed_still | +0.0097 (0) | +0.0106 (0) | +0.0120 (0) | +0.0117 (0) |
| set-piece | +0.0009 (1) | +0.0010 (3) | +0.0011 (3) | +0.0008 (3) |

**T95 and w_bt are bets on the T=2 reading.** If the organisers' control needs under 2 s they are worth about +0.001 to +0.003 together, not +0.015. T100, T112+T113 and w_bd do not depend on the convention.

## 4. Leave-one-source-out for the learned models

ballteam and balldead refitted on IDSSE 1 only (three gate benches; 58,560 and 86,531 rows) and used on Metrica 1 and 2. The reverse case already exists: the leave-one-match-out model of IDSSE 1 is Metrica-only. ctl T=2, Metrica matches.

| | w_bt, LOMO | w_bt, IDSSE-only | w_bd, LOMO | w_bd, IDSSE-only | FINAL - BASE, LOMO | IDSSE-only |
|---|---|---|---|---|---|---|
| 3 standard runs | +0.0049 | +0.0013 (2/6 neg) | +0.0175 | +0.0196 (0/6 neg) | +0.0592 | +0.0578 |
| 4 shifted runs | +0.0055 | +0.0041 (3/8 neg) | +0.0103 | +0.0111 (1/8 neg) | +0.0564 | +0.0558 |

Metrica-only models on IDSSE 1: w_bt +0.0072 / +0.0074 (standard / shifted; negative in 2 of 7 runs), w_bd +0.0108 / +0.0082 (0 of 7 negative).

- **balldead transfers across sources in both directions** with no loss.
- **ballteam does not transfer cleanly**: trained on the other source it keeps a quarter of its gain on standard benches (+0.0013) and is negative in 5 of 14 Metrica cells. FINAL as a whole loses only 0.001 from the swap, because the step is small.

## 5. Ranking and verdicts

| Rank | Change | Verdict | Why |
|---|---|---|---|
| 1 | T100 flight direction (ADR-024) | GENERAL | +0.013 to +0.022 in all 7 runs, both sources, all four truths; 1 of 21 match cells below zero (-0.0003) |
| 2 | w_bd + ed_still (ADR-028) | GENERAL | positive in 21 of 21 match cells, all truths, survives the source swap; small when 60% of the ball is hidden |
| 3 | T112 line-crossing entry (ADR-025) | GENERAL | positive mean in all runs and truths; size follows the hiding pattern |
| 4 | T113 off-pitch dead ball (ADR-025) | GENERAL | adopted and measured with T112 as one step; not separated here |
| 5 | T95 stricter control (ADR-023) | CONDITION-DEPENDENT | never negative at T>=1, zero on the old truth (3 of 7 runs below zero): a definition bet |
| 6 | w_bt learned ball team (ADR-026) | CONDITION-DEPENDENT | gain only at T>=2, carried by Metrica, a quarter left after the source swap, 7 of 21 match cells negative; three-match mean never negative at T=2 |
| 7 | set-piece rule (ADR-029) | CONDITION-DEPENDENT | mean +0.001, sign changes with condition (-0.002 raw ball); simulated tags, so generality to real tags is untested |

NOT GENERAL: none. No step has a negative mean on a whole source at T=2 on the fresh-seed benches, and under shifted conditions the only negative three-match means are the set-piece rule's (-0.0002 at 20% missing, -0.002 with a raw ball).

**Switch off for safety: nothing.** The one candidate is `w_bt=0`. Removing it would cost about 0.006 at T=2 and nothing at T<=1 or on the old truth; left on, its worst three-match mean on any truth is -0.003. That is too small a risk to justify changing a frozen default hours before the deadline.

## 6. Limits of this audit

- Each shifted condition is one seed. Single steps vary by about 0.005 between seeds (ADR-026: +0.002 to +0.019), so step cells inside ±0.005 are not separable from zero. Only FINAL - BASE has a confidence interval here.
- "Jitter x2" is mild: ball white noise 2 -> 4 mm and teleports 1 -> 2 per minute, on an unsmoothed ball. Player-position jitter was not tested.
- The shape model inside both learned models stayed leave-one-match-out (not refitted by source).
- IDSSE is one match. Three matches from two providers cannot show match-to-match spread.

## 7. What the single held-out look would add (Metrica 3 + IDSSE 2; not run)

One thing these tests cannot give: **matches that no constant, model or decision has seen.** In particular a second IDSSE match, which turns every n = 1 IDSSE statement above into n = 2, and a clean reading of w_bt outside its training matches. It would not add anything on conditions or label conventions (same providers, same synthetic degradation, tags simulated again), and with two matches its CI90 (about ±0.015) can confirm the total but not rank steps.

## Recommendation

Keep FINAL unchanged. Report in TECHNICAL.md that T95 and w_bt pay only if "control" needs about 2 s, and that the set-piece rule is unproven on external data.
