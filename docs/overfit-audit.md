# Overfit audit of the 17 November default (L21, cold)

Question: is anything in the final default overfit? BASE = the 15 November system (`shape.DEC_SHAPE_L4`, tag flags off). FINAL = today's default (ADR-023, 024, 025, 026, 028, 029 on top).

**Verdict: keep FINAL.** On degradation seeds that nobody used today, FINAL beats BASE by **+0.049 spellF1 [CI90 +0.040, +0.058]** (controlled truth T=2; old truth +0.038 [+0.031, +0.046]). The ADRs claimed +0.057 in total, so about 15% of the claimed gain was selection noise. No step loses on fresh seeds. Two steps are weaker than claimed: the learned ball-team evidence (ADR-026) and the set-piece rule (ADR-029).

What this audit cannot show: the fresh seeds reuse the same three tune matches. They test seed selection, not whether the gain carries to a new match.

Script `scripts/overfit_audit.py`; numbers in `eval/results/overfit_audit_summary.json` and `overfit_audit_bench_*.json`. Held-out matches and the public truth were not touched.

## 1. Seed selection: fresh seeds

Fresh benches: real2 seeds 5, 6, 7; real1 seeds 2, 3; orig seeds 2, 3. Seeds used anywhere before (result files in `eval/results` and all worktrees): real2 0-3, real1 0-1, orig 0-1. Three tune matches. The two learned models are leave-one-match-out (the gate's own models: `a8_r6_models_pool3.json` for w_bt, a refit as in `a10_vdead.py` for w_bd). Set-piece tags are simulated as in `a12_setpiece.py` (sim `base`). Paired 2-minute block bootstrap, CI90. "avg" is the mean of the three bench types.

| Step (ctl T=2) | real2 fresh | real1 fresh | orig fresh | avg fresh | avg claimed | Verdict |
|---|---|---|---|---|---|---|
| ADR-023 T95 | +0.0111 [+0.005, +0.017] | +0.0016 [-0.001, +0.004] | +0.0015 [-0.001, +0.004] | +0.0048 [+0.002, +0.007] | +0.0055 | SOUND (see note) |
| ADR-024 T100 | +0.0127 [+0.005, +0.021] | +0.0119 [+0.006, +0.019] | +0.0188 [+0.012, +0.026] | +0.0145 [+0.009, +0.020] | +0.0154 | SOUND |
| ADR-025 T112+T113 | +0.0117 [+0.007, +0.017] | +0.0070 [+0.003, +0.011] | +0.0089 [+0.005, +0.014] | +0.0092 [+0.006, +0.013] | +0.0126 | SOUND |
| ADR-026 w_bt | +0.0097 [+0.002, +0.018] | +0.0022 [-0.003, +0.007] | +0.0008 [-0.004, +0.006] | +0.0042 [-0.000, +0.009] | +0.0081 | SHAKY |
| ADR-028 w_bd + ed_still | +0.0058 [+0.003, +0.009] | +0.0215 [+0.017, +0.026] | +0.0137 [+0.009, +0.018] | +0.0136 [+0.011, +0.017] | +0.0112 | SOUND |
| ADR-029 set-piece | +0.0006 [-0.001, +0.002] | +0.0058 [+0.003, +0.009] | +0.0012 [+0.000, +0.003] | +0.0025 [+0.001, +0.004] | +0.0047 | SHAKY (small, harmless) |
| **FINAL vs BASE** | **+0.0516** [+0.039, +0.064] | **+0.0500** [+0.039, +0.061] | **+0.0450** [+0.034, +0.057] | **+0.0488** [+0.040, +0.058] | +0.0575 | keep |

Claimed sums per bench type: real2 +0.0573 (fresh +0.0516, 10% less), real1 +0.0657 (fresh +0.0500, 24% less), orig +0.0493 (fresh +0.0450, 9% less). Per fresh bench, FINAL minus BASE: real2 +0.055 / +0.053 / +0.047, real1 +0.057 / +0.043, orig +0.046 / +0.044. Every one is positive.

Old truth, FINAL vs BASE: real2 +0.029, real1 +0.043, orig +0.042, avg +0.038 [+0.031, +0.046].

Note on T95: its gain exists only on the T=2 truth (old truth +0.0002). It is real if the organisers' "control" needs about 2 s, and zero if it needs less. That is a definition bet (ADR-022), not seed overfit.

## 2. Match-level

FINAL minus BASE per match (ctl T=2, fresh seeds):

| | Metrica 1 | Metrica 2 | IDSSE 1 |
|---|---|---|---|
| real2 | +0.051 | +0.069 | +0.035 |
| real1 | +0.067 | +0.048 | +0.035 |
| orig | +0.067 | +0.036 | +0.032 |

No match loses and no single match carries the gain. IDSSE 1 gains least (about half of Metrica 1). Single steps that lose on one match: w_bt on IDSSE 1 (real1 -0.0055, orig -0.0046), T112 never, set-piece on Metrica 2 real2 (-0.0013). Three matches from two providers is still a small sample; this is the main risk that remains.

## 3. The learned models

- Size: ballteam 21 weights and no intercept; balldead 21 weights plus an intercept. Logistic, L2 = 10 on standardised features. Each fold trains on every third visible frame of two matches on three benches (order 10^5 rows, strongly autocorrelated; the real unit count is two matches).
- LOMO against the committed (in-sample) constants on fresh seeds: w_bt +0.0042 vs +0.0044; w_bd + ed_still +0.0136 vs +0.0142. The gap is at most 0.0006, so the models do not memorise their training matches.
- Leakage: none through labels. Known and unchanged: some features look 0.2-0.6 s ahead (legal, the task is offline); the shape feature of training matches comes from a shape model that saw the test match (second order).
- One-sided provided match (59.6% home): not testable on external data. The ball-team model has no intercept, so it cannot hold a "home has the ball" prior; its features are signed home minus away.
- w_bt is shaky for another reason: the gain is on real2 only (+0.010); on real1 and orig it is not different from zero, and IDSSE 1 loses there.

## 4. Influence of the public clip

| # | What | Judgment |
|---|---|---|
| 1 | ADR-027: catalogue of error types on the clip, read by agents; input inspected at those frames | Hypothesis only. `scripts/a10_vdead.py` holds the four catalogue frame ranges (`CAT`) for a transfer printout; they are not in `src/` and choose nothing. |
| 2 | ADR-028 (w_bd, ed_still) | Constants were set in L8 rounds 6-7 on external data before the catalogue. The clip changed the decision from "keep off" to "on". The external gate passes without the clip and fresh seeds give more than claimed. Clean. |
| 3 | ADR-029 mechanism and constants | Spot geometry: Laws of the Game. Window -3 s..+15 s: from the provided INPUT inventory (restart found -1.7..+5.3 s from the tag start; no truth). 15 s fill: same inventory (our stoppage ended 7-15 s before the kick). On external benches 10, 15 and 20 s score the same on real2 and within 0.001 on real1, and lead ±2 s and jitter 0-3 s change nothing, so no value was picked by a score. Free-kick exclusion: an external failure (real2 seed 2, false kicks after quick free kicks). `hid=1, take_s=0`: best of five variants on external benches by 0.0006, which is noise. |
| 4 | ADR-029 adoption | **The one place where the clip touched a choice.** The rule failed the strict gate and was adopted as a recorded exception because the user saw the goal-kick error on the clip. The envelope also confirms that the rule repairs that case (frames 18320-18389, inside the clip). No constant depends on the clip, but the on/off decision does. Disclosed in TECHNICAL.md. Effect on the provided match: 699 frames (0.45%), all to D. |
| 5 | Public score 0.6353 -> 0.6588 after ADR-028/029 | Reported after adoption. It must not be read as validation: both changes target errors first seen on that clip, so a rise there is expected. |
| 6 | Earlier | 11 Nov: agents read slices with the public truth to learn what the labels mean (allowed by the organisers' email). Bench real2, the tag simulator's lead and rates: INPUT aggregates of the provided match, no labels. Public score reported for every version, not used to choose. |

Open defect, not clip-related: `setpiece.py` goal-area half-width 14.66 m instead of 9.16 m.

## 5. The gate: seven adoptions on the same benches

Using the fresh-seed bootstrap p-values (one-sided, avg of bench types) with a Holm correction over the six adoption steps (T112 and T113 were adopted as one): T100, T112+T113, w_bd, set-piece and T95 survive (p <= 0.0005). w_bt does not (p = 0.058 against 0.05). So five of six steps, six of seven changes, stand after correction. About 50 numbered candidates (T95-T141) were tried in L8; the fresh seeds are the answer to that selection, and they show 15% shrinkage, not collapse.

## 6. Held-out split (Metrica 3, IDSSE 2): not run

The project rule says no looks remain (three were used on 14 November). A single report-only look at FINAL vs BASE would add the one thing this audit lacks: two matches no constant or model has seen. It would tell us whether the +0.049 survives a change of match, and whether w_bt helps outside its training matches. It could not tell us how the system does on the provided match (same providers, same synthetic degradation, 59.6% home share not represented), nor whether the set-piece rule works with real tags (they would be simulated again). With two matches the CI90 would be about ±0.015, enough to confirm the total but not to rank single steps. If the user takes it, it must be after the default is frozen and reported whatever it shows.

## Recommendation

Keep FINAL. Expected gain over BASE on unseen data from the same kind of source: about +0.04 to +0.05 (claimed +0.057). No fallback is better: every step is non-negative on fresh seeds. If one step had to be dropped on evidence alone it would be w_bt (`--dec w_bt=0`, ADR-025 default plus ADR-028/029), but its fresh-seed mean is still positive, so dropping it is not advised.
