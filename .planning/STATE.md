# STATE — barca-possession

**Updated:** 2026-11-17 (deadline today 23:59 CET)
**Current phase:** final submission; all experiments finished.
**Final draft (user decision):** `submissions/draft_2026-11-17f.csv` = output of the default (`shape.DEC_SHAPE` + `pipeline.TAGS`, no flags); 38 frames differ from 17e (20 A->H, 18 H->A), possessions 326 -> 330. 17e differed from 17d on 1,839 frames.
**Default = ADR-022 + ADR-023 (T95 `ctl_hi=2.0, ctl_touch=1`) + ADR-024 (T100 `w_fly=0.7`, ramp 4->8 m/s) + ADR-025 (T112+T113 `ed_cross=5.0, ed_cross_strict=1, ed_xdead=1, ed_xdead_min=2.0`) + ADR-026 (T125 `w_bt=0.5`, `ballteam.py`) + ADR-028 (`w_bd=1.0`, `balldead.py`; `ed_still=1.0`) + ADR-029 (`sp=1 sp_hid=1 sp_take_s=0`, `setpiece.py`; RECORDED GATE EXCEPTION) + ADR-030 (`ctl_s` 1.0 -> 0.6, hedge across label readings, NOT a gate pass, user decision).** ADR-027: public-clip error types used for diagnosis only (disclosed in TECHNICAL §10).

## Public 5 min (reported only, never used to choose)
17f (final): spellF1 0.6588, macro F1 0.8886, 17 predicted vs 17 true (organisers' `run_evaluation.py`), unchanged from 17e. Earlier: 17d 0.6353, 15d 0.5684, 17c 0.5231.

## Gates (ADR-016/022 rule, controlled truth T=2, delta vs previous default; real numbers from ADRs)
| Step | real2 | real1 | orig | Held-out | Gate |
|---|---|---|---|---|---|
| ADR-023 T95 | median +0.011 (s0 [+0.0009, +0.0136]) | +0.0017 | +0.0035 | not measured | pass |
| ADR-024 T100 | s0 +0.0133, s1 +0.0204 | +0.0159 | +0.015 | not measured | pass, both truths |
| ADR-025 T112+T113 | s0 +0.0170 [+0.0105, +0.0242], s1-s3 +0.0140/+0.0118/+0.0136 | +0.0105 [+0.0052, +0.0161] | +0.0132 | not measured | pass |
| ADR-026 T125 | median +0.0133; pooled +0.0120 [+0.0044, +0.0193] | +0.0089 [+0.0019, +0.0158] | +0.0034 | not measured | pass (per-seed reading) |
| ADR-028 w_bd+ed_still | s0/s1/s2 +0.0033/+0.0006/+0.0042 | +0.0187 [+0.012, +0.026] | +0.0121 [+0.007, +0.018] | not measured | pass (cold QA ADOPT) |
| ADR-029 set-piece (simulated tags) | s0 +0.0030, s1 +0.0016, s2 -0.0005, s3 +0.0033 | +0.0100 | +0.0021 | not measured | STRICT FAIL; recorded exception |
| ADR-030 ctl_s 0.6 (5 benches pooled) | old truth +0.0050*, T=1 +0.0010, T=2 -0.0015, T=3 -0.0027*; worst-case loss 0.0102 -> 0.0052 | | | not looked at | not a gate pass; user decision (insurance) |

## Open requests (all the user's)
- Fill `<TEAM NAME>` / `<PUBLIC REPO URL>` placeholders (README, TECHNICAL).
- PDF export of `docs/TECHNICAL.md`.
- Push to the public repo (never push without asking).
- Submission form.
- smooth_touch / sd_prior: no organiser answer received before submission; both stay on (TECHNICAL §8).

## Done
- All experiments finished (L22-L30; rejected ones listed in ADR-030 and TECHNICAL §4). Final-day robustness audits: docs/overfit-audit.md, docs/generality-method-audit.md, docs/generality-code-audit.md (disclosed in TECHNICAL §8, §9).
- ADR-029 open defects (goal-area half-width 14.66 vs 9.16, throw-in outer bound, rest-vs-reappear order, 2-frame H spell) not fixed; disclosed.

## Standing user decisions
- T98, T120, T137: off. Public 5 min: reported, never used to choose or tune. No tuning on challenge data.

## Risks
- Over-calling D on hidden frames; bench realism; ADR-026 QA caveats; unknown dating / short-possession convention of the organisers' labels (ADR-030 hedge).
