```yaml
id: L2-12-audit-fable
status: done (tune only; no held-out look taken, no default changed, nothing committed)
matchlens_start: nothing in MatchLens covers bench degradation (checked eval/metrica.py interpolate_ball = the straight fill the bench already ports)
files: scripts/bench.py (get_case/case_features variant=; "" unchanged, reproduces 0.5845), scripts/transfer_check.py, scripts/audit_bench.py, eval/results/audit_transfer{,_real1}.json, audit_steps_{orig,dead1,fault1,real1}.json
```
**1. Transfer table (provided INPUT vs 5 bench matches; no ground truth opened)**
| signal | provided | bench | |
|---|---|---|---|
| visibility / gap lengths | 0.593, 99.7% in gaps >= 5 s | 0.58-0.60, 99.8% | ok |
| still < 0.3 m/s / < 1 m/s | 0.045 / 0.133 | 0.15-0.46 / 0.34-0.49 | DIFFERS |
| speed median | 4.1 m/s | 1.2-2.3 | DIFFERS |
| nearest player <= 1 m | 0.285 | Metrica 0.38-0.43, IDSSE 0.25-0.27 | Metrica differs |
| touch-hold share (onsets/min) | 0.265 (3.9) | 0.52-0.65 (5.6-7.3) | DIFFERS |
| detector dead share, visible | 0.079 | 0.20-0.46 (truth D 0.32-0.44) | DIFFERS |
| repeats / freezes / piecewise-linear | 0.1% / none / no | Metrica is 5 Hz linear; IDSSE frozen 41-45% | bench artefacts |

The provided ball shows no sample-and-hold and no long freezes; ball cleaning removes 0.0% of it.

**2. Realistic bench** `variant="real1"` (`dead1` = dead ball only, `fault1` = faults only): in every true stoppage the ball rolls out, is carried on a wandering path, lies at the restart spot 0.3-1.5 s. Wander (4 m, 1 s) set so input speed stats meet the provided match (real1: still 0.045-0.08, slow 0.13-0.17, median 3.8-4.1). Faults: one small jump/min; hold and freeze options exist but stay off (not in the input). Truth, hidden mask and LOMO shape models identical: no refit needed.
Not reproduced: touch share (real1 0.67-0.72) and detector dead share (0.18-0.24) stay above the provided match; placed ball gives ~1 freeze/min.

**3. Re-score (tune, LOMO)**
| step | orig | real1 | gain orig -> real1 |
|---|---|---|---|
| rule | 0.3724 | 0.3556 | |
| ADR-005 decoder | 0.3976 | 0.3771 | +0.025 -> +0.022 |
| ADR-006 shape | 0.4932 | 0.4619 | +0.096 -> +0.085 |
| ADR-007 touch | 0.5657 | 0.5092 | +0.073 -> +0.047 |
| ADR-008 cleaning | 0.5679 | 0.5113 | +0.002 -> +0.002 |
| ADR-009 context | 0.5845 | 0.5276 | +0.017 -> +0.016 |

dead1 default 0.5310, fault1 0.5804. Detector D recall (visible): 0.70/0.72/0.95 -> 0.40/0.43/0.57. Removing touch or shape from the default still loses on real1 (0.4780, 0.4398).

**Recommendation.** Safe: ADR-005, 006, 009, 008 (008 inert on the real match). At risk: ADR-007 (a third of its gain was the drawn dead ball; real touch activity is half the bench's) — keep, still positive in all matches. Largest exposure is not an ADR-005..009 step: the stoppage detector depends on a still/frozen dead ball. Change no default before submission; expect ~0.53 or less, not 0.58. Decide future changes on real1. Note: CLAUDE.md still forbids reading data/challenge (ADR-001 allows it); a directory listing there was denied.
