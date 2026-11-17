```yaml
id: L4-ctl-qa-fable
reviews: agent/a5 (6cce2a5, caa0a71) — ctl truth + ctl_s post-pass
verdict: PASS WITH CHANGES (record-keeping only; no code defect)
```
**Reproduced exactly** (cached inputs, own script; `evaluate.py --bench real1 --dec ctl_s=1.0` = 0.5941). spellF1 tune mean, default -> ctl_s=1:

| truth | real2 s0 | s1 | s2 | s3 | real1 | orig |
|---|---|---|---|---|---|---|
| old | .5618->.5589 | .5519->.5505 | .5654->.5610 | .5549->.5537 | .6068->.5941 | .6283->.6113 |
| ctl T=2 | .6213->.6333 | .6068->.6244 | .6306->.6415 | .6140->.6244 | .6103->.6352 | .6339->.6561 |
| ctl T=3 | .6158->.6377 | .6091->.6359 | .6230->.6446 | .6104->.6336 | .5992->.6245 | .6236->.6468 |

**Pooled paired 2-min block bootstrap, CI90** (6 benches x 3 matches, shared blocks): T=2 **+0.0163 [+0.0106, +0.0219]**; T=3 +0.0236 [+0.0169, +0.0301]; T=1 +0.0057 [-0.0000, +0.0111]; old -0.0066 [-0.0125, -0.0007]. real2 x4 only: T=2 +0.0127 [+0.0058, +0.0195]; T=1 +0.0015 [-0.0050, +0.0081]. All 18 match x bench deltas positive on T=2 and T=3.

**Findings**
1. Gain is not T-independent: it grows with T and is zero on real2 at T=1. ctl_s=1 pays only if the organisers' control needs >= ~2 s.
2. Old truth: fails ADR-016 on real1 (-0.0127 [-0.0220, -0.0039]) and orig (-0.0169); idsse1 carries it (-0.032/-0.034). The ADR must state this is a definition change, not a gate pass.
3. Compliance: no constant from the provided match, no GT opened (diff grep clean). The ADR must justify 1.0 vs 2.0 from external numbers only, never from 107.1 vs the public clip's possession rate.
4. Correctness, `src/possess/ctl_truth.py:54-72`: 15 edge cases + 300 random sequences pass (period blocks, D/X never written, first spell after D kept, chains, idempotent). On all 18 real decodes D and X unchanged, post-pass == relabel(default). Kept interception starts at its first frame; in `A b a B` with len(b) <= len(a) (tie, line 60), b returns and B starts after a.
5. Minor: `ctl_truth.py:88-89` returns the input object when T <= 0. Post-pass (`decode.py:521-524`) precedes main's ADR-021 kick-off D; merge is clean, re-run transfer after merge.

**Transfer** (INPUT): 412 -> 378 possessions, 116.7 -> 107.1 per half hour; D share 0.4395 unchanged; 0.24% frames changed.

pytest 162 green. Nothing committed; no source edited.
