```yaml
id: L4-control-definition-fable
status: done; opt-in only, defaults unchanged; code commit 6cce2a5 (agent/a5)
```
**ctl rule** (`src/possess/ctl_truth.py:relabel`): a team spell is controlled if it lasts >= T s. An uncontrolled spell directly after a live spell of the other team returns to that team (A b A -> AAA, A b D -> AAD), shortest first, merging. The first spell after D is kept ("clear possessor"); controlled spells keep their first frame. Source: CHALLENGE.md 53-78. **T = 2 s**: smallest reading of "a few seconds"; the pass proxy (> 2 passes) rescues 0 spells under 2 s (Metrica; IDSSE events lack frames). True possessions 1198 -> 1044 / 875 / 773 (T 1/2/3).

**spellF1 tune mean, old / ctl T=2 truth**
| config | real2 s0 | s1 | s2 | s3 | real1 | orig |
|---|---|---|---|---|---|---|
| default | .5618/.6213 | .5519/.6068 | .5654/.6306 | .5549/.6140 | .6068/.6103 | .6283/.6339 |
| ctl_s=2 | .5415/.6457 | .5374/.6469 | .5434/.6622 | .5355/.6335 | .5522/.6521 | .5637/.6671 |
| ctl_s=1 | .5589/.6333 | .5505/.6244 | .5610/.6415 | .5537/.6244 | .5941/.6352 | .6113/.6561 |
| confirm_s=2 | .5523/.6278 | .5390/.6121 | .5569/.6375 | .5434/.6166 | .5722/.6356 | .5884/.6549 |

ctl_s=2 on ctl truth: +0.020..+0.042, CI90 lower bound > 0 on all six; old truth -0.015..-0.065 (fails ADR-016); T=1 truth: down to -0.030. ctl_s=1 gains on T=2 and T=3 truth everywhere; old -0.001..-0.017. T58 flicker + protect and c_switch 6/8: weaker or bench-dependent (JSON).

**Recommended flag**: `--dec ctl_s=2.0` (post-pass in `decode_halves`; the user's rule: a returning deflection is no change, a kept interception starts at its first touch) if T=2 is accepted; hedge `ctl_s=1.0`. `confirm_s` (touch level) is weaker.

**Transfer** (provided INPUT, possessions per half hour): default 116.7 (412); ctl_s=1 107.1; ctl_s=2 83.8 (-28%; real2 -20%); D share unchanged.

**Commands** (via `uv run python scripts/ramlock.py --`): `uv run python scripts/a5_ctl.py truth | save bench=real1|orig | run bench=real2|real1|orig seed=0..3 | transfer`. Results `eval/results/a5_ctl_*.json`; pytest 162 green; no public truth opened.
