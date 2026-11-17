```yaml
id: L2-6-decoder-fable
status: win (tune only; held-out not looked at; no CI yet)
matchlens_delta: zone constants and frame rule reused from team_control.py; nothing in MatchLens does whole-half decoding (checked possession.py, deadball.py ports).
```

**Why v1 lost (0.3597 vs rule 0.3724)**
- Hidden-ball blocks: no evidence, so a switch could sit anywhere at equal score; v1's tie-break put it at an edge. Same emission, switch moved to the block middle: 0.3539 -> 0.3650 (start 0.3598, end 0.3546).
- Flat zone emission: on Metrica truth, log-odds of possession are 3.0 for a team < 0.5 m from the ball but only 0.5-1.2 at 0.5-1 m; v1 scored both the same.
- Not the cause: under-switching (counts barely move) or soft stoppages (soft 0.3910 vs hard 0.3881).

**Changed** (`src/possess/decode.py`, opt-in, defaults = v1 exactly, pinned by test)
- `soft=1`: emission `ln(1+min(d,cap)/d0)` difference between teams, zero beyond `cap`.
- `gap_pos`: switch inside a hidden block placed at that fraction.
- `hard_dead=1`: D copied from the detector. `w_rule`: frame rule as extra emission.
- Loop in plain Python (3 matches in ~3 s). `tests/test_decode.py` (8 tests; suite green).

**Result** (22 configs, `eval/results/tune_decoder_v2.json`)
| config | mean | G1 | G2 | IDSSE1 | pred/true |
|---|---|---|---|---|---|
| best, `w_rule=0.3` | **0.3976** | 0.452 | 0.390 | 0.351 | 289/357, 280/330, 243/511 |
| simpler, no rule | 0.3881 | 0.434 | 0.388 | 0.342 | 299/357, 283/330, 240/511 |

All three matches up in both. Reproduce:
`uv run python scripts/evaluate.py --no-public --variant p3v2_rule --decoder viterbi --dec hard_dead=1 soft=1 cap=1.5 d0=0.25 c_switch=4.0 c_same=1.0 gap_pos=0.5 w_rule=0.3`

**Risks**
- +0.025 with no bootstrap CI; neighbours span 0.373-0.391, so `w_rule` (+0.01) may be noise. `cap` is sensitive (1.0: 0.373, 2.0: 0.374).
- IDSSE still under-segments (243/511).
- `team_control.py` changed mid-session (b17aeef); final numbers re-run after it. Earlier scratch runs with `w_rule` are superseded.
- I never ran the public window (`--no-public`); orchestrator's call.
- `tune_decoder.py` now calls `decode_halves` directly (detector once per match); `evaluate.py` gives identical numbers.
