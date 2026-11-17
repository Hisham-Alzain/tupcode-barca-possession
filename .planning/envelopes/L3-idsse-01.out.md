```yaml
id: L3-idsse-01
status: done
files: [scripts/idsse_cache.py (new), scripts/bench.py, tests/test_bench_idsse.py (new)]
matchlens_start: matchlens_ref/eval/sportec.py (SportecMatch.iter_frames, _is_alive, _is_goalkeeper, _team_id) — ported conventions (centre metres, conf 1.0, GK = "TW", team 1 = home); kloppy replaced by an iterparse of the raw DFL XML so BallPossession survives and the parse runs once.
results_file: eval/results/bench_team_control_v1_tune_idsse_s0.json (not committed; outside owned files)
```

**Wiring.** `idsse1` = J03WMX (tune), `idsse2` = J03WPY (heldout split; not run). Rows: half 1, then 25 "X" rows (no spell crosses half time), then half 2. IDSSE is 25 Hz (T step 0.04 s), no resampling. Truth: BallStatus 0 -> D, else BallPossession 1 -> H / 2 -> A; outside halves -> X. Ball = IDSSE's own (kept through stoppages), then `hide_mask` (same histogram, seed); `show_dead_ball` not applied. Players: same `tracks` shape as Metrica, referees dropped.

**Parse/cache.** iterparse + clear: idsse1 28 s, idsse2 25 s (one-off) -> `data/external/_bench/idsse_<name>_v1.npz`; reload 0.4 s. Bench runtime idsse1 80 s first time (MatchLens replay + distances, then cached).

**Tune, team_control_v1, seed 0**

| match | spellF1 [CI] | macroF1 | aliveDeadF1 | possessions true -> pred | mean s true -> pred |
|---|---|---|---|---|---|
| metrica1 | 0.3777 [0.345, 0.415] | 0.651 | 0.802 | 357 -> 377 | 9.9 -> 11.0 |
| metrica2 | 0.3348 [0.302, 0.368] | 0.670 | 0.821 | 330 -> 369 | 10.1 -> 11.5 |
| idsse1 | 0.3353 [0.300, 0.371] | 0.751 | 0.890 | 511 -> 286 | 6.7 -> 11.7 |
| mean | 0.3492 (was 0.3562 on Metrica only; Metrica numbers unchanged) | | | | |

**Truth sanity idsse1:** H 26.8%, A 31.7%, D 41.5% of in-half frames; 511 possessions, mean 6.7 s.

**Odd / notes**
- D share 41.5% is high vs Metrica; DFL BallStatus marks long dead spells (restarts held dead until taken).
- IDSSE possessions are shorter (6.7 s) and the pipeline under-segments: 286 predicted vs 511 true, so merges likely drive the IDSSE misses (per-match breakdown is in the results JSON; the totals above are pooled). DFL possession flips more often than Metrica event-derived truth.
- Coordinates are fixed to the stadium (teams swap ends at half time), same as Metrica.
- Player IDs still present (no ID-dropping yet), as for Metrica.
