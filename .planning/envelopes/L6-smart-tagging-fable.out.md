```yaml
id: L6-smart-tagging-fable
status: INCOMPLETE - built and unit-tested, NOTHING SCORED (machine out of RAM)
```
**Blocker.** Every data-loading command was refused by `ramlock` (free RAM 1.3-1.7 GB < 3 GB, held by desktop apps); the last retry loop was killed by Claude Code for low memory. I did not bypass the lock or restart. No number below comes from tracking data.

**Tag inventory (XML text only).** 638 clips; Barcelona 314, Atletic Club 288, N/A 36. 571 team-phase clips, median 21.8 s. SET PIECES 60: throw-in 26, corner 12, goal kick 10, free kick 5, kick-off 6; only 27 carry a taker. GOALS 5. Alignment with our output (restart position in clip, agreement visible/hidden): **not measured**.

**Built (all opt-in, default off, labels byte-identical, pytest green):**
- (a) `w_tag`/`tag_vis`/`trim_s`: phase clips as team prior in Viterbi emissions (0 where both teams overlap). new_angle vs T34/T35/T88-90: external source, independent of ball and shape.
- (b) `w_tagr`/`tagr_hold_s`: tagged taker for 4 s after the restart found by a first decoding pass. new_angle vs T16/P4: taker named, not guessed. Low expectation (T16: oracle +0.002).
- (c) goals -> conceding team's kick-off, folded into (b); 4 events, not separately testable.
- CLI: `possess --tags w_tag=.. w_tagr=..`. matchlens_start: nothing covers tags (possession, impulse, deadball, attacking_direction, set_piece_detector checked).

**Simulator** `scripts/a8_simtags.py` (stats / run / transfer; 11 settings, 15 configs): written, checked on synthetic labels only. `CAL` is a placeholder.

**Candidates table, seeds, bootstrap, sensitivity, transfer:** none. **Recommendation:** do not enable any flag; run the four steps in `.planning/research/smart-tagging.md` once >= 3 GB is free (about 12 locked runs).

**Risks:** decode two-pass and simulator untested on real inputs; `%TEMP%/barca_possession_ram.lock` may be a stale lock from my killed job (expires after 20 min).

**Files:** src/possess/smarttags.py (new), decode.py, pipeline.py, scripts/a8_tags.py, scripts/a8_simtags.py, tests/test_a8_smarttags.py, .planning/research/smart-tagging.md.
