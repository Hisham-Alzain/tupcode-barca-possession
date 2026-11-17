```yaml
id: L3-idsse-01
goal: Add the two IDSSE (DFL) matches to the external possession bench, with real per-frame truth (ball status + ball possession) and the ball tracked during stoppages.
matchlens_start:
  files:
    - matchlens_ref/eval/sportec.py (loader: positions XML, BallStatus/BallPossession per frame) — also D:\ITE\year5\graduation project\The project\footballtactics\backend\events\eval\sportec.py (read-only original; same code, plus OPEN_MATCH_IDS / load_sportec_match)
    - footballtactics backend evaluation/adapters/metrica_replay.py build_clip (dataset="sportec") for how MatchLens already feeds IDSSE to its Eventer
  baseline: "none yet on IDSSE; current bench mean tune spellF1 0.3562 (team_control_v1) on Metrica G1/G2"
context_refs:
  - scripts/bench.py (read all; extend it, keep every existing CLI and cache name working)
  - .planning/LOOP.md (protocol: tune = Metrica G1, G2, IDSSE match 1; held-out = Metrica G3, IDSSE match 2)
  - data/external/idsse/ (two matches: DFL_02_01 matchinformation, DFL_03_02 events, DFL_04_03 positions ~400 MB each). Data is git-ignored: it exists ONLY in the main checkout D:\ITe\year5\graduation project\The project\barca-possession\data\external\idsse — read it from there by absolute path.
owned_files: [scripts/bench.py, scripts/idsse_cache.py (new), tests/test_bench_idsse.py (new)]
constraints:
  - Never read anything under data/challenge/ (not needed for this task).
  - footballtactics is read-only.
  - Parse each 400 MB XML once and cache compact arrays (npz) under the MAIN checkout's data/external/_bench/ (absolute path); later runs must load the cache in seconds. Watch RAM (16 GB laptop): stream-parse (iterparse), clear elements.
  - Truth per frame: BallStatus dead -> "D"; else BallPossession team (home=1 -> "H", away=2 -> "A"); frames outside the two halves -> "X". Home/away from matchinformation.
  - The pipeline must see: players per frame with team only (same `tracks` dict shape as Metrica: {"players":{id:{map_position:(x_m,y_m), team, conf}}, "goalkeepers":{...}, "ball":{1:{map_position, conf}}}), pitch-centred metres, x along the length. Ball: IDSSE's own ball (kept during stoppages — this is the point), then the bench's long-block hiding (hide_mask, same histogram) on top. Do NOT apply show_dead_ball to IDSSE.
  - Downsample if needed so frame rate is 25 fps (IDSSE is 25 Hz; check).
  - Register names "idsse1", "idsse2"; put idsse1 in SPLITS["tune"], idsse2 in SPLITS["heldout"]. Do not run the heldout split.
  - Run: `python scripts/bench.py --pipeline team_control_v1 --split tune` (with the footballtactics venv python: D:\ITE\year5\graduation project\The project\footballtactics\backend\.venv\Scripts\python.exe, and PYTHONPATH including your worktree src). Report idsse1 numbers next to Metrica G1/G2.
deliverable: code + a small test (truth mapping on a synthetic snippet) + output envelope .planning/envelopes/L3-idsse-01.out.md (<= 300 words) with: tune numbers per match, how long parsing/caching took, truth sanity (share of H/A/D, possessions count, mean length), and anything odd.
done_when: bench tune split runs end to end on metrica1, metrica2, idsse1; tests green; committed on your worktree branch.
budget: 1 round
```
