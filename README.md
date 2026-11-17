# barca-possession

More Than A Hack 2027, Phase 1: one label per frame of Barcelona vs Athletic, H (Barcelona has the ball), A (Athletic), D (ball out of play) or X (no answer), from automatic, identity-free tracking. Ranked by spellF1. Rules: `docs/RULES.md`.

The repository is self-contained: no code outside it is needed to regenerate the submission.

## Reproduce the submission
Requirements: Python 3.12 and [uv](https://docs.astral.sh/uv/). All versions are pinned in `uv.lock`.
```bash
uv sync
# the match data is NOT in this repo (data/ is git-ignored); put the organisers' bundle under data/challenge/ so that these exist:
#   data/challenge/tactical/BAR-ATH_tactical_atd.{xml,txt}            (tracking, required)
#   data/challenge/tactical/BAR-ATH_tactical_smart_tagging.xml        (kick-off and set-piece tags)
uv run possess --data data/challenge --out submissions/draft_2026-11-17f.csv
uv run python vendor/mtah/validate_submission.py submissions/draft_2026-11-17f.csv
uv run pytest                             # tests need no match data
```
The submitted file is `submissions/draft_2026-11-17f.csv`, the output of the default settings (`shape.DEC_SHAPE`: ADR-022 plus ADR-023..026, ADR-028 and ADR-030 (`ctl_s = 0.6`); `pipeline.TAGS`: the ADR-029 set-piece rule, `sp=1 sp_hid=1 sp_take_s=0`; see `.planning/DECISIONS.md`); no flag is needed. ADR-029 is a recorded exception to our adoption gate (`docs/TECHNICAL.md` §10). Inputs read: the tactical tracking and, from the Smart Tagging file, the "Kick Off (Start)" tags (to find each half's kick-off) and the goal kick, corner and throw-in clips (only as a gate for finding the physical restart); any `*tactical_smart_tagging.xml` under `--data` is found. No video, homography or broadcast data. `MTAH_DATA_DIR` sets the default for `--data`. The run needs no external data: the team-shape models, the ball-team model and the visible-dead-ball model (trained on external matches only) are stored as coefficients in `src/possess/shape.py` (`MODELS`), `src/possess/ballteam.py` (`MODEL`) and `src/possess/balldead.py` (`MODEL`).

Team: **TupCode** · repository: https://github.com/Hisham-Alzain/tupcode-barca-possession · technical document: `docs/TECHNICAL.md`.

## Pipeline (`src/possess/`)
| Step | Module | Origin |
|---|---|---|
| Read tracking (EPTS via kloppy), team slots, goalkeeper guess | `epts.py` | ours |
| Per-frame features: nearest home/away player to the ball, team centroid speeds | `features.py` | ours |
| Kick-off of each half from the Smart Tagging "Kick Off (Start)" tag; graded frames before it are D | `kickoff.py` | ours (ADR-021) |
| MatchLens dead-ball state: out of bounds, stationary ball, placed ball, leaky-bucket hysteresis | `stoppage_ml.py` | ported from MatchLens `deadball.py` + `ball_strike.py` + ball speed from `kinematics.py`; **ghost rule off** (a missing ball is not a stoppage, ADR-002) |
| Stoppage detector: ball far from everyone + near a line or collective slowdown, plus the MatchLens state | `deadball2.py` | ours (loop 3) |
| Team-shape models (who has the ball, is it dead) from anonymous player positions; attacking-direction vote; per-match hidden-frame dead-ball prior; the default decoder settings `DEC_SHAPE` | `shape.py` | ours; direction vote adapted from MatchLens `attacking_direction.py` |
| Frame rule: Vidal-Codina possession/duel zones evaluated per team, retention before a switch | `team_control.py` | ours, zone idea from MatchLens `possession.py` (ADR-003) |
| Ball cleaning before touch detection | `ballclean.py` | ours |
| Whole-half Viterbi decoder (H, A, D-after-H, D-after-A), touch detection (impulse, low-passed and contact-point modes, reappearing ball), ball flight direction as team evidence (`w_fly = 0.7`) | `decode.py` | ours (flight direction ADR-024); touch rule ported from MatchLens `impulse.py` |
| Learned ball-team evidence: logistic model of which team holds the visible ball (`w_bt = 0.5`), fitted on external tuning matches only | `ballteam.py` | ours (ADR-026) |
| Visible dead ball: learned "the visible ball is dead" logistic model added to the stoppage evidence (`w_bd = 1.0`), fitted on external tuning matches only | `balldead.py` | ours (ADR-028) |
| Hidden stoppage ends snapped to the players' speed-up; stoppage start moved to the ball's line crossing; visible ball beyond a line for ≥ 2 s is dead; ball at rest ≥ 3 s with no player within 1 m is dead (`ed_still = 1.0`) | `edges.py` | ours (line crossings ADR-025, still ball ADR-028) |
| Set-piece dead ball: inside a tagged goal kick / corner / throw-in, frames from our last dead evidence to the physical restart (ball at rest on the required spot then kicked, or reappearing there) are D; free kicks excluded; no restart found, no change | `setpiece.py`, `pipeline.py` | ours (ADR-029, recorded gate exception) |
| Controlled-possession post-passes: a team spell that interrupts the other team is given back if under 0.6 s (`ctl_s = 0.6`), or under 2 s without one uncontested touch of its own (`ctl_hi = 2.0, ctl_touch = 1`) | `ctl_truth.py`, `decode.py` | ours (ADR-022, ADR-023; 0.6 s since ADR-030) |
| Orchestration and CLI | `pipeline.py` | ours |

`cleanup.py` is imported on every run but its clean-up rules are off in the default; `smarttags.py` holds Smart Tagging experiments that were tested and rejected (flags off in the default); `metrica_csv.py` is used only by the external bench.

What was not brought over from MatchLens: the streaming Eventer, the per-player possession tracker (it leaves 18–20% of visible live frames without a team), set-piece and event detectors, schema, perception. The goalkeeper-holding override was dropped: the ported detector agrees with the full MatchLens run on 100% of frames without it (Metrica G1).

Ball gap-filling experiments (10 methods scored, none beat the unfilled ball) live in `src/possess/ball_fill/`.

## Evaluation (external data only)
| Path | What |
|---|---|
| `scripts/bench.py` | External bench: Metrica G1–G3 and IDSSE (DFL) truth, ball hidden in long blocks like the challenge, official scorer, error breakdown by cause. Tune split: Metrica G1, G2, IDSSE 1; held-out: Metrica G3, IDSSE 2. |
| `scripts/idsse_cache.py` | IDSSE positions XML → cached arrays with per-frame BallStatus/BallPossession. |
| `scripts/oracle.py` | Oracle ablations: which component limits the score. |
| `scripts/eval_ball_fill.py` | Masking test for ball gap filling. |
| `vendor/mtah/` | The organisers' official scorer and validator (their MIT licence in `vendor/mtah/LICENSE`, copied unchanged). |
| `eval/results/` | Every scored run. |

External data goes in `data/external/` (git-ignored): Metrica sample data (https://github.com/metrica-sports/sample-data) and IDSSE (https://doi.org/10.1038/s41597-025-04505-y). Scoring the public 5-minute window needs the organisers' repo (`git clone https://github.com/metrica-sports/more-than-a-hack-2027 third_party/mtah`); it is a pipeline check, reported for every version, never used to choose or tune (submitted version `draft_2026-11-17f.csv`: spellF1 0.6588, macro F1 0.8886, 17 predicted vs 17 true possessions, verified with the organisers' `run_evaluation.py`, unchanged from 17e; the organisers note the window holds only 17 possessions and is not a leaderboard, so this is not a full-match estimate; earlier drafts 0.6353 (17d), 0.5684 on 15 Nov and 0.5231 on 17 Nov).

## Records
`.planning/` holds the state, decisions (ADRs), log and the improvement loop (`LOOP.md`); `docs/research/` the research notes.

## Possession viewer
```bash
uv run python viewer/server.py --video "path/to/BAR-ATH_tactical.mp4"
```
Then open http://127.0.0.1:8765. It shows the tactical video with the pipeline's labels, every possession change (turnovers highlighted, previous/next with `[` `]`), a ±30 s timeline with nearest-player distances, mini-pitch tracking, an optional compare file and the public 5-minute truth band (check only), and a Run pipeline button that writes `submissions/viewer_run_<time>.csv`. It listens on localhost only because the match data must not leave the machine. Default video path: `data/challenge/tactical/BAR-ATH_tactical.mp4`, or the `MTAH_VIDEO` environment variable.

## Licence
Our code is MIT-licensed (`LICENSE`). `vendor/mtah/` keeps the organisers' own licence. The match data is not part of this repository and is not covered by either licence.
