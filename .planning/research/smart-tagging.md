# Smart Tagging as input (L6-smart-tagging, 2026-11-16)

**Status (L8, 2026-11-16 23:40 CET): SCORED, REJECTED.** All four steps ran; numbers and the
conclusion are in `.planning/research/autonomous-L8.md` rows 1 and 7-11 (TRIED drafts T91, T96, T97).
Short form: the phase clips are padded windows (agreement with our own team call 0.82 in the first
fifth of a clip, 0.56 in the last; 0.76 on visible frames overall), the simulator calibrated on those
INPUT statistics gives a clip purity of 0.77-0.80 against a decoder that is right on 0.92-0.94 of the
frames where it names a team, and every weight loses on ctl T=2 truth (real2 -0.001..-0.023, real1
-0.003..-0.044); an error-free simulated tagger gains nothing either. The restart taker loses even
when every restart is tagged with the true taker (the 35 s clip cannot be anchored to one stoppage).
All flags stay off; Smart Tagging is used for the kick-offs only. `CAL` (p_wrong 0.10, jitter 3 s)
is now a measured choice: it reproduces the provided file's 0.761 visible agreement.

## Tag inventory (from the XML text only; no tracking data was loaded)

638 clips. Team labels: Barcelona 314, Atletic Club 288, N/A 36. Home team name is read from the
ATD metadata (`Score@idLocalTeam`), not hard-coded.

| code | n | | code | n |
|---|---|---|---|---|
| PLAYERS IN THE BOX | 98 | | CREATING CHANCES | 31 |
| BALL IN FINAL THIRD | 94 | | LONG BALL | 30 |
| BALL IN THE BOX | 73 | | DEFENSIVE TRANSITION | 25 |
| PROGRESSION | 67 | | ATTACKING TRANSITION | 25 |
| DEFENDING IN MIDDLE THIRD | 67 | | DEFENDING IN ATTACKING THIRD | 23 |
| SET PIECES | 60 | | BUILD UP | 23 |
| DEFENDING IN DEFENSIVE THIRD | 31 | | GOALS | 5 |

SET PIECES types: Throw-in 26, Corner Kick 12, Goal Kick 10, Free Kick 5, Kick Off (After Goal) 4,
Kick Off (Start) 2 (one clip without a type). Kick-offs and throw-ins carry Team = N/A, so at most
27 restarts have a tagged taker, plus the kick-offs after the 5 GOALS clips (conceding team, Law 8).
571 team-phase clips, median length 21.8 s.

Not measured (needs the tracking input): where the restart sits inside a SET PIECES clip, the
agreement of the phase clips with our decoded team on visible / hidden frames, our D share inside
clips. `scripts/a8_tags.py` prints all of it and writes `eval/results/a8_tags_inventory.json`.

## Mechanisms (opt-in, default off)

| flag | what | new_angle vs TRIED |
|---|---|---|
| `w_tag`, `tag_vis`, `trim_s` | team-phase clips as a per-frame team prior in the Viterbi emission: +1 / -1 where one team is named, 0 where clips of both teams overlap; full weight on hidden frames, `tag_vis` scale on visible ones | T34 / T35 / T88-T90 read the team from the players only; this is an external source that does not depend on the ball or the shape model |
| `w_tagr`, `tagr_hold_s` | tagged set-piece taker for 4 s after the restart; the restart is the end of the last stoppage of a first decoding pass that starts inside the clip | T16 (other team restarts) and the P4 geometry idea guess the taker; here the tag names him. T16 note stands: the restart team was already right 340/421 and the oracle was +0.002, so little is expected |
| goals | folded into `w_tagr`: the kick-off clip after a GOALS clip belongs to the conceding team | not tried before; 4 events in the match, cannot be validated on its own |

MatchLens: nothing reads Smart Tagging or any tag prior (checked possession.py, impulse.py,
deadball.py, attacking_direction.py, detectors/set_piece_detector.py, which takes restart type and
executor from geometry).

## Simulator (`scripts/a8_simtags.py`), designed, synthetic check only

Clips are drawn until they cover 0.68 of the played frames; each is anchored on a random live frame
of the external truth, speaks for the team in possession there, and takes its length from the
provided file's own clip lengths. Assumptions to vary: anchor position (uniform / centre / whole
true phase + 3 s), the truth the tagger sees (ctl 2 s or raw), wrong-team share 0 / 0.1 / 0.2,
mis-timing sigma 0 / 3 / 6 s, dropped tags (cover 0.45). Set-piece clips are drawn around true
restarts at the provided rate. The calibrated setting is meant to be the one whose INPUT statistics
(both-team share, agreement with our own default labels on visible frames) match the provided file;
`CAL` in the script is a **placeholder**, not a measured choice.
On a synthetic label sequence: coverage 0.68 reached with ~125 clips; clip purity 0.85 (uniform
anchor), 0.94 (phase), 0.74 (pessimistic). These say nothing about real matches.

## To finish (in this order, each through ramlock)

1. `scripts/a8_tags.py` (provided input alignment).
2. `scripts/a8_simtags.py stats bench=real2 seed=0`, set `CAL`.
3. `scripts/a8_simtags.py run bench=real2|real1|orig seed=0 sim=cal`, then sims `phase`, `pess`,
   `raw`, `drop` for the best config, then real2 seeds 1-3 (15 configs in `CONFIGS`).
4. `scripts/a8_simtags.py transfer`.
Decision by ADR-016 / ADR-022 on ctl T=2 truth; a gain that exists only under `phase` is not a gain.
