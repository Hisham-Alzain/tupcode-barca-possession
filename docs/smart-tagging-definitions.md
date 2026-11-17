# Smart Tagging codes: official definitions, measured behaviour, use for H / A / D

Work package L12-defs, 2026-11-17. Question: "what are the official definitions of the tags?"
File described: `data/challenge/tactical/BAR-ATH_tactical_smart_tagging.xml` (organiser INPUT, 638 instances).
Nothing here was fitted or tuned. The ground truth was never opened. Measurements in section 3 use the
provided INPUT only (tag XML + automatic tracking) and exist to understand what a tag means; any
threshold that follows from them must be validated and tuned on `data/external/` before use.

Three kinds of statement, marked throughout:

- **[OFFICIAL]** text written by the organisers (Metrica Sports + Barça Innovation Hub) in the challenge repository.
- **[MARKETING]** Metrica Sports product pages. They name categories; they define nothing.
- **[INFERRED]** our reading of the file and the tracking. Not a definition.

## 1. What exists and what does not

| source | what it gives | status |
|---|---|---|
| `third_party/mtah/OVERVIEW.md:96-135` (public copy: https://github.com/metrica-sports/more-than-a-hack-2027/blob/main/OVERVIEW.md) | one-sentence definition per code, the tag groups, the warning | **[OFFICIAL]**, the only definitions that exist |
| `third_party/mtah/DATA.md:123-180` | counts, tag values, time base, three caveats | **[OFFICIAL]** |
| `third_party/mtah/FAQ.md:92-99`, `CHALLENGE.md:117`, `starter/load_smart_tagging.py:1-15` | "use it", "inferred", "phases, not possessions" | **[OFFICIAL]** |
| `C:\Users\ASUS\Downloads\Barcalona Hackathon.docx`, paragraphs 119-126, 271 | the user's summary of the lines above; adds nothing | summary of OFFICIAL |
| https://www.metrica-sports.com/blog/smart-tagging-3-0-football-analysis , https://www.metrica-sports.com/blog/smart-tagging-football-analysis , https://www.metrica-sports.com/blog/metrica-nexus-all-in-one-sports-video-analysis-software-for-every-team | category names only | **[MARKETING]** |
| Metrica help centre (https://www.metrica-sports.com/help-center/...) | manual coding (Tags, Code Table, Coding Sequences). No article on how Smart Tagging detects a pattern was found | nothing |

**No official detection logic exists in public.** No page, organiser file or help article states a
threshold, a zone boundary in metres, a pre-roll / post-roll padding, a minimum duration, or the rule
that assigns the Team. Everything about *how* a tag is produced is [INFERRED] in this document.

Limit of the web search: this session could run searches but could not open pages (the page fetcher
is blocked by the context-mode hook and its replacement tool was not connected). The Metrica pages
are therefore cited from search-engine excerpts, not read in full; the category names below are as
the excerpts report them. The organiser files were read in full from the repository.

## 2. Official definitions, verbatim

All from `third_party/mtah/OVERVIEW.md` (line numbers given). **[OFFICIAL]**

| code in the XML | official definition (verbatim) | tags | line |
|---|---|---|---|
| `BUILD UP` | "The set of offensive actions whereby a team initiates and develops possession from its own half, aiming to overcome the opponent's first line of pressure and organise the advance toward more forward zones." | Half, Team | 101-103 |
| `PROGRESSION` | "Offensive actions advancing the ball primarily through the middle third, overcoming rival lines via passes, carries or support play." | Half, Team | 104-105 |
| `CREATING CHANCES` | "Offensive actions in the final third that generate clear danger or finishing situations — key passes, assists, actions leaving a teammate in an advantageous position." | Half, Team | 106-108 |
| `DEFENDING IN ATTACKING THIRD` / `MIDDLE THIRD` / `DEFENSIVE THIRD` | "Defensive actions performed in each third of the pitch: pressure and space control high up, protection of central spaces to limit progression, and marking, covering and blocking near the team's own goal." | Half, Team | 109-111 |
| `ATTACKING TRANSITION` | "The counter-attack: actions immediately after recovering the ball, exploiting the rival's defensive disorder." | Half, Team | 112-113 |
| `DEFENSIVE TRANSITION` | "Actions immediately after losing possession — reorganising, protecting key spaces, pressing or dropping back." | Half, Team | 114-115 |
| `SET PIECES` | "Situations initiated after a stoppage: corners, free kicks, throw-ins, goal kicks, kick-offs." | Half, Team, Type | 116-117 |
| `GOALS` | "The ball completely crossing the goal line into the rival net." | Half, Team | 118 |
| `LONG BALL` | "A direct long-distance pass, generally from the defensive third, to gain territory quickly or attack the space behind the defence." | Half, Team | 119-120 |
| `BALL IN FINAL THIRD` | "The team maintains or plays the ball in the zone closest to the rival goal." | Half, Team, Side, Max players in the box | 121-122 |
| `BALL IN THE BOX` | "The ball enters or is played inside the rival box, by pass, cross or carry." | Half, Team, Direction of ball entry, Max players in the box | 123-124 |
| `PLAYERS IN THE BOX` | "The quantity of attacking players inside the rival box during an offensive action." | Half, Team, Max players in the box | 125-126 |

Labels (OVERVIEW.md:128-131, verbatim): "*Team* — the team performing the event. *Half* — first or
second. *Type* — the set-piece category. *Max players in the box* — the maximum number of attacking
players in the box at the moment of the event. *Side* — left, centre or right. *Direction of ball
entry* — diagonal, vertical or horizontal."

Set-piece `Type` values have **no definition** beyond their names (DATA.md:156): `Throw-in`,
`Corner Kick`, `Goal Kick`, `Free Kick`, `Kick Off (Start)`, `Kick Off (After Goal)`. No penalty type
exists in the file.

Official caveats that bound every use (verbatim):

- OVERVIEW.md:133-135: "Smart Tagging is not manually ground-truthed. It is inferred through automated processes and may contain both false positives and false negatives. It is a strong free prior, not an answer key."
- DATA.md:170-172: "Every phase carries a team tag and every possession has a team; every set piece is a restart and every restart ends a dead-ball stretch. That is a strong prior, for free, on both halves of the label you are producing."
- DATA.md:177-178: "`Team` is not always a team. 36 instances are tagged `N/A`, including **every throw-in** — exactly the restarts where knowing the side would be worth the most."
- DATA.md:179-180: "The intervals are phases, not possessions. A Build Up can span one possession, stop short of one, or cover two. Copying an interval across as a label imports its errors wholesale."
- DATA.md:164-166: "`<start>` and `<end>` are **seconds from the first frame of the tactical video**, so `round(seconds * 25)` is a video frame and there is nothing else to correct for."
- FAQ.md:94-96: "It is not ground truth and not derived from it; Metrica generates it automatically from the same video and tracking you have, without manual review".

What the official text does **not** say: which team a `DEFENDING ...` or `DEFENSIVE TRANSITION` clip
names (the defender, by "the team performing the event"; confirmed in section 3), where the restart
sits inside a `SET PIECES` clip, whether a `GOALS` clip ends at the goal or at the kick-off, and what
"third" means in metres.

**[MARKETING]** (search excerpts of the Metrica blog pages listed in section 1): Smart Tagging is
described as AI event detection run in the cloud on the automated tracking data, exported as an XML
for the video tool. Category names reported: "Build Ups", "Counter Attacks", "Direct Attacks",
"Organized Attacks", "Shots and Crosses", ball in the final third / in the box, players in the box,
defensive and attacking transitions, and set pieces "Goals", "Throw-Ins", "Goal Kicks", "Corner
Kicks". Two things follow: the product's own name for `ATTACKING TRANSITION` is "counter attack"
(same word as OVERVIEW.md:112), and goals are filed with the set pieces. No definition, threshold or
padding is given.

## 3. What the file itself shows (measured on the INPUT, no truth) — all [INFERRED]

Method: tag XML + provided tracking (`features_v1.npz`, `positions_v1.npz`), run through
`scripts/ramlock.py`. Ball position is expressed in the tagged team's attacking frame, `x_att` from
-52.5 (own goal line) to +52.5 (rival goal line), using `possess.shape.attack_dir`. Pitch taken as
105 x 68 m (player x 0.5-99.5 percentile -50.9 .. 50.6). Thirds at |x| = 17.5 m; box = `x_att > 36`,
`|y| < 20.16`. Quantiles are p10 / p25 / p50 / p75 / p90 unless said. Scripts are throw-away
(scratchpad `tagstats.py`, `tagstats2.py`, `tagstats3.py`); nothing was written to the repo.

### 3.1 Inventory and clip lengths

| code | n | Team | length s (p10 p25 p50 p75 p90) | min / max | shape |
|---|---|---|---|---|---|
| PLAYERS IN THE BOX | 97 | Bar 51, Ath 46 | 18.5 22.5 27.7 46.6 62.4 | 17.1 / 83.7 | variable |
| BALL IN FINAL THIRD | 93 | Bar 54, Ath 39 | 22.4 24.0 27.0 34.0 39.5 | 20.6 / 70.8 | variable, floor 20 s |
| BALL IN THE BOX | 72 | Bar 42, Ath 30 | 15.6 16.0 17.4 19.0 21.1 | 15.2 / 32.5 | variable, floor 15 s |
| PROGRESSION | 66 | Bar 47, Ath 19 | 11.1 13.1 17.6 23.6 27.7 | 7.7 / 37.1 | variable |
| DEFENDING IN MIDDLE THIRD | 66 | Ath 47, Bar 19 | same 66 intervals | | mirror |
| SET PIECES | 59 | N/A 32, Bar 16, Ath 11 | 35.0 x 53, 5.0 x 6 | 5.0 / 35.0 | **fixed** |
| CREATING CHANCES | 30 | Bar 19, Ath 9, N/A 2 | 30.0 x 30 | 30.0 / 30.0 | **fixed** |
| DEFENDING IN DEFENSIVE THIRD | 30 | Ath 19, Bar 9, N/A 2 | same 30 intervals | | mirror |
| LONG BALL | 29 | Ath 20, Bar 9 | 9.0 10.6 12.4 14.9 18.2 | 8.1 / 22.2 | variable |
| ATTACKING TRANSITION | 24 | Bar 13, Ath 11 | 10.7 12.1 15.4 19.0 20.5 | 9.0 / 22.0 | variable |
| DEFENSIVE TRANSITION | 24 | Ath 13, Bar 11 | same 24 intervals | | mirror |
| BUILD UP | 22 | Bar 15, Ath 7 | 13.7 17.6 20.5 23.2 25.2 | 11.7 / 28.6 | variable |
| DEFENDING IN ATTACKING THIRD | 22 | Ath 15, Bar 7 | same 22 intervals | | mirror |
| GOALS | 4 | Bar 2, Ath 2 | 70.9, 148.4, 190.2, 250.7 | | goal to kick-off |

Set-piece types: Throw-in 26 (Team N/A, 35 s), Corner Kick 12 (Bar 7, Ath 5; 35 s), Goal Kick 10
(Bar 7, Ath 3; 35 s), Free Kick 5 (Ath 3, Bar 2; 35 s), Kick Off (After Goal) 4 and Kick Off (Start) 2
(Team N/A, 5 s). Clip starts are not on a time grid (all 25 frame phases occur).

### 3.2 The defending codes are copies

Every defending clip has the **same start and end** as one attacking clip of the **other** team:

| attacking code (team X) | identical interval, team Y | pairs |
|---|---|---|
| BUILD UP | DEFENDING IN ATTACKING THIRD | 22 / 22 |
| PROGRESSION | DEFENDING IN MIDDLE THIRD | 66 / 66 |
| CREATING CHANCES | DEFENDING IN DEFENSIVE THIRD | 30 / 30 (2 with Team N/A on both) |
| ATTACKING TRANSITION | DEFENSIVE TRANSITION | 24 / 24 |

So the 142 defending clips carry **no information of their own**: "Y defends in its attacking third"
is the label written on "X builds up". The thirds in the defending codes are counted from the
defender's own goal. Using both sides of a pair counts one observation twice.

### 3.3 Padding and anchors

| code | measured | reading |
|---|---|---|
| BALL IN FINAL THIRD | first frame with the ball at `x_att > 17.5`: 0.0 1.4 **10.0 10.0 10.0** s after clip start; clip end minus last such frame: 0.0 0.0 **10.0 10.0 10.0** s | clip = [entry into the final third - 10 s, exit + 10 s]; the third boundary is 1/3 of the pitch length (17.5 m from the centre) |
| BALL IN THE BOX | first frame with the ball in the box: 4.1 **10.0 10.0 10.0 10.0** s; end minus last: 0.5 **5.0 5.0 5.0 5.0** s | clip = [box entry - 10 s, box exit + 5 s]; box = the penalty area |
| PLAYERS IN THE BOX | first frame with at least one attacker in the box: 7.2 **10.0 10.0 10.0 10.0** s; end minus last: 0.2 **5.0 5.0 5.0 5.0** s. Label vs our own count of the maximum number of attackers in the box: equal 0.76, within one 0.95, correlation 0.96 | clip = [first attacker enters - 10 s, last attacker leaves + 5 s]. It is about players, not the ball: the ball is seen in the box in 87% of the clips only |
| CREATING CHANCES | always 30.0 s; ball first in the box at 10.7 12.6 14.2 s (p25 p50 p75); the clip starts about 4 s before the matching BALL IN THE BOX clip (most frequent offset) | a fixed 30 s window around one event near the box entry (about 15 s before, 15 s after). The event itself is not identifiable |
| ATTACKING TRANSITION | minimum `x_att` in the clip (the ball stops going toward the tagged team's goal and turns) at 1.2 2.9 3.6 4.9 5.4 s; ball at start -12 m, at end +30 m; in 11 of 24 clips another clip of the **other** team is still running 2 s before the start | about 5 s of pre-roll: the regain is **inside** the clip, about 3-5 s after its start. The first seconds belong to the team that loses the ball |
| LONG BALL | highest ball speed at 1.8 2.5 3.8 7.0 9.0 s; ball at start -32 m (own third), at end +9 m; shortest clip 8.1 s | about 5 s of pre-roll before the kick, a few seconds after the landing. 20 of 29 are Athletic |
| BUILD UP | ball in the first second at -48 .. -11 m (median -28); 74% of visible frames in the own third; at the end median -2 m (the halfway line); net advance +26 m | starts with the ball in the own third, ends when the ball reaches about the halfway line or is lost |
| PROGRESSION | ball in the first second -30 .. +15 m (median -7); 62% of visible frames in the middle third; at the end median +17 m; 73% of clips see the ball reach the final third | ball in the middle third; ends near the entry into the final third (17.5 m) or at the loss |
| SET PIECES (35 s) | goal kick: ball still on the goal-area line (4.6 m from the goal line, 0.8 m/s) from 0 to +3 s, moving at +4..+5 s; leaves its spot at 0.0 1.5 3.8 5.0 6.5 s. Throw-in: ball 1.5 m **outside** the touchline from 0 to +3 s, back inside at 4.3 4.8 7.4 9.7 18.8 s. Corner: ball not tracked from -5 to +2 s, first seen at 4.6 5.9 6.8 8.1 10.0 s, travelling at 25 m/s. Free kick: first seen at 6.1 s (median) | the clip **starts during the stoppage** and the restart comes about 2-7 s later; the remaining 28 s or so are the play that follows. Consistent with [restart - 5 s, restart + 30 s] with a few seconds of detection error |
| SET PIECES kick-offs (5 s) | ball first seen at 3.1-3.3 s (after goal) near the centre, moving | [kick-off - about 3 s, + 2 s] |
| GOALS | start = start of the BALL IN FINAL THIRD clip of the scoring attack (4 / 4); end = start of the next Kick Off clip + 1.0 s (4 / 4); the ball is deepest 9.6 - 31.9 s into the clip; ball visible 3-16% of the clip | [scoring attack enters the final third - 10 s, kick-off]. Everything after the goal moment is celebration |

Taker check (clips of the next play that overlap the 5-10 s after the clip start): corner -> the
tagged team owns 12 of 13 BALL IN FINAL THIRD, 13 of 14 BALL IN THE BOX; free kick -> 6 of 6; goal kick ->
BUILD UP 6 of 6 and LONG BALL 3 of 3 of the tagged team, with the ball at -46 .. -50 m in the tagged
team's frame (8 of 8 visible). `Team` on a set piece is the **taker**.

Side (BALL IN FINAL THIRD): Left = ball at median lateral +25 m, Right = -28 m, Central between
about -18 and +17 m, i.e. the corridor boundaries sit near the edges of the penalty area (20 m).

### 3.4 Coverage, overlap, ball visibility

- The four attacking phases (BUILD UP, PROGRESSION, CREATING CHANCES, ATTACKING TRANSITION) cover
  40.2% of the graded frames: one team 38.6%, both teams 1.6%, none 59.8%.
- All eight attacking codes: one team 58.9%, both 9.2%, none 31.8%.
- Chaining is loose: of consecutive attacking phases, the next one starts -12.7 / -2.9 / +9.6 / +28.6 /
  +55.7 s after the previous ends (p10..p90). BUILD UP -> PROGRESSION of the same team overlaps in
  only 4 of 22. The codes are separate detectors, not a segmentation of the match.
- Ball visibility inside the clips: BUILD UP / PROGRESSION 0.93, ATTACKING TRANSITION 0.92, LONG BALL
  0.87, BALL IN THE BOX 0.69, BALL IN FINAL THIRD 0.65, CREATING CHANCES 0.58, PLAYERS IN THE BOX 0.51,
  GOALS 0.12; graded average 0.59; outside every clip 0.35. **The ball-driven codes exist almost only
  where the ball is tracked**, which is where our own decoder is already strong.
- SET PIECES clips cover 27.7% of the graded frames; 10 of them start less than 35 s after the
  previous one. 7 of 12 corner clips start **inside** a CREATING CHANCES clip: the zone codes run
  straight through stoppages.
- Recall of the set pieces is partial: 59 tagged restarts in the match against about 90 stoppages in
  our own output; only 5 free kicks are tagged. No tag does not mean no stoppage.

### 3.5 The tags against our own final output (`submissions/draft_2026-11-17d.csv`)

This compares two automatic outputs. It says where they differ, not which one is right.

Share of frames where our output names the **tagged** team, among frames where it names a team
(and our D share), by part of the clip:

| code | part: tagged-team share (our D share) |
|---|---|
| PROGRESSION | 0-3 s 0.95 (0.00) · middle 0.90 (0.03) · last 5 s 0.77 (0.18) |
| BUILD UP | 0-3 s 0.82 (0.25) · middle 0.94 (0.03) · last 5 s 0.92 (0.07) |
| ATTACKING TRANSITION | 0-3 s **0.27** (0.09) · 3-6 s 0.74 · middle 0.89 (0.00) · last 3 s 0.72 |
| LONG BALL | 0-3 s 0.73 (0.27) · 3-6 s 0.80 · rest 0.67 |
| CREATING CHANCES | 0-10 s 0.92 (0.36) · 10-20 s 0.67 (0.32) · 20-30 s **0.28** (0.62) |
| BALL IN FINAL THIRD | pre-roll 0.90 (0.38) · ball in the third 0.66 (0.11) · post-roll 0.53 (0.55) |
| BALL IN THE BOX | pre-roll 0.85 (0.23) · ball in the box 0.54 (0.10) · post-roll 0.38 (0.43) |
| PLAYERS IN THE BOX | pre-roll 0.76 (0.14) · players in the box 0.54 (0.56) · post-roll 0.31 (0.36) |

- ATTACKING TRANSITION: our output switches to the tagged team inside [-2, +10] s in 22 of 24 clips,
  at 1.0 1.5 3.4 4.2 4.6 s after the start; before the start it has the other team in 19 of 24.
- GOALS: from 40 s into each clip to its end our output is D on 100% of the frames (4 / 4). After the
  kick-off it names the conceding team in the 3 clips where it names a team within 5 s.
- SET PIECES, our D share around the clip start (seconds): throw-in -2..0 0.80, 0..+2 0.88, +2..+4
  0.95, +6..+8 0.33; corner -10..0 1.00, 0..+2 0.94, +6..+8 0.27; free kick 0..+4 0.91-0.98; kick-off
  after goal 1.00 until +2 s; **goal kick 0.45-0.49 from -10 to +2 s, 0.40 at +2..+4**.
- Our restart (end of our stoppage nearest the clip start), seconds after the clip start: throw-in
  3.6 4.7 6.2 7.0 15.5; corner 3.1 5.5 6.1 6.7 8.5; free kick median 6.1; kick-offs 3.2-3.3; goal kick
  -10.8 -9.7 0.7 4.8 5.6.
- First team in our output after that stoppage vs the tagged taker: corner 8 of 12, goal kick 6 of
  10, free kick 3 of 5 (17 of 27).
- Outside every clip (17.7% of the graded frames) our D share is 0.69; inside BUILD UP / PROGRESSION /
  ATTACKING TRANSITION / LONG BALL it is 0.07; over the match 0.45.

## 4. Per code: detection logic, implication for H / A / D, reliability

Definitions: section 2 (official). Detection logic: [INFERRED] from section 3. "X" is the tagged
team, "Y" the other. "Dead" follows the organisers' sentence "every set piece is a restart and every
restart ends a dead-ball stretch" (DATA.md:170). Restart times and agreement figures in this table
come from our own output (section 3.5), not from a reference.

| code | detection logic [INFERRED] | implication for the label | timing | reliability |
|---|---|---|---|---|
| BUILD UP | X has the ball in its own third; clip ends near the halfway line or at the loss; often starts at a goal kick | X in possession in the middle of the clip. The first 3 s can still be dead (our D 0.25) | edges are not possession edges | high in the middle (0.94 agreement with our output), 22 clips, ball visible 0.93: little new information |
| PROGRESSION | X has the ball in the middle third; ends at the entry into the final third or at the loss | X in possession, start and middle | last 5 s often already Y or dead (0.77) | high (0.90-0.95), 66 clips, ball visible 0.93 |
| CREATING CHANCES | fixed 30 s window around a chance event near Y's box | X attacks Y's goal. First 10 s: X (0.92). Last 10 s: mostly dead or Y (our D 0.62; X 0.28) | no usable edge; the window crosses corners and goal kicks | low as a possession label; fair for "X had the ball during the 10 s before" |
| ATTACKING TRANSITION | X regains the ball and advances fast; clip = [regain - about 5 s, end of the run + a few s] | a change of possession **to X** lies 1-5 s after the clip start; before it Y had the ball; then X until near the end | the best timing information in the file: turnover within about +-2 s of start + 3.5 s | good (22 of 24 confirmed by our output), 24 clips |
| DEFENSIVE TRANSITION | copy of ATTACKING TRANSITION with Team = Y | same as above with the team inverted; nothing new | | drop |
| DEFENDING IN ATTACKING / MIDDLE / DEFENSIVE THIRD | copies of BUILD UP / PROGRESSION / CREATING CHANCES with Team = the defender | the **other** team is in possession (as far as the mirrored code says so); nothing new | | drop, or use instead of the mirror, never both |
| LONG BALL | a long forward pass by X from its own third or half, about 5 s pre-roll | X in possession from the kick until the ball is won. The first 3 s can be dead (goal kick, free kick: our D 0.27). After the landing it is a duel: either team | kick near start + 2.5 .. 7 s | medium (0.67-0.80) |
| BALL IN FINAL THIRD | ball inside the third next to Y's goal while X attacks; clip = [entry - 10 s, exit + 10 s] | says where the ball is, and that X attacks that end. Pre-roll: X (0.90). Inside the third: X only 0.66. Post-roll: dead or Y | entry = start + 10 s, exit = end - 10 s, exact when the ball is tracked | zone: high. Team as possessor: low |
| BALL IN THE BOX | ball inside Y's penalty area; clip = [entry - 10 s, exit + 5 s] | ball in Y's box between start + 10 s and end - 5 s. Possessor there: X only 0.54. After the exit: mostly Y or dead (X 0.38) | entry and exit exact to the frame when tracked | zone: high. Team as possessor: none |
| PLAYERS IN THE BOX | at least one X player inside Y's box; clip = [first in - 10 s, last out + 5 s] | nothing about the ball. Long clips cover corners and free kicks being prepared (our D 0.56 in the core) | | counts are right (within one in 95%). As a possession label: none |
| SET PIECES, Throw-in | ball leaves over the touchline; clip starts during the stoppage, throw about 4-7 s later | **dead at the clip start and for about 3 s after** (our D 0.88-0.95); a restart follows by about +7 s. Team is N/A: the taker is not given | restart at start + 3.6 .. 7.0 s (p10..p75) in our output | timing good, 26 clips, no team |
| SET PIECES, Corner Kick | ball out over Y's goal line; clip starts during the stoppage | dead from at least 10 s before the clip start until about +5 s; then **X (the taker)** until the first touch by Y | restart at start + 5.5 .. 6.7 s (p25..p75) | timing good; taker confirmed 12 of 13 by the following clips; our output gives the first spell to the taker in 8 of 12 |
| SET PIECES, Goal Kick | ball still on the goal-area line of X | dead until the kick, then X | ball still from 0 to +3 s, kick near +4 s | taker right (8 of 8 by ball position). Our output is D on only 45% of the frames around the start: the largest disagreement found |
| SET PIECES, Free Kick | ball still, then kicked | dead at the start, then X | restart about +6 s (5 clips) | only 5 of the match's free kicks are tagged |
| SET PIECES, Kick Off | 5 s clip around the kick | dead before, restart about +3 s; Team N/A. After a goal the taker is the conceding team (Law 8) | within about 1 s | high, 6 clips; already used (`kickoff.py`) |
| GOALS | from the scoring attack (final-third entry - 10 s) to the next kick-off | X scored. **Dead from the goal to the end of the clip**; the kick-off that follows belongs to Y | end of clip = kick-off clip start + 1.0 s, exact; the goal moment is somewhere 10-32 s into the clip | high, 4 clips; our output already agrees (D on 100% after 40 s) |
| labels | Team = attacker or taker (defender on the mirror codes); Side = corridor of the ball (wings outside about +-18..20 m); Max players in the box = maximum count of X players in Y's box during the clip; Direction of ball entry = not checked | none for possession | | Team N/A on throw-ins, kick-offs and 2 CREATING CHANCES pairs |

Summary of what the tags can and cannot say:

- **Which team**: reliable only for BUILD UP, PROGRESSION and the middle of ATTACKING TRANSITION, all on
  frames where the ball is tracked 92-93% of the time. The zone codes name the team that attacks that
  end, not the team on the ball.
- **When the ball is dead**: at the start of every SET PIECES clip and for about 3 s after it; from
  the goal to the kick-off inside GOALS. Nowhere else: the phase and zone clips cross stoppages.
- **Timing**: exact for zone entry and exit (start + 10 s, end - 5 or - 10 s); within a few seconds for
  the restart (start + 3..7 s) and for the turnover in ATTACKING TRANSITION (start + 1..5 s). No clip
  edge is a possession edge.
- Earlier result that still stands (`.planning/research/smart-tagging.md`, L8): a team prior built
  from the clips lost on every external benchmark, also with a simulated error-free tagger.

## 5. Papers

Verification level is stated per paper. Only the first was read from the PDF; the others were
checked from abstracts and search excerpts (pages could not be opened in this session), so their
numeric thresholds are not confirmed.

| paper | what it says | one rule for a team-level, ID-free, offline labeller with the ball missing 41% |
|---|---|---|
| Vidal-Codina, Evans, El Fakir, Billingham (2022), "Automatic event detection in football using tracking data", Sports Engineering, doi 10.1007/s12283-022-00381-6, arXiv 2202.00804. **Read (local PDF).** | Rules only. Possession zone: "a circular area of radius Rpz around every player, such that if at any given frame the ball is within a player's PZ, then that player is deemed to be in possession"; radii "50 cm for provider A and 1 m for both provider B and provider C ... Rdz = 1 m for all providers"; two opponents inside the duel zone = duel. A gain needs a change of ball direction or speed, otherwise the frames are discarded; a ball that leaves and comes back with no other touch is not a loss. Dead ball: "the ball is dead, signaled by either a binary boolean or by the absence of ball tracking data". Set piece by geometry during the dead interval: kick-off (all players in own halves, one at the centre mark), goal kick (a player in his own goal area), corner (a player within a tolerance of the corner mark), throw-in (a player beyond the touchline), free kick = the rest; confirmed when that player has the ball on the first live frame. Reported set-piece detection 82-100% by type. | The restart taker comes from geometry, not from the ball: **the team of the player standing at the corner mark / beyond the touchline / in his goal area during the stoppage is the team in possession when play resumes.** This gives a team to the 26 throw-ins the tags leave N/A. (Already our base: `r_pz = r_dz = 1.0`.) |
| Morra, Manigrasso, Canto, Gianfrate, Guarino, Lamberti (2020), "Slicing and dicing soccer: automatic detection of complex events from spatio-temporal data", ICIAR 2020, arXiv 2004.04147. Abstract only. | Two tiers: atomic events (possession, kick, ball out) from distances and speeds, then complex events composed with interval temporal logic; synthetic SoccER data. Thresholds not confirmed. | Write the restart as an interval rule: ball out -> dead interval -> first kick by a player of the team entitled to it -> that team until the first touch by the other. |
| Khaustov and Mozgovoy (2020), "Recognizing events in spatiotemporal soccer data", Applied Sciences 10(22):8046, doi 10.3390/app10228046. Abstract and a citing paper. | Rule-based possession, passes, shots from ball speed and direction and the nearest player; F about 0.93 passes, 0.95 shots on 5 matches (secondary source). | A ball in flight stays with the passer's team until an opponent touches it: never switch team on a hidden-ball gap alone. |
| Sanford, Gorji, Hafemann, Pourbabaee, Javan (2020), "Group activity detection from trajectory and video data in soccer", CVPR Workshops, arXiv 2004.10299. Abstract. | Self-attention over player and ball trajectories; events located at under 0.5 s. | 0.5 s is the realistic resolution of an event time from trajectories: do not tune edges finer than the scorer's 0.5 s tolerance. |
| FIFA, "Enhanced Football Intelligence" explanation document v1.1 (2023), fifatrainingcentre.com. Excerpts. | Phases are computed from tracking as shares of ball-in-play time: build-up, progression ("The aim of this attacking phase is to advance ball into the final third"), final third ("in possession of the ball within the attacking third of the pitch"), long ball, counter-attack, transitions. No duration rule for a transition was found. | The same vocabulary and the same zone logic as section 3: the pitch third is the separator. In FIFA's version phases exist only in play; in this file they do not (section 3.4), so a phase tag is not evidence against D. |
| Stats Perform, "Phases of play: an introduction" (statsperform.com); Li and Link (2026), arXiv 2606.09289; OpenSTARLab phase estimation. Excerpts. | Rules-based or learned frame-level phases (transition, build-up, established possession); macro F1 0.87 on 7 Bundesliga matches for the learned one. | A phase label is a function of who has the ball and where: it cannot be more accurate than the possession it was computed from. |
| Lang, Wild, Isenko, Link (2022), "Predicting the in-game status in soccer with machine learning using spatiotemporal player tracking data", Scientific Reports, doi 10.1038/s41598-022-19948-1. Abstract. | In play / out of play from the players only, up to 92% of frames; stoppages longer than 2 s found with 81% precision; 102 matches. | With the ball hidden, decide D from player kinematics, and do not create stoppages shorter than about 2 s. |
| Kim et al. (2023), "Ball trajectory inference from multi-agent sports contexts using set transformer and hierarchical Bi-LSTM", KDD, arXiv 2306.08206. Abstract. | Ball position from player trajectories: error under 3.7 m, possession accuracy 64.7%. | Player-only possession is right two times in three: a prior for hidden frames, never a decision. |
| Ganchev, Graca, Gillenwater, Taskar (2010), "Posterior regularization for structured latent variable models", JMLR 11:2001-2049; Richard et al. (2018) NN-Viterbi, arXiv 1805.06875; Li et al. (2021) timestamp supervision, arXiv 2103.06669. Abstracts. | Coarse or noisy temporal labels enter a sequence model as soft constraints on the posterior or as terms in the Viterbi path score, not as hard labels. | Tags go into the decoder as per-frame log-potentials or as "at least one switch / one D run inside this window" constraints, with a weight fixed on external data. Never clamp a frame. |

## 6. Rule ideas, ranked

Every idea needs a simulator or a rule that runs without tags on `data/external/` before it can be
adopted (the external matches have no Smart Tagging). None was tried here; nothing was tuned. The
numbers of affected clips are for this file and are upper bounds on the effect.

| # | idea | from | size | what is new against TRIED | risk |
|---|---|---|---|---|---|
| 1 | **Restart taker anchored on the stoppage that contains the clip start.** For Corner Kick, Goal Kick, Free Kick: take our D run that covers (or is nearest to) the clip start; the first live spell after it belongs to the tagged taker until the first touch by the other team | 3.3, 3.5, DATA.md:170 | 27 clips; our output disagrees on 10 | T96 / T97 anchored on "the last stoppage that starts inside the 35 s clip" and concluded the clip cannot be anchored. The clip start is inside the stoppage (corner: D on 100% of the 10 s before it) and the restart is 3-8 s later | a short taker spell before a clearance can add a spurious spell; the simulator must place clip starts at restart - 5 s +- 2 s |
| 2 | **Throw-in taker from geometry, gated by the tag.** During a tagged throw-in stoppage, the team of the player nearest the touchline at the ball's exit point takes it | Vidal-Codina 2022; MatchLens `detectors/set_piece_detector.py` (executor from geometry) | 26 clips with Team N/A | T16 guessed "the other team restarts"; here the tag supplies type and time, the players supply the team | ID-free tracks near the touchline are noisy; ball boys and substitutes are not tracked, linesmen may be |
| 3 | **Dead until the kick at tagged goal kicks and free kicks.** Inside the first seconds of such a clip, a ball that is still stays D until it moves | 3.3 (ball still at 0..+3 s), 3.5 (our D only 0.45 there) | 10 + 5 clips | the still-ball rule (L2-11) was switched off because it did not transfer when applied everywhere; here it is gated to stoppages a second source confirms | a keeper playing from the ground in open play is live; the tag itself can be a false positive |
| 4 | **Automatic consistency checks, no tuning.** (a) D from the goal to the kick-off in every GOALS clip; (b) the first team after a Kick Off (After Goal) is the conceding team; (c) a switch to the tagged team within 6 s of an ATTACKING TRANSITION start; (d) D at every SET PIECES start | 3.3, 3.5 | today: (a) 4 of 4, (b) 3 of 3, (c) 22 of 24, (d) corners and throw-ins yes, goal kicks 45% | allowed as "automatic checks using only the provided data"; flags a broken run before submission | none if used as a report only; becomes tuning if versions are chosen by it |
| 5 | **If a team prior from the tags is ever retried: per-code windows instead of whole clips.** PROGRESSION [0, end - 5 s]; BUILD UP [3 s, end]; ATTACKING TRANSITION [5 s, end - 3 s]; the zone codes and CREATING CHANCES only in their first 10 s; all mirror codes dropped | 3.2, 3.3, 3.5 | 142 duplicate clips removed; the windows with 0.28-0.54 agreement removed | T91 used whole clips with one symmetric trim and both members of each mirror pair | the reliable windows are the ones where the ball is tracked and our decoder is already right; the L8 result (no gain even from an error-free tagger) probably still holds |
| 6 | Use zone entry and exit as position evidence for the hidden ball: between start + 10 s and end - 10 s of a BALL IN FINAL THIRD clip the ball is in that third | 3.3 | 93 + 72 clips, ball hidden on 31-35% of their frames | not tried: the tags have only been used as team evidence, never as ball-position evidence | helps the D / live decision more than the team; small |
| 7 | Do not treat "no clip" as dead and do not treat a phase clip as live | 3.4 | | prevents two tempting rules | |

What should not be done: copy a clip as a label (DATA.md:179-180); use a defending code next to its
mirror; read the Team of BALL IN THE BOX, BALL IN FINAL THIRD, PLAYERS IN THE BOX or CREATING CHANCES
as the team on the ball; write any rule on a frame number of this match.

