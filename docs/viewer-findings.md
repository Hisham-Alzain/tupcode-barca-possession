# Viewer findings (L20, 2026-11-17, final draft 17e)

Sources: `viewer/cache/` (evidence.npz rebuilt for the current default, aim.npz, marks.json, restarts.json),
the viewer feature cache (ball, nearest-player distances), Smart Tagging, our labels in
`submissions/draft_2026-11-17e.csv`, and the error-TYPE catalogue of the public clip (ADR-027). No truth
file, `/api/gt` or `/api/score` was used; nothing was scored. Frames are VIDEO frames (type them into the
viewer). Analysis scripts: scratchpad `mine*.py` (not committed).

## Viewer status on 17e

- Caches rebuilt: `viewer_evidence.py` (through ramlock) and `viewer_cache.py` (68 line crossings, 33
  restarts, pan median 0.03 m/s). Evidence labels vs 17e: see "Label check" below.
- Endpoints (python urllib): `/api/campan` 39,863 blocks x 5 frames (last 4 frames of the match dropped,
  harmless), 3 unknown, max 15.9 m/s; `/api/marks` 68 out runs + 33 restarts, sorted; `/api/evidence`
  correct window at both ends and for negative frames, terms bt/touch/dist/rule/fly/shape, `stop` in
  [-20, 20], vit in {0..3}. `node --check` on the page script: OK. No viewer bug found; nothing in
  viewer/ changed.
- Caveats (not bugs, but they can mislead):
  1. `scripts/viewer_evidence.py --check` still defaults to `draft_2026-11-17d.csv`; pass the 17e path.
  2. `restarts.json` comes from the a12 worktree inventory (`bp-wt/a12/eval/results/a12_inventory.json`),
     not from the final `src/possess/setpiece.py`. Two of its 33 restarts lie outside the rule's window
     (182616 is 9.8 s before its tag clip at 182862; 184518 is 3.4 s before its clip at 184603) and look
     like live-play false detections. Throw-in restarts carry no team.
  3. The "D while panning" mark count is 22 on 17e (19 on 17d: 17e added D).

Label check: rerun with `--check submissions/draft_2026-11-17e.csv`: "0 of 199319 frames differ". The
WHY row shows the evidence of the pipeline that produced the final draft.

## Ranked findings

### 1. Flight term alone opens sub-second spells after a dead ball (4-5 in the match; suggests a rule)
What: right after a D spell, the first team spell lasts 0.8-1.0 s, carried only by the flight-direction
term at saturation (|fly| ~ 1.4) with touch = 0 and the learned judge (bt) neutral or favouring the OTHER
team; the next spell is the other team. Typical at the restart kick or throw (ball in flight, no touch).
Where: 64187-64210 (H, next A), 113741-113761 (A, next H, centre spot), 127626-127647 (A, next H,
throw-in at 127599), 187674-187693 (A, next H, throw-in at 187670); also 29785-29797 (A 0.5 s after the
Barcelona corner at 29781, fly -1.4 against rule +0.6). These are 5 of the 7 team spells < 1 s in the
whole match. Across all 108 D -> team starts, fly carries the first 0.5 s in 14 and is the only support in 4.
When active, fly agrees with the final label only 71% of frames (touch 96%, dist 94%, bt 93%).
Why it matters: each is an extra predicted spell (precision) and delays the true spell start by ~0.8 s;
the catalogue lists "short spells" as a remaining error type.
Rule? Yes, testable on external data: "after D, a team spell supported only by fly (touch = 0, bt not
agreeing) for < 1 s is relabelled to the following team / stays D". Check its frequency and sign on
external truth before adopting.

### 2. Proximity-only switches come ~0.6 s before the touch (32 of 218 switches; suggests a rule)
What: switch timing follows the summed margin exactly (the label changes on the frame the sum changes
sign at all 218 H<->A switches). Usually touch flips the sum (median lead 0 frames). In 32 switches
neither touch nor bt supports the new team at the switch frame: 12 driven by dist + rule, 9 by rule
alone, 4 by shape, 4 with nothing (ball hidden). Touch confirms within 2 s in 13 of them, median 14
frames later; 19 never get a confirming touch within 2 s.
Where: both early catalogue switches are this class: 11711 (A, dist + rule, touch 14 frames later; the
catalogue calls it an extra 3.4 s Athletic spell) and 13690 (A, dist + rule, touch 15 frames later; the
catalogue says 0.6 s early). Others: 20961, 21972, 34992, 54529, 81409, 114293, 119552, 142535 (dist +
rule); 19799, 27417, 40278, 77495, 123144, 126329 (rule only).
Term lead over all switches: shape leans to the new team > 2 s early in 50 of 171 switches and rule in
33 of 193, but both are small (p95 |margin| 0.40 and 0.60) and only decide when touch and bt are silent.
Why: early dating and extra spells are the second-largest catalogue error (6.4 s + 3.4 s).
Rule? Testable: "a switch with no touch/bt support waits for the confirming touch (up to ~1 s), and is
dropped if none comes". Overlaps L11-B (switch dating, not adopted: external truth dates at first touch,
organisers at control), so measure on external data per source before any adoption.

### 3. Hidden-ball team frames abutting a stoppage (17 edges >= 2 s, 73 s in the match; rule only after external check)
What: runs of team-labelled frames with no ball position directly before or after a D spell; all terms
~0, so the label is inertia. The catalogue's largest remaining error (6.6 s hidden block labelled Barcelona,
truth Athletic then dead) is this kind.
Where (start or end of the run): 182473 (H 5.7 s hidden, D before, throw-in restart at 182616),
29968 (A 11.2 s), 76697 (A 6.4 s), 69286 (end, 6.8 s), 46811 (end, 5.4 s), 19150 (end, 4.9 s),
163306 (end, 4.4 s), 123468, 168420 (end, 4.0 s), 136590, 147285, 147573, 154988, 146817, 80106,
168973, 69676 (2.0-3.6 s).
Why: if these are dead, each is up to several seconds of team-for-D plus a misplaced boundary.
Rule? Only as a hypothesis: "hidden-ball run adjacent to D -> extend D" must first be measured on
external truth (how often such edges are truly dead). ADR-022/028 history says hidden-ball D rules are
fragile; do not adopt without that run.

### 4. Restarts vs labels: mostly consistent; two isolated cases
- 33 restarts: 31 have D in the second before the kick. The two that do not (182616, 184518) are the
  out-of-window detections in caveat 2, i.e. probably detector noise, not label errors.
- Wrong team after a tagged kick: 37377 (goal kick tagged Barcelona, we label Athletic from the kick
  frame; touch says Athletic 1 s later while bt favours Barcelona) and 29781 (finding 1). One-offs.
- D continuing after the kick: 25541 (1.0 s), 85003 (1.8 s), 120707 (1.5 s), 127599 (1.1 s), and 27803
  / 50996 stay D > 3 s after the detected kick (probably detector timing, ball reappearing). Minor.

### 5. Ball beyond a line with no stoppage (6 runs; noise without ball height)
68 out runs; 12 are not followed by D within 1 s, but 6 of those are the thrower holding the ball at a
throw-in (19772, 44739, 85745, 179457, 182616, 188609: the run IS the restart). The rest:
143368 (4.2 m over the touchline 1.4 s, back inside 1 s later), 180406 (4.7 m, 1.2 s, back inside),
20118 (5.1 m, 0.4 s), 126396 (0.5 m), 143837 (6.3 m then hidden; we keep A 9.4 s, goal-kick tag at
144189: worth a look), 147110 (1.5 m over the goal line, D 5.8 s later, corner tag nearby).
The ball returning inside within a second points to aerial balls projected outside, not real exits. Not a
rule; 143837 is the only one worth a look in the video.

### 6. Camera "D while panning" marks: noise as a detector (22 marks)
None shows clear live play called dead from input alone. 12 are ball-hidden D spells of 12-221 s, three
with GOALS tags (47100, 74350, 158025): replays, celebrations and the operator re-aiming; the homography
pan there means nothing. Visible-ball cases: 15940 and 18075 sit inside D spells the catalogue confirms as
dead (the camera pans during dead ball), 37190 is the ADR-029 goal kick (ball being placed), 25375 and
39070 are throw-in preparations (out run then throw restart), 42135, 53400, 173945 have the ball at rest
with no player within 4 m. Only 69815-69944 (ball 38% visible, moving 1.9 m/s, player 0.7 m away, in the
box) and 72430-72604 (ball 14% visible, player 0.4 m) are worth a look. Not a rule.

### 7. Other observations
- No term is dead weight: fly and dist are zero in 79% / 83% of in-play frames but decisive when active.
- No D spell < 1 s survives; team spells < 1 s: 7 (5 in finding 1, 37060-61 the known ADR-029 2-frame
  defect, 120465-66 a 2-frame A spell at a hidden-ball edge).
- Post-passes change 6,377 frames relative to the Viterbi path: 2,016 team -> D, 334 D -> team,
  1,094 H <-> A.
- Strong evidence against the label for >= 2 s happens once (17386-17459, H while the sum says A; Viterbi
  said D there, a post-pass made it H; inside the catalogue's 3.2 s "dead labelled H" at 17380-17461).

## Worth a rule (external validation first)
1. Fly-only sub-second spell after D (finding 1): small, clean, 4-5 cases, cheap to test.
2. Touch-confirmation for proximity-only switches (finding 2): 32 cases, overlaps L11-B.
3. Hidden edge next to D (finding 3): largest seconds, highest risk.
Findings 4-6 are one-offs or noise.
