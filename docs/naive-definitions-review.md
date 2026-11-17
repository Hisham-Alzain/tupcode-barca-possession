# Naive definitions review (L6) — our system against "Barcalona Hackathon.docx"

Date: 2026-11-16. Reviewer: cold, no prior project context. Read-only on code and data.

**How this was done.** The reference document was read locally (unzipped `word/document.xml`; line numbers below are lines of that plain-text extraction, 522 lines). Our behaviour was read from the code (`src/possess/`), not from our docs. The only data touched was our own output `submissions/draft_2026-11-15d.csv` (label counts). No ground-truth file was opened, no pipeline was run, nothing was uploaded. Earlier reviews, ADRs and research notes were not read.

**Caveats.**
- The document looks like a second-hand summary of the organisers' repository, not their original text (line 489 reads "which is tomorrow (your project notes say 23:59 CET)"). Where it matters, check the wording against the organisers' own files.
- Not verified by me: the H/A team mapping in `epts.py`, and the internals of `stoppage_ml.py`, `shape.py` (models), `edges.py` and `ballclean.py` beyond their docstrings and switches.
- "Active value" means the value in the default configuration: `pipeline.DEC = dict(DEC_BASE, **DEC_SHAPE)` (`pipeline.py:34-37`, `shape.py:521-606`).

## 1. Verdict counts

| Verdict | Count |
|---|---|
| MATCH | 19 |
| PARTIAL | 13 |
| MISMATCH | 1 |
| NOT COVERED | 3 |
| AMBIGUOUS-IN-DOC | 3 |

## 2. Numbers from our draft (our output only)

| Quantity | Our draft 15d | Document |
|---|---|---|
| Rows | 199,319 | 199,319 (line 419) |
| X on graded frames | 0 | allowed, never required |
| X outside graded frames | 40,462 (all of them) | free (line 22) |
| Graded H / A / D | 51,117 / 36,428 / 71,312 (D = 44.9%) | not published |
| Possessions, half 1 / half 2 | 179 / 192 = 371, mean 9.4 s | not published |
| Half 1 first label | A at 11,185 (no leading D) | "graded from kickoff itself" (line 433) |
| Half 2 first live frame | 113,741 | kick-off at 113,742 (line 432); 1 frame is free |
| Public window (11,185–18,684): possessions | 21, mean 12.7 s | 17, mean 14.2 s (line 304) |
| Public window: D frames | 821 (10.9%) | about 1,465 (19.5%), derived as 7,500 − 17 × 14.2 s × 25 |

The last two rows use only statistics printed in the document. They suggest we split possessions more often than the truth does and call less dead ball in that window. Seventeen possessions is a small sample, and the rules forbid using it to choose anything; it is reported as a pipeline check only.

## 3. Definition table

### 3a. Labels and states

| # | Document (quote, line) | What the code does | Verdict | Example |
|---|---|---|---|---|
| 1 | "H home team in possession, ball alive / A away team" (10–13); "Home is Barcelona (H)" (113) | Team 1 → `d_home`, team 2 → `d_away` (`features.py:52-58`); decoder states H, A (`decode.py:55`) | MATCH (mapping not verified by me) | — |
| 2 | "D dead ball — the ball is not in play" (14–15) | D comes from proxies, not from the Laws: ball far from everyone and near a line, or teams slow, or off the pitch (`deadball2.py:88-90`, `DEAD2` in `pipeline.py:29-30`), plus a player-only stoppage model (`shape.py`), soft on visible frames (`sdead_vis=2.5`, `w_live=0.0`) | PARTIAL | Foul in midfield, whistle, ball stays at a player's feet for 20 s: the document says D from the whistle; we say D only once the players' slowdown outweighs a ball sitting at a player's feet |
| 3 | "X no answer … or the frame has no ground truth" (16–17); "Before kickoff, halftime and after the final whistle are never scored" (22) | X on every frame outside `HALVES` (`pipeline.py:132-137`) | MATCH | — |
| 4 | "Only three states: H, A or D. There is no contested, transition or unknown." (58) | Viterbi emits only H/A/D inside a half (`decode.py:468`); 0 X on graded frames | MATCH | — |
| 5 | "Dead ball is always D … don't copy that team label onto dead-ball frames" (34) | No ball-owner field is read from the tracking; D is one label (the DH/DA split is internal only, `decode.py:55,468`) | MATCH | — |
| 6 | "While it is live, one team always has possession" (42) | No live frame without a team | MATCH | — |
| 7 | "A possession is a maximal run of one team with the ball alive: H H H A A is 2" (60–61); "Dead ball is never a possession" (321) | Same counting (`ctl_truth.py:109-115`); each half decoded separately (`decode.py:503-516`) | MATCH | — |

### 3b. When possession changes

| # | Document (quote, line) | What the code does | Verdict | Example |
|---|---|---|---|---|
| 8 | "Change of possession: an opponent gains controlled possession of the ball" (38) | A switch costs `c_switch=4.0` (`pipeline.py:34`); a team spell under `ctl_s=1.0` s that follows the other team's live spell is given back (`shape.py:606`, `ctl_truth.py:39-76`) | PARTIAL | Athletic clear the ball, it runs loose for 1.5 s, Barcelona pick it up again. Document: one Barcelona possession. Us: H, A (1.5 s), H = three possessions |
| 9 | "'Controlled' … a player keeps the ball for a few seconds" (47–48) | Threshold is 1.0 s, not "a few seconds" | PARTIAL | As row 8 |
| 10 | "a team completes more than two passes in a row" (49) | The pass test exists (`min_pass`, `ctl_truth.py:47-52`) but only for external truth with event data; predictions never count passes or touches | NOT COVERED | Three one-touch Athletic passes in 0.9 s, then a tackle: document = Athletic possession; we erase it |
| 11 | "NOT changes: a touch without control" (51–52) | Removed only when the resulting spell is under 1 s. A touch starts a hold of up to 4 s for the toucher (`max_hold_s=4.0`, `decode.py:189, 285-287`), which can itself stretch the spell past 1 s. The purpose-built switch `confirm_s` exists but is off (default 0.0, `decode.py:193`; not in `DEC_SHAPE`) | PARTIAL | Athletic defender toes the ball away, it rolls free 2 s, Barcelona recover: document H throughout; we likely give A for those 2 s |
| 12 | "a deflection that doesn't intercept the pass" (53) | Same mechanism as row 11. Works when the pass still arrives within 1 s | PARTIAL | Clipped pass that arrives 0.4 s later: both say H. Clipped pass that loops 1.5 s before arriving: we may say A in between |
| 13 | "an aerial duel won without that team then controlling the ball" (54) | No ball height. A header is an ordinary touch. Contested ones are ignored (`duel=2`, `decode.py:254, 285`); an uncontested header is treated as row 11 | PARTIAL | Athletic centre-back heads a goal kick clear unchallenged, Barcelona collect 2 s later: document H; we may give A for 2 s |
| 14 | "When the ball is loose or contested, possession stays with the last team that had it" (43) | Contested: yes (`duel=2`; frame rule keeps the team, `team_control.py:60-65`). Loose: only by the switch cost. The distance score gives up to ln 5 = 1.6 per frame to the team within 1 m (`cap=1.0`, `d0=0.25`, `decode.py:63-66`), so about 5 frames beside the ball pay for a switch and back; the hold lapses after 4 s; on hidden frames the shape model decides | PARTIAL | Loose ball rolls past an Athletic player for 0.3 s without a touch: we can open an A spell; under 1 s it is erased, over 1 s it stays |
| 15 | Date of the change: at first touch or at control? "an opponent gains controlled possession" (38) | A kept spell starts at its first frame ("a kept interception starts at its first touch", `ctl_truth.py:20`); frame-rule switches are back-dated (`backdate=True`) | AMBIGUOUS-IN-DOC | Interception at t, control at t + 1.5 s. We switch at t. If the truth switches at control we are 1.5 s early; "1 s late" scores 0.65 (line 334). Only the public truth can settle this |
| 16 | "The public ground truth is the reference … the ground truth wins" (56) | Not opened by me. Printed statistics against our draft: 21 vs 17 possessions, about 821 vs 1,465 D frames | PARTIAL | See section 2 |

### 3c. Dead ball and restarts

| # | Document (quote, line) | What the code does | Verdict | Example |
|---|---|---|---|---|
| 17 | "The referee decides when the ball is live" (41) — start of D | D starts where dead evidence began (back-dated 50 frames, `deadball2.py:101-103`, `on_frames=50`) or where the stoppage model's score crosses zero | PARTIAL | Ball crosses the touchline while hidden: the document's D starts at the line; ours where the players' slowdown shows, often later |
| 18 | Same — end of D | Hidden restarts of stoppages ≥ 3 s are moved to the largest step-up of fast-player speed within ±2 s (`ed_exit=1`, `shape.py:565`). The detector's own "live" is a player within 2 m of the ball for 5 frames (`r_near=2.0`, `off_frames=5`; `v_kick` off), which is the taker arriving, not the kick; it is only soft evidence (`w_live=0.0`) | PARTIAL | Throw-in: taker holds the ball 6 s. Document: D until release. We: live once players accelerate, which can be before or after the release |
| 19 | "After a restart the possessor is always clear" (45) | The restart team is not read from anything; it is inferred. Staying with the pre-stoppage team costs `c_same=1.0` extra (`pipeline.py:34`, `decode.py:428-448`) and the frame rule flips the team after every stoppage (`restart_flip=True`, `pipeline.py:31`). Both rest on "the other team restarts after 78–90% of stoppages", measured on external truth where every deflection is its own spell | PARTIAL | Barcelona attack, Athletic block the cross behind, Barcelona corner. Document: H, D, H (same team). External-truth view: H, A (the block), D, H = "other team restarts". Our prior leans to A after the stoppage |
| 20 | "set pieces mark the end of dead-ball stretches"; "team tags hint at possession" (161); types: corner 12, goal kick 10, free kick 5, kick-off after goal 4, kick-off start 2, throw-in 26 with Team = N/A (164, 222) | Only "Kick Off (Start)" is read (`kickoff.py:39`). The other 57 set-piece tags, their Team value and the 4 goals are unused | NOT COVERED | A corner tag naming Barcelona: the document treats it as a hint that D ends there and H follows; we ignore it |
| 21 | "The second half is graded from frame 110,808 but kicks off only at 113,742. Those 2,934 frames are graded and all D." (432) | Frames from the graded start to the kick-off are written D (`pipeline.py:153-166`); kick-off = frame inside the tag window with the ball nearest the centre spot (`kickoff.py:48-61`) | MATCH | Our first live frame is 113,741 |
| 22 | "The first half is graded from kickoff itself" (433) | No pre-kick-off span unless the tag starts after the graded start (`kickoff.py:74-79`); draft starts with A at 11,185 | MATCH | — |
| 23 | Goal → dead until the kick-off (implied; "GOALS 4", "Kick Off (After Goal) 4") | No specific rule; relies on the ball off the pitch and the stoppage model | NOT COVERED | After a goal the ball is carried back to the centre: we depend on the slowdown model to keep D for the whole celebration |

### 3d. Frames, data and format

| # | Document (quote, line) | What the code does | Verdict |
|---|---|---|---|
| 24 | "Tracking: video_frame = frame_id − 1" (243) | `rows[a:b+1] = lab[a+1:b+2]` (`pipeline.py:136`), `_period_of` (`pipeline.py:85-90`) | MATCH |
| 25 | "Smart Tagging: video_frame = round(seconds × 25)" (244); match case-insensitively (157) | `int(round(float(start) * fps))`, labels lower-cased (`kickoff.py:33-43`) | MATCH |
| 26 | Graded frames 11,185–90,651 and 110,808–190,197 (267–268) | `HALVES` (`pipeline.py:40`) | MATCH |
| 27 | "105.3 × 68.0 m according to games.py, 105 × 68 according to the metadata" (147) | 105.0 × 68.0 (`epts.py:16`) | AMBIGUOUS-IN-DOC (0.3% in length, negligible against 1–2 m radii) |
| 28 | No player identities, work at team level (149) | Team-level throughout | MATCH |
| 29 | Ball missing = NaN; "design for missing data" (151) | A hidden ball is no evidence, never D by itself (`deadball2.py:11-12`) | MATCH |
| 30 | Homography (233–236), broadcast (238–240) | Not used | n/a (optional inputs, not counted) |
| 31 | Header `frame,label`, 199,319 rows, frames 0…199,318, no blanks, UTF-8 (461–465) | `write_submission` (`epts.py:157-162`); draft has 199,319 rows | MATCH |

### 3e. Scoring

| # | Document (quote, line) | What we do | Verdict |
|---|---|---|---|
| 32 | Ranking by spellF1, macro-F1 tie-break (108) | Official scorer vendored in `vendor/mtah/possession_metrics.py` | MATCH |
| 33 | X on graded frames: miss on macro-F1, previous label on spellF1; "at a change of possession, guess" (353–356) | Never X on graded frames; always a guess | MATCH |
| 34 | Lag: 1 frame free, 3 frames 0.98, 8 frames 0.89, 1 s 0.65 (329–337) | Known timing weak points: a turnover inside a hidden block is not placed (`gap_pos=-1.0` in `DEC_SHAPE` overrides 0.5); hidden restarts are snapped within ±2 s | PARTIAL |
| 35 | "possession_metrics.py defines every number" (402). `run_evaluation.py` "checks format", `validate_submission.py` "scores" (408–410), but line 467 says `validate_submission.py` applies the official checks | We have `validate_submission.py` and no `run_evaluation.py` | AMBIGUOUS-IN-DOC (the two descriptions look swapped) |

### 3f. Rules and deliverables

| # | Document (quote, line) | What we do | Verdict |
|---|---|---|---|
| 36 | Public 5 minutes: "only to check your pipeline works … not for ranking, training or tuning" (306, 386) | Stated the same in `docs/RULES.md` and CLAUDE.md | MATCH |
| 37 | "No hand labels on the provided game"; external data from other games is fine (95–97) | Training and tuning data are Metrica and IDSSE (TECHNICAL §5) | MATCH |
| 38 | "Uploading any competition material (videos, data, extracts) to a third-party hosted or cloud LLM. The Terms and Conditions forbid it." (452) | `docs/RULES.md` treats this as superseded by an organisers' email of 11 Nov ("Yes you can use agents to read raw data"); AI agents read `data/challenge/` | MISMATCH with this document. I cannot see the email. The document's last word is "open an issue and ask" (456) |
| 39 | Repository: source, pinned dependencies, the command that regenerates the file, no match data (472–474) | `uv.lock`, `uv run possess …`, `data/` and `submissions/` git-ignored | MATCH |
| 40 | Technical document must cover approach, data sources used and dropped, training data, "Possession rules you implemented: loose balls, challenges, restarts", limitations, reproduction (475–481) | All sections exist, but the draft is stale against the code (section 4) | PARTIAL |

## 4. Our docs against our code

| # | Doc statement | Code | Fix |
|---|---|---|---|
| C1 | TECHNICAL §4: "Smart Tagging: no (yet)". README: "Only the tactical tracking is read". `pipeline.run_challenge` docstring: "tactical tracking only" | `kickoff.py` reads the Smart Tagging XML for the kick-off tags (`pipeline.py:145-149`) | Say Smart Tagging is used for kick-offs only (the explainer already does) |
| C2 | TECHNICAL §3.5 and explainer: turnovers inside ball gaps are "placed mid-gap" / "in the middle of the stretch" | `DEC_SHAPE` sets `gap_pos=-1.0` (`shape.py:521`), which overrides `DEC_BASE`'s 0.5; placement is left to the shape scores and the tie-break | Correct both docs |
| C3 | TECHNICAL §6 lists four rules | Missing: the 1 s controlled-possession rule (`ctl_s=1.0`), D before the kick-off, the reappearing-ball rule (`reappear=1.0`), contact-point touches (`vtx=2.0`) | Add them; the organisers ask for exactly this section |
| C4 | TECHNICAL §6: a touch is "within 1 m of one team" | On a low-passed ball track (the provided match) the radius is 2 m (`SMOOTH_TOUCH`, `decode.py:126`); §7 says so later | Align §6 |
| C5 | TECHNICAL §7 results table ends at the per-match dead-ball prior | Four later default changes (reappear, vtx, ctl_s, kick-off) have no row | Add rows with real numbers |
| C6 | TECHNICAL §7: "17 vs 17 possessions" in the public window | Draft 15d has 21 | Update |
| C7 | Explainer: practice truth re-marked with "about two seconds" of control | The rule applied to our predictions is 1.0 s | Say both numbers, or one will read as the other |
| C8 | README pipeline table has 6 modules and says attacking direction was not brought over | The default path also runs `decode.py`, `shape.py`, `ballclean.py`, `edges.py`, `ctl_truth.py`, `kickoff.py`; TECHNICAL §3 mentions an attacking-direction vote | Update the README table |
| C9 | CLAUDE.md says the scorer is in `vendor/mtah/` | `kickoff.py` and the README also cite `third_party/mtah/`; both folders exist | Harmless; cite one |
| C10 | TECHNICAL has `<TEAM NAME>`, `<PUBLIC REPO URL>` and a `[FINALISE …]` note about the organisers' answer on the per-match prior | — | Resolve before submitting |
| C11 | `docs/RULES.md`: "Not allowed: checking our output against the video" | README describes a viewer that shows the video with our labels. The document itself suggests the homography "to look at your own mistakes" (512) | Decide which statement stands and say so |

## 5. Top issues by expected score impact

Ordering is my judgement from the scorer's behaviour (one extra short spell inside a true possession breaks one match and adds false ones). None of it is measured.

1. **Control threshold too permissive (rows 8, 9, 11–14).** 371 possessions with a 9.4 s mean; the public window's printed mean is 14.2 s and we have 21 spells where the truth has 17. The 1–4 s opponent spell after a single touch is the typical case.
2. **Restart team (row 19).** The "other team restarts" prior was measured under a different definition of possession. A wrong team after a stoppage loses the whole first possession.
3. **Dead-ball extent (rows 2, 17, 18).** About 821 D frames against about 1,465 in the public window. Every D edge is also the start or end of a possession.
4. **Set-piece tags unused (rows 20, 23).** About 31 tags that name a team (corners, goal kicks, free kicks, kick-offs after goals) could fix both the restart frame and the restart team.
5. **Timing of hidden turnovers (row 34).**

## 6. Plan for each non-MATCH

| Item | Smallest change | Risk | Checkable on external data? |
|---|---|---|---|
| Rows 8, 9 | Raise `ctl_s` from 1.0 towards 2–3 s (`--dec ctl_s=…`). Declare beforehand which external truth version stands for the document's "a few seconds" (T = 2 or 3) and choose on that alone | Erases real short possessions (interception, then lost at once). The code note says 2.0 loses on the T = 1 truth | Yes: existing controlled-truth bench. Never choose from the public window |
| Rows 11–13 | Switch on `confirm_s` (already written: a touch answered by the holder within the window changes nothing), and/or let an unconfirmed touch hold for less than 4 s | Interacts with `ctl_s`; two new values to choose | Yes, same bench |
| Row 10 | Count same-team touches in a spell: 3 or more makes it controlled even when under `ctl_s` (reuse `touch_frames`) | Touch detection is weak on the smoothed ball; small gain | Yes |
| Row 14 | Probably covered by the two rows above; otherwise no distance evidence for the non-holder without a touch | Could delay real gains | Yes |
| Row 15 | None now. Read the public truth for understanding only (allowed) and write down which reading it supports | Turning that reading into a setting would be tuning | Partly: external controlled truth can be built both ways |
| Row 19 | Re-measure the "other team restarts" share on the controlled external truth (`ctl_truth.relabel`, then count stoppage flanks). If it falls towards half, set `c_same=0` and `restart_flip=False` | Low; one count and one bench run | Yes |
| Rows 20, 23 | Soft use of set-piece tags: the tag start as an end-of-stoppage hint, the tag Team as a restart-team score for corners, goal kicks, free kicks and kick-offs after goals | What "Team" means per tag type cannot be checked without the video or the truth; external matches carry no Smart Tagging, only event feeds from which tags could be imitated | Weakly. Highest potential, highest rules and semantics risk one day before the deadline |
| Rows 2, 17, 18 | No small change. Report the D share and keep the limitation in the document | — | Already benched |
| Row 34 | With shape evidence, placement inside a hidden block is already data-driven; fix the docs (C2) | — | Yes |
| Row 16 | Keep reporting the public window per version; do not act on it | — | n/a |
| Row 38 | Keep the email on file, keep the disclosure in TECHNICAL §9, and ask the organisers to confirm in writing, since their FAQ text still forbids it | Disqualification risk if the email is read more narrowly than we read it | n/a |
| Row 40, C1–C11 | Bring TECHNICAL.md, README and the explainer in line with the code | None | n/a |
| Rows 27, 35 | Nothing | — | — |

## 7. Agreement with the earlier review

Written after the verdict above. Only the summary table of `docs/organiser-definitions-review.md` (lines 23–34) was read; its inventory, the ADRs and the research notes were not.

| Earlier item | This review | Agreement |
|---|---|---|
| 1. Controlled possession (was DIFFERENT, in progress) | Rows 8–14: now PARTIAL. The 1 s rule is in the default, but 1 s is short of "a few seconds", `confirm_s` is off and the pass proxy is absent | Agree on the issue; I rate the fix as incomplete |
| 2. Second half before the kick-off (was DIFFERENT) | Rows 21–22: MATCH; the draft's first live frame is 113,741 | Fixed |
| 3. Restart possessor (PARTIAL / UNKNOWN) | Row 19 PARTIAL, row 20 NOT COVERED | Agree. New here: the "other team restarts" prior was measured under the old definition and should be re-measured on the controlled truth |
| 4. Gates defined on the old truth | Not examined | — |
| 5. Technical document | Row 40 and C1–C11 | Agree, with a longer list of stale statements |
| 6. Hosted-LLM clause against the email | Row 38 | Agree; still open |

New in this review: the possession-count and D-share comparison with the document's printed public-window statistics (section 2), the unused 4 s touch hold interaction (row 11), the unused set-piece and goal tags (rows 20, 23), and the doc/code contradictions C1, C2, C6, C7, C11.
