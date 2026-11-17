# Organiser definitions vs our system — review and plan

Date: 2026-11-15 · Reviewer: Fable (read-only audit, envelope `L4-definitions-review-fable`) · Deadline: 2026-11-17 23:59 CET

User request: "I want my system to follow the organizers' definitions in every term."

## 0. Scope, sources and what was not available

**Read (organiser text):** `third_party/mtah/` `CHALLENGE.md`, `OVERVIEW.md`, `DATA.md`, `EVALUATION.md`, `FAQ.md`, `RULES.md`, `SUBMISSION.md`, `README.md`, `games.py`, `evaluation/possession_metrics.py`, `starter/baseline.py`, `starter/load_smart_tagging.py`. Below, `mtah/` stands for `third_party/mtah/`.

**Read (emails):** only the transcript in `docs/RULES.md` and `.planning/DECISIONS.md` ADR-001, plus our unsent question `.planning/organiser-question.md`.

**Not available:**
- Gmail is not connected and Google Drive holds no hackathon files. The original emails were not read.
- Things that need the original emails or the signed terms (section 5): the full text of both "Clarification of the rules" emails, the submission-instructions email expected on 2026-11-14 (`docs/RULES.md:42`), the Google Form URL and any instruction in it, and the Terms and Conditions.

**Not opened:** `third_party/mtah/evaluation/gt/*` and every public truth file. Numbers about the public window below are the ones the organisers print in their own documents.

**Not run:** no bench, no pipeline, no tracking or external data loaded. One check went beyond code and docs: label counts of our own output `submissions/draft_2026-11-15a.csv` around the two kick-offs (row D1). It reads no input data and no truth.

**Verdicts:** MATCH · PARTIAL · DIFFERENT · UNKNOWN (cannot be decided from text and code). **Impact** is the expected effect on spellF1 (high / med / low), my judgement unless a number is given.

## 1. Summary

| Rank | Difference | Verdict | Impact | Status |
|---|---|---|---|---|
| 1 | "Controlled" possession: a touch, a non-intercepting deflection or an aerial duel without control is not a change. Our pipeline and our external truth both count them. | DIFFERENT | high | in progress (`L4-control-definition-fable`, worktree `bp-wt/a5`) |
| 2 | Second half, frames 110,808–113,741: organisers say graded and all `D`. Our current draft writes 1,455 `H` + 44 `A` there, as 6 possessions. | DIFFERENT (verified on our draft) | low on spellF1 (about +0.004, estimate), certain; helps the macroF1 tie-break | not started |
| 3 | Restart possessor ("there is a clear possessor" at every restart). We use a soft prior for the other team and no set-piece information; external truth may carry the old team for a few frames after a restart. | PARTIAL / UNKNOWN | med | not started |
| 4 | Our acceptance gates (ADR-016) are defined on the old external truth, which rewards the behaviour the organisers exclude. | process gap | high (decides what gets adopted) | needs an ADR before item 1 is merged |
| 5 | Technical document: the "possession rules you implemented" section is not in the organisers' vocabulary and the labelled-data table lacks "who labelled it and how much". | PARTIAL | none on score; required deliverable | not started |
| 6 | Terms clause against giving the data to hosted LLMs versus the email "Yes you can use agents to read raw data". | UNKNOWN | compliance, not score | needs the original email and the terms |

Everything else in the inventory is MATCH or low impact.

## 2. Inventory

### A. Labels and ball state

| # | Term | Organiser text (verbatim) | Our pipeline | External truth / bench | Verdict | Impact |
|---|---|---|---|---|---|---|
| A1 | `H` / `A` | "`H` \| **home** team in possession, ball alive" / "`A` \| **away** team in possession, ball alive" (`mtah/CHALLENGE.md:22-23`). "Home is Barcelona (`H`), away is Athletic (`A`)." (`mtah/DATA.md:4`) | Home = the team whose `ground` ends with "home" in the ATD metadata → team 1 → `H` (`src/possess/epts.py:44-47`; `src/possess/features.py:65`; `src/possess/decode.py:447`). Draft row 11185 is `A`, as in the organisers' README example (`mtah/README.md:19`). | Metrica: `Team` "home" → 1 (`src/possess/metrica_csv.py:312-315`). IDSSE: `Role="home"` → 1 (`scripts/idsse_cache.py:48-49`, `:64-65`). | MATCH | — |
| A2 | `D` | "`D` \| **dead ball** — the ball is not in play" (`mtah/CHALLENGE.md:24`). "Dead ball is a stoppage, never a possession." (`mtah/CHALLENGE.md:96`) | Internal states `DH`, `DA` both written as `D` (`src/possess/decode.py:55`, `:447`). | Metrica: ball absent → `D` (`scripts/bench.py:101-102`). IDSSE: `BallStatus 0` → `D` (`scripts/idsse_cache.py:59-67`). | MATCH | — |
| A3 | Who decides in play / dead | "The referee determines when the ball is in play." (`mtah/CHALLENGE.md:62`) | Inferred from geometry: ball far from every player and near a line or both teams slow, or outside the pitch (`src/possess/deadball2.py:10-12`, `:89`), MatchLens detector, player-only stoppage model on hidden frames (`src/possess/decode.py:368-387`). A missing ball alone is never dead (`docs/TECHNICAL.md:45`). | Both sources carry a provider in-play flag, the same concept. | PARTIAL (same concept; our evidence is indirect) | med |
| A4 | Dead-ball ownership | "**Dead-ball ownership is deliberately not asked for.** … When the ball is dead the answer is `D`." (`mtah/CHALLENGE.md:42-45`) | The team before a stoppage is kept only inside the decoder to price the restart; output is `D` (`src/possess/decode.py:49-50`, `:447`). | `D` only. | MATCH | — |
| A5 | Three states | "Every graded frame is `H`, `A` or `D`. There is no \"contested\", no \"transition\" and no \"unknown\" in the ground truth." (`mtah/CHALLENGE.md:91-92`) | Viterbi writes only H / A / D on graded frames (`src/possess/decode.py:447`, `:475-486`). | Same three labels. | MATCH | — |
| A6 | `X` | "`X` \| **no answer** — your system declines to guess, or the frame has no ground truth" (`mtah/CHALLENGE.md:25`). "`spellF1` … never sees it: the scorer fills an abstention with the last label you gave in that half" (`mtah/CHALLENGE.md:31-33`). "`macroF1` charges every `X` on a graded frame as a miss" (`mtah/EVALUATION.md:83`). | `X` only outside the graded halves (`src/possess/pipeline.py:132-137`); never on a graded frame. Equal to abstaining on spellF1, better on the macroF1 tie-break. | Bench scores with the official fill (identical scorer, row E1). | MATCH | — |

### B. Possession while the ball is live

| # | Term | Organiser text (verbatim) | Our pipeline | External truth / bench | Verdict | Impact |
|---|---|---|---|---|---|---|
| B1 | Always one team in possession | "there is always one team in possession while play is live" (`mtah/CHALLENGE.md:56-57`) | Every live frame is `H` or `A` (`src/possess/decode.py:447`). | Metrica: team of the last event. IDSSE: `BallPossession`. | MATCH | — |
| B2 | Change of possession | "A change of possession occurs when a player from the opposing team gains **controlled** possession of the ball." (`mtah/CHALLENGE.md:55-56`) | A switch needs evidence above `c_switch = 4.0` (`src/possess/pipeline.py:34`). About three frames of an opponent alone at the ball, or one detected touch (`w_touch = 0.7` for up to 4 s, `src/possess/decode.py:188-278`), is enough. No control test. | Metrica: any event by the other team switches at once (`scripts/bench.py:92-95`). IDSSE: provider flag as is. "Half of all possessions last under 4 s … 588 of 1,198" (`docs/TECHNICAL.md:17`). | DIFFERENT | high (covered by `L4-control-definition-fable`) |
| B3 | Proxies for control | "a player keeps the ball for a few seconds" / "a team completes more than two consecutive passes" (`mtah/CHALLENGE.md:73-74`) | Not implemented. The nearest thing: a gain commits after `k = 5` control frames, 0.2 s (`src/possess/pipeline.py:31`; `src/possess/team_control.py:7-11`). | Not implemented. | DIFFERENT | high (same work) |
| B4 | Touch without control | "a simple touch of the ball without control" does not count (`mtah/CHALLENGE.md:78`) | A touch by the other team starts a hold for that team (`src/possess/decode.py:267-268`); the docstring says so: "a one-touch deflection or tackle is a possession of its own" (`src/possess/decode.py:198-200`). | Counted as a possession. | DIFFERENT | high (same work) |
| B5 | Deflection | "a deflection of a pass that does not intercept it" does not count (`mtah/CHALLENGE.md:79`) | As B4. The user's rule (ball back to the passing team → no switch; kept by the interceptor → switch from the first touch) is the brief of the same envelope. | Counted. | DIFFERENT | high (same work) |
| B6 | Aerial duel | "an aerial duel won without subsequent control of the ball by that team" does not count (`mtah/CHALLENGE.md:80`) | No ball height in the tracking. An uncontested header is a touch like any other (B4). A contested one (both teams within the touch radius) changes nothing, `duel = 2` (`src/possess/decode.py:207-214`, `:267`). | Metrica: `CHALLENGE` events, aerial ones included, keep the previous team (`scripts/bench.py:40`, `:96-97`): matches. IDSSE: unknown. | PARTIAL | med (same work) |
| B7 | Contested or loose ball | "In phases where the ball is contested or loose, possession stays with the last team that had it." (`mtah/CHALLENGE.md:63-64`) | Duel, nobody in a zone and hidden ball keep the current team (`src/possess/team_control.py:12-13`, `:59-83`); the decoder is sticky; contested touches keep the last clean toucher. But a loose ball that rolls to within 1 m of an opponent switches after a few frames (B2). | Metrica: `BALL LOST`, `BALL OUT`, `CHALLENGE` keep the previous team (`scripts/bench.py:8-9`, `:40`). IDSSE: unknown. | PARTIAL | med |
| B8 | Starting point after a stoppage | "Whenever play restarts after a stoppage, there is a clear possessor." (`mtah/CHALLENGE.md:66`) | Soft prior only: restarting with the same team costs `c_same = 1.0` more (`src/possess/pipeline.py:34`; `src/possess/decode.py:21-22`, `:407`), then ball and player evidence decide; the first visible frame after a hidden block gives the ball to the only team within 1 m (`src/possess/decode.py:218-222`, `:269-270`). Set-piece tags are not used (`docs/TECHNICAL.md:32`). | Metrica: the team changes at the first event frame after the ball reappears; if the ball is present before that event, the old team is carried for those frames. Not measured. IDSSE: provider flag. | PARTIAL (pipeline) / UNKNOWN (truth) | med |
| B9 | A possession | "A possession is a maximal run of one team with the ball alive. `H H H A A` is two possessions. `H H D D H H` is two possessions with a stoppage between them, not one." (`mtah/CHALLENGE.md:93-95`) | Counted by the official scorer; the segment clean-up rules that would merge runs are off by default (`src/possess/cleanup.py:20-21`). | Same scorer (E1). | MATCH | — |
| B10 | Reference when text and labels disagree | "if your reading of the definition disagrees with it, the ground truth is right." (`mtah/CHALLENGE.md:86-87`) | Public window scored and reported, never used to choose (`docs/RULES.md:35-37`). | — | MATCH (by rule) | — |

**Organiser-published numbers on the public window, for understanding only, never a tuning target.** 17 possessions of mean 14.2 s, `H` 3,454 · `A` 2,577 · `D` 1,469 (`mtah/DATA.md:276-277`), 21 segments (`mtah/README.md:68`). That is 4.2 possessions per minute of live play. Our draft has 412 possessions of mean 8.6 s (`.planning/DECISIONS.md:184`), about 6.9 per live minute. The direction agrees with B2–B5 (we split more than the organisers' labels do). The sample is 17 possessions from the opening minutes, so it shows direction, not size.

### C. Stoppage timing

| # | Term | Organiser text (verbatim) | Our pipeline | External truth / bench | Verdict | Impact |
|---|---|---|---|---|---|---|
| C1 | When a stoppage starts and ends | "a frame is `D` where the corrected tracking marks the ball dead" (`mtah/CHALLENGE.md:98-99`). No rule for the exact frame (whistle, ball over the line, restart kick) is given anywhere. | Start back-dated to where the dead evidence began (`src/possess/deadball2.py:105-107`); hidden restarts moved to the players' speed-up (`docs/TECHNICAL.md:75`). | Provider flags; `real1` / `real2` add a moving dead ball and hiding tied to stoppages (`.planning/DECISIONS.md:141`). | UNKNOWN | med ("perfect but 1 s late \| 0.6539", `mtah/EVALUATION.md:69`) |
| C2 | Kick-off of the second half | "the second half is graded from frame 110,808 but does not kick off until **113,742**. Those 2,934 frames are graded and all of them are `D`. The first half, by contrast, is graded from the kickoff itself." (`mtah/FAQ.md:38-40`) | No rule. Draft `2026-11-15a`, rows 110,808–113,741: `H` 1,455, `A` 44, `D` 1,435; six possessions (five `H`, one `A`) that cannot exist, the first one 692 frames long from the first graded frame. | No bench contains this case: Metrica truth is `X` before the first event (`scripts/bench.py:89`, `:103-104`) and IDSSE rows start at the half (`scripts/idsse_cache.py:11`). | DIFFERENT (verified) | low on spellF1, certain; see D1 |
| C3 | Stoppage after a goal | Not stated. Smart Tagging has `GOALS` (4) and `Kick Off (After Goal)` (4) (`mtah/DATA.md:150`, `:156`). | No rule; Smart Tagging unused. | Provider flags. | UNKNOWN | low (4 events) |

### D. Frames, halves, alignment

| # | Term | Organiser text (verbatim) | Our pipeline | External truth / bench | Verdict | Impact |
|---|---|---|---|---|---|---|
| D1 | Graded frames | "The two halves: video frames **11,185–90,651** and **110,808–190,197**." (`mtah/FAQ.md:35`) | `HALVES = [(1, 11185, 90651), (2, 110808, 190197)]` (`src/possess/pipeline.py:40`). Halves decoded separately, no state carried across (`src/possess/decode.py:478-486`). | Scorer never matches across the half-time hole (`mtah/evaluation/possession_metrics.py:259-271`). | MATCH on the boundaries; C2 is the exception inside them | — |
| D2 | Row count and index | "exactly **199,319** data rows" / "`frame` is `0 … 199318`, in order" (`mtah/SUBMISSION.md:21-22`) | `N_FRAMES = 199319` (`src/possess/pipeline.py:39`); writer (`src/possess/epts.py:157-162`). Draft file: header + 199,319 rows. | — | MATCH | — |
| D3 | ATD frame index | "`video_frame = frame_id - 1`" (`mtah/DATA.md:245`) | `row = frame_id - 1` (`src/possess/pipeline.py:86`, `:136`); period mask shifted the same way (`src/possess/pipeline.py:47`). | — | MATCH | — |
| D4 | Frame rate | "25 frames per second" (`mtah/CHALLENGE.md:9`) | `FPS = 25.0` (`src/possess/epts.py:17`). | External data 25 Hz (`scripts/idsse_cache.py:13`). | MATCH | — |
| D5 | Pitch size | "The metadata reports 105 × 68 m; `games.py` carries 105.3 × 68.0 … The 30 cm matters only if you are computing distances in metres." (`mtah/DATA.md:94-95`) | 105 × 68 (`src/possess/epts.py:16`). We do compute metres; the error is 0.3% along the length. | 105 × 68. | PARTIAL | low |
| D6 | Smart Tagging time base | "`video_frame = round(seconds * 25)`" (`mtah/DATA.md:246`) | Not used by the pipeline. | — | n/a | — |
| D7 | Broadcast recording | "It is **not frame-aligned to the tactical video**" (`mtah/DATA.md:228`) | Not used (`docs/TECHNICAL.md:33`). | — | n/a | — |

### E. Ground truth and metric

| # | Term | Organiser text (verbatim) | Our pipeline | External truth / bench | Verdict | Impact |
|---|---|---|---|---|---|---|
| E1 | Scorer | "**This is the file that produces the official result.**" (`mtah/EVALUATION.md:25`). "If this document and that file ever disagree, the file is right." (`mtah/EVALUATION.md:153`) | — | `vendor/mtah/possession_metrics.py` is byte-identical to the official file (`cmp`); the bench imports it (`scripts/bench.py:32-33`). | MATCH | — |
| E2 | spellF1 | "Predicted possessions are matched one-to-one against true ones by **temporal IoU**, and the F1 of that matching is averaged over thresholds 0.3, 0.4, 0.5, 0.6 and 0.7" (`mtah/EVALUATION.md:53-54`). Code: same-team pairs only, greedy by descending IoU (`mtah/evaluation/possession_metrics.py:472-508`). | Tuned on this number. | Same. | MATCH | — |
| E3 | macroF1 | "The mean of the per-class F1 scores for `H`, `A` and `D`, over every graded frame." (`mtah/EVALUATION.md:35`); first tie-break (`mtah/CHALLENGE.md:144`). | Reported, not optimised. | Same. | MATCH | — |
| E4 | What is free and what is charged | "perfect but 1 frame late \| **1.0000**" (`mtah/EVALUATION.md:67`). "going quiet through an occlusion **inside** a possession is free on the ranking metric … At a transition it is the opposite — guess" (`mtah/EVALUATION.md:91-94`). | We always write a label. Switches inside a hidden block are placed in the middle of it (`gap_pos = 0.5`, `src/possess/pipeline.py:34`; `src/possess/decode.py:281-305`). | Same. | MATCH | — |
| E5 | Noise band | "**Differences below ~0.04 are noise** on the full match" (`mtah/EVALUATION.md:72`) | Changes are adopted on paired bootstrap and several seeds (`.planning/DECISIONS.md:143`). | — | MATCH (awareness) | — |
| E6 | How the truth is made | "**The ground truth is derived from corrected tracking data** — not from a human watching video, and not from event data. The derivation is mechanical … Nothing is smoothed, interpolated or hand-adjusted afterwards." (`mtah/CHALLENGE.md:97-100`) | — | Metrica truth is built from event data (`scripts/bench.py:7-10`): a different kind of source. IDSSE truth is a provider's per-frame flag (`scripts/idsse_cache.py:15-16`): the same kind. Acknowledged in `docs/TECHNICAL.md:95`. | PARTIAL | med (feeds B2) |
| E7 | Public window | "Use the public window to check that your pipeline is *correct* … never to choose between two versions of it, and never as training data." (`mtah/EVALUATION.md:139-141`) | Reported only (`docs/RULES.md:35`). `scripts/make_agent_slices.py:35-40` reads the public truth into agent slices for understanding the labels (ADR-001). | — | MATCH | — |

### F. Rules and deliverables

| # | Term | Organiser text (verbatim) | Our system | Verdict | Impact |
|---|---|---|---|---|---|
| F1 | No fitting on the game | "Training, fitting or calibrating your algorithm on the ground-truth possession data or on the provided video" is not allowed; "you MUST tune them on external data" (email, `docs/RULES.md:8`, `:16`) | All parameters from external data. Two features adapt to the provided input at run time (`smooth_touch`, `sd_prior`; ADR-017, ADR-018, ADR-020), disclosed in `docs/TECHNICAL.md:81`, `:85`. The question to the organisers is drafted but not sent (`.planning/organiser-question.md`). | UNKNOWN until answered | compliance |
| F2 | Frame-specific constants | "turning what you saw into labels, corrections, or rules and constants tied to specific frames of the game" is not allowed (`mtah/RULES.md:69-71`) | None in `src/possess/` apart from the published half boundaries. Relevant to how C2 is fixed (section 3, step 1). | MATCH | compliance |
| F3 | Hand labels, human data | "**No hand labels on the provided game, in any amount and for any purpose**" (`mtah/RULES.md:56`); no "human-produced data about the provided game" (`mtah/RULES.md:82`) | None used. | MATCH | — |
| F4 | Reproducibility | "Running your repository against the provided data must reproduce your submission. Not approximately — the same file." (`mtah/RULES.md:88-89`) | `uv run possess --data … --out …` (`src/possess/pipeline.py:146-155`); team-shape models are stored in the source (`src/possess/shape.py:484-512`), so no external data is needed at run time. Not re-run in this review. | MATCH (by inspection) | — |
| F5 | Data not redistributed | "**Do not redistribute the data.** That includes committing it to a public repository." (`mtah/RULES.md:107-108`) | `.gitignore` excludes `data/`, `submissions/`, `third_party/mtah/`; no file under `data/` is tracked. | MATCH | — |
| F6 | Hosted LLMs | "participants shall not upload, disclose or otherwise make available to any third-party hosted or cloud-based LLM any videos, datasets, confidential information or other restricted materials provided in connection with the Competition, or any extracts thereof." (`mtah/FAQ.md:109-112`; `mtah/OVERVIEW.md:154-156`) | Agents read the raw data on the strength of the email "Yes you can use agents to read raw data" (`docs/RULES.md:5`; ADR-001), which says it takes precedence. The repository clause quotes the Terms and Conditions; the repository holds only a transcribed fragment of the email. | UNKNOWN | compliance; the user's call |
| F7 | Technical document | Must cover "**the possession rules you implemented** — where you drew the line on loose balls, challenges, restarts" and "**training data** — every labelled dataset you used, its source, who labelled it and how much" (`mtah/SUBMISSION.md:68-71`) | `docs/TECHNICAL.md:44-48` has four bullets in our own terms; the data table (`:36-41`) gives source and licence, not who labelled or how much. Line 42 says "3 looks so far", line 89 says 7. | PARTIAL | none on score |
| F8 | File names, form | "Name the CSV and the technical document after your team — <team_name>.csv and <team_name>.pdf" (`mtah/SUBMISSION.md:94`); "One response per team." (`mtah/SUBMISSION.md:97`) | The user fills the form and names the files. | user's | — |
| F9 | Team size | "Each team must be composed of 3 to 5 members" (`mtah/RULES.md:95`) | Not recorded in the repository. | UNKNOWN | eligibility |

## 3. Plan, in order

Two days remain. Steps 1 and 2 change labels; steps 3 to 5 are records and documents; step 6 is not for this deadline.

### Step 1 — `D` before the second-half kick-off (C2). Half a day. Do first.

- **Why first:** the only difference where the organisers state the labels outright, and our draft is measurably wrong on it. It is independent of the control work.
- **Change:** in each graded half, frames before that half's kick-off are `D`. Locate the kick-off from the Smart Tagging `SET PIECES` / `Kick Off (Start)` tag of that half: rows from the first graded frame of the half up to the tag's start frame become `D`; the decoder keeps deciding from there. The tag is automatic input the organisers checked against both kick-offs (`mtah/DATA.md:164-166`; `mtah/starter/load_smart_tagging.py:16-19`). In the first half the tagged window contains the first graded frame, so nothing changes there.
- **Port:** the reader from `mtah/starter/load_smart_tagging.py` (standard library only) into `src/possess/`, with the citation. MatchLens has no Smart Tagging reader; its `detectors/set_piece_detector.py` is the closest file and should be named in the envelope's `matchlens_start`.
- **Not recommended:** writing 113,742 into the code. It is published (`mtah/FAQ.md:39`; `mtah/starter/load_smart_tagging.py:40`), but it is a constant tied to one frame of this game (F2). Use it only if the organisers say so.
- **Validation:** no external match has Smart Tagging or a graded pre-kick-off stretch, so `orig`, `real1`, `real2` must be byte-identical (assert it). The ADR-016 gate "at least one gains" cannot be met by construction. Record this in a new ADR as a definition alignment justified by `mtah/FAQ.md:38-40`, not by a score, so no gate is lowered silently. Unit test: a synthetic half with a kick-off tag.
- **Expected effect (estimate, not measured):** six false possessions removed out of 412; with F1 = 2·TP / (predicted + true), about +0.004 spellF1. 1,499 graded frames corrected, about +0.01 accuracy for the macroF1 tie-break.
- **Then check C3** on input only: what the draft writes between each `GOALS` tag and the following `Kick Off (After Goal)` tag. Add the same rule there only if the draft shows possessions in those stretches and the tag pairs are clean (4 events).

### Step 2 — controlled possession (B2–B6). Already running; do not redesign.

Envelope `L4-control-definition-fable`. Three inputs from this review:

1. **Gates (rank 4).** ADR-016 says "orig drop <= 0.005" on the old truth. The old truth rewards a possession for every touch, so a rule that follows the organisers' text will lose on it by design. Before merging, write an ADR that names the `ctl` truth as the deciding truth for this change (real2 + real1 co-decide on `ctl`, at least 3 real2 seeds, paired bootstrap), with old-truth numbers reported beside it. Otherwise the change either fails a gate that no longer measures the task or passes by a gate lowered without record.
2. **Restart slivers (B8).** While building `ctl`, count runs of the form `D`, old team for under 1 s, other team. The organisers' text leaves no room for them ("there is a clear possessor"). If they exist in the external truth, relabel them to the restarting team in `ctl` and report the count.
3. **Aerial duels (B6).** Metrica truth already keeps the previous team on `CHALLENGE` events; check that the `ctl` relabelling does the same on IDSSE, where no event type is available.

The threshold for "controlled" must come from the text and external data. The public-window numbers in section 2 show direction only.

### Step 3 — send the question to the organisers. Today.

`.planning/organiser-question.md` has been waiting since 2026-11-14. Add two lines: (a) may the published second-half kick-off frame or the Smart Tagging kick-off tag be used to write `D` before the kick-off; (b) confirmation that the agents email overrides the hosted-LLM clause (F6). Fallbacks if no answer arrives: keep step 1 on the tag (input, not a constant); `smooth_touch=0` / `sd_prior=0` are the existing switches for a refusal.

### Step 4 — technical document (F7). After steps 1 and 2, half a day.

Rewrite section 6 under the organisers' own headings: ball in play, starting point, controlled possession, touch without control, deflection, aerial duel, contested or loose ball. One line each on what we do and where we knowingly differ. Add "who labelled it and how much" to the data table. Add Smart Tagging to the assets table if step 1 lands. Fix the "3 looks" / "7 times" contradiction.

### Step 5 — final checks before upload.

Regenerate from a clean clone with the documented command and compare the file byte for byte (F4). Run `vendor/mtah/validate_submission.py` on the exact file. Confirm no file under `data/` or `third_party/mtah/` is tracked (F5).

### Step 6 — backlog, not before the deadline.

- Set-piece `Team` tags as restart evidence (B8). Throw-ins are tagged `N/A` (`mtah/DATA.md:177-178`), so coverage is partial, and no external match can validate it.
- Stoppage start and end timing (C1): nothing in the organisers' text to align to; stays an external-data question.
- Pitch length 105.3 m (D5): a 0.3% change in distances; would need a re-check of every radius for no expected gain.

## 4. What stays unknown from text alone

- What the corrected tracking "means by \"dead\" and by \"owning\"" (`mtah/CHALLENGE.md:102-103`). The organisers say the public window is where to see it; the text does not define it.
- How long "a few seconds" is (B3).
- The exact frame at which a stoppage starts and ends (C1).
- Whether stoppages after goals are handled correctly by the current pipeline (C3): needs an input-only check.

## 5. Needs the original emails or the terms

- Full text of both "Clarification of the rules" emails. The repository has a transcript of the second one's rule list and one quoted sentence.
- The submission-instructions email due 2026-11-14 and the form URL: any instruction there is not in the repository.
- Terms and Conditions: the hosted-LLM clause (F6) and data-use terms.
- Any organiser reply about possession definitions sent by email rather than through the repository.
