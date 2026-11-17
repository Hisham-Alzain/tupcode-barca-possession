# Team possession from uncorrected tracking — short technical document

More Than A Hack 2027, Phase 1 · Barcelona vs Athletic
Team: TupCode · Code: https://github.com/Hisham-Alzain/tupcode-barca-possession
Submitted file: `submissions/draft_2026-11-17f.csv` (the output of the default pipeline).
This is the short version. Every number, decision record (ADR) and test run is in the full document, `docs/TECHNICAL.md`, and in `.planning/DECISIONS.md`.

**Compliance in one paragraph.** Nothing was trained, fitted or tuned on the provided game or on its truth. Every rule, weight and threshold was chosen on five other matches (Metrica and IDSSE, §3). The public 5-minute clip was used only to check that the pipeline and file format work, and, from ADR-027 on, to understand the *types* of error we make (for example "a visible dead ball labelled as a team"); its score is reported for every version, but it was never used to choose a version or a value. There is one exception to our own adoption gate, and we state it plainly: the set-piece rule (ADR-029) fails our strict gate (on `real2`, our main external bench, the median gain is about +0.002 against the +0.005 we required, and two of four seeds have intervals that include zero; `real1` and `orig` pass and no bench drops). The user adopted it anyway as a recorded exception, because `real2` hides the ball during every stoppage, so a rule that needs a visible ball at the restart can barely act there, while on the provided input the ball is visible in the first 3 s of 80% of goal-kick clips (measured on the input only, not on labels). Code was written with Claude Code AI agents; agents read the provided raw data only after the organisers' written clarification ("Yes you can use agents to read raw data", 11 Nov 2026). No labels were created by hand.

## 1. Approach

**What we built.** For every frame we decide which team has the ball (H, A), whether the ball is dead (D), or that the frame is not graded (X). We work from the tactical tracking alone, plus two narrow uses of Smart Tagging. The key finding that shaped everything: the ball is missing in long blocks (99.9% of missing-ball frames sit in gaps of 5 s or more), so the ball cannot simply be filled in. We tried 10 fill methods and none beat leaving the ball missing. Possession therefore has to be read from the players as much as from the ball.

**How it works, in football terms.**

1. **Read the tracking** and work out each team's attacking direction per half by vote. We never use player identities.
2. **Find the kick-offs** from the Smart Tagging "Kick Off (Start)" tags: the kick-off is the frame in that window where the ball is nearest the centre spot.
3. **Spot stoppages**: ball out of play, a stationary or placed ball, a ball far from everyone near a line, or the whole pitch slowing down (the speed of the fastest players drops). A small learned model also says when a visible ball looks dead.
4. **Collect team evidence** every frame: which team is nearer the ball (capped at the 1 m control zone), ball touches (a sudden change of the ball's speed or direction with a player close by), the direction the ball is flying (the team attacking that way gets credit), a small learned model of which team holds the visible ball, and team-shape models that recognise "this team has the ball" from the players' positions alone. The shape models matter most when the ball is hidden.
5. **Decide each half as a whole.** Instead of deciding frame by frame, a 4-state Viterbi decoder (Home, Away, Dead-after-Home, Dead-after-Away) finds the most plausible sequence for the whole half. Switching team, stopping and restarting all have a cost, so a single noisy frame cannot flip possession. Restarting toward the other team is cheaper, because the other team restarts after 78–90% of stoppages on external truth.
6. **Clean up** with the football rules of §4 (control, line crossings, set-piece restarts).
7. **Write** 199,319 rows, X outside the graded halves.

**Why this rather than something else.** Three findings from external data drove the design. A missing ball is not a dead ball: possession usually survives a gap, and the MatchLens rule "ball unseen 0.5 s → dead" over-split possessions, so we removed it. Who has the ball matters more than when play stops (oracle ablations on external matches), so team evidence comes first. And half of all possessions are short (588 of 1,198 last under 4 s on external truth), so we need touch detection for deflections, tackles and clearances. Whole-half decoding was the largest single step: on tuning data the untuned MatchLens tracker scored 0.316 (Metrica 1–2 only), the decoder 0.398, team shape 0.493 and touch detection 0.566.

## 2. Assets used and dropped

| Asset | Used? | How |
|---|---|---|
| Tactical tracking (ATD) | yes | main input |
| Smart Tagging | partly | "Kick Off (Start)" finds each half's kick-off (ADR-021). Goal kick, corner and throw-in clips are only a gate for finding the physical restart in the tracking (ADR-029); the tag's time and Team field are never used as labels. All other tag types are not read. |
| Homography | no | the tracking is already in pitch coordinates; a camera-pan signal from it was tested and dropped (L15, ADR-030) |
| Videos, broadcast tracking | no | not read by the pipeline; broadcast is not synchronised |
| Public 5-minute truth | check only | format check, reporting, error-type diagnosis (ADR-027); never used to choose or tune |

**Tried and dropped.** The external matches have no Smart Tagging, so every tag idea was tested with a tag simulator built from the external truth (error-free to realistic timing and team errors). Rejected: a team-phase prior from the clips (lost on every external bench, even with an error-free simulated tagger); clips as tie-breaker or turnover timing; a fuzzy rule system over the clips (at most +0.002 to +0.003 with near-perfect clips, clearly negative with realistic ones); a tagged set-piece taker (no external gain). Also dropped: ball gap filling (a hybrid fill halved the hidden ball's position error, 8.7–11.0 m against 15.5–16.6 m for linear, but gave no spellF1 gain); re-dating switches after duels; transition costs measured from external data (−0.001 to −0.010); and a broad still-ball rule (+0.028 on the bench but would not transfer: a still ball is 4.5% of visible frames on the real match vs 21–42% on the bench).

## 3. Training data: source, who labelled it, how much

None of it comes from the provided game. We labelled nothing by hand.

| Dataset | Use | Source, who labelled it |
|---|---|---|
| Metrica Sample Games 1, 2 | tuning | github.com/metrica-sports/sample-data; per-frame H/A/D derived by our code (`metrica_csv.py`) from the event data Metrica Sports publishes with its sample tracking |
| Metrica Sample Game 3 | held-out | same |
| IDSSE (DFL) J03WMX, J03WPY | tuning / held-out | per-frame BallStatus and BallPossession flags published with the dataset (Bassek et al. 2025, Sci. Data, doi:10.1038/s41597-025-04505-y) |

**How much.** Five external matches: three for tuning (Metrica 1, 2, IDSSE J03WMX) and two held out (Metrica 3, IDSSE J03WPY). On this truth, 588 of 1,198 true possessions last under 4 s. Tuning scores are leave-one-match-out; the small learned models (team shape, ball-team, ball-dead) were fitted only on the tuning matches and their constants are stored in the repository.

**Making external data look like ours.** We hide the ball in blocks drawn from the challenge's gap-length histogram (~40%) and never use player identities. Three benches: `orig` (clean tracks, random hiding), `real1` (moving dead ball, small tracking jumps) and `real2` (real1 plus a Gaussian-smoothed ball, hiding tied to stoppages, player loss in hidden blocks). Their constants come from input statistics of the provided match only, never its labels.

**Two external truths.** The published external labels count every deflection as a possession; the organisers do not. So we also built a "controlled" truth by a fixed rule (T = 2 s, read from "a few seconds" in the text, not fitted): a team spell shorter than 2 s that interrupts the other team's live spell goes back to that team. From ADR-022 on, changes were judged on the controlled truth.

## 4. Possession rules

- **H / A / D / X.** Only H, A and D inside a half; while the ball is live one team always has it. X on every frame outside the graded halves (11,185–90,651 and 110,808–190,197), never on a graded frame.
- **Kick-off.** Graded frames before a half's kick-off are D. Detected kick-offs 11,187 and 113,741 (published 113,742: 1 frame).
- **Gaining the ball.** A team gains the ball by being alone near it for several frames (1 m control zone) or by a touch. On this match's heavily smoothed ball track, a touch is credited around the reconstructed contact point within 2.0 m. A touch keeps the team in possession until the next touch, at most 4 s. A ball reappearing after a hidden block of 1 s or more with exactly one team within 1.0 m counts as that team's touch.
- **Loose balls and challenges.** Possession stays with the last team. Contested touches (both teams close) keep the last clean toucher's hold; a loose ball is kept by the switch cost and the touch hold, unless team-shape, ball-flight or ball-team evidence says otherwise. There is no ball height, so a header is an ordinary touch.
- **Change of possession needs control.** A team spell under 0.6 s that interrupts the other team's live spell is not a change (ADR-030; it was 1.0 s under ADR-022). A spell under 2 s is also not a change unless it holds at least one uncontested touch of its own team (ADR-023). A kept spell starts at its first touch; the first spell after a stoppage is always kept.
- **Dead ball.** A missing ball alone never makes a frame dead. D comes from stoppage evidence; a stoppage's start moves to the ball's line crossing when one is seen within 5 s; a visible ball more than 0.5 m beyond a line for 2 s or more is D until seen inside again (Law 9); a ball at rest for 3 s or more with no player within 1 m is D; a learned "visible ball is dead" model adds to the stoppage evidence.
- **Restarts.** The restart team is inferred, not read: the decoder prefers the other team after a stoppage, but player and ball evidence decide. Inside a tagged goal kick, corner or throw-in, frames from our last dead evidence (within 15 s) to the physical restart (ball at rest on the required spot then kicked, or reappearing there and moving) are D. Free kicks have no such rule (no fixed spot).

## 5. Limitations

- **Expect a lower full-match score than the public clip.** The submitted file scores spellF1 **0.6588** (macro F1 0.8886) on the public 5-minute clip, with 17 predicted against 17 true possessions. That is a check, not an estimate: the clip holds 17 possessions, and the organisers' own two baselines score 0.3256 and 0.4242 on the clip but 0.2433 and 0.2242 on the full match, in the opposite order (`third_party/mtah/EVALUATION.md`). The honest expectation for the full match is `real2`-like.
- **Definitions are approximated.** D comes from proxies, not the whistle: a whistle with the ball at a player's feet is D for us only once the players slow down. Control is approximated by 0.6 s, or 2 s without a clean touch, rather than "a few seconds"; passes are never counted. One clean but uncontrolled touch can still produce an opponent spell over 0.6 s. No rule for goal → kick-off or for free kicks.
- **The 0.6 s threshold is a hedge, not a gate pass.** Under four readings of the labels it changes the external score by +0.0050 (old truth) to −0.0027 (T = 3 s); it was chosen for the smallest worst-case loss (0.0052 against 0.0102 for 1.0 s).
- **The set-piece rule** (ADR-029) fails our strict gate and has known open defects we would fix with more time: the goal-area half-width in `setpiece.py` is 14.66 m where it should be 9.16 m; the throw-in spot has no outer bound; a later "ball at rest" restart wins over an earlier "ball reappears" one.
- **Date of a change.** We date a turnover at its first touch, like the external truth. If the organisers date it at control, our switches after duels sit early; no external data can check this, and the dating convention alone changes our external score by up to 0.09.
- **Hidden-ball dead share may be over-called.** On the provided match a per-match prior estimated without labels raises the hidden dead-ball share from 0.724 to about 0.897.
- **Generality.** The entry point runs this one match (frame count and graded halves are constants). Two switches read the ball track's smoothing (acceleration p99 against 150 and 70 m/s²). Without a Smart Tagging file, the kick-off and set-piece rules are lost.
- **Held-out data** was looked at 7 times, so held-out numbers are slightly optimistic.

## 6. Reproduction

The match data is not in the repository. Put the organisers' bundle under `data/challenge/` so that these exist:

- `data/challenge/tactical/BAR-ATH_tactical_atd.{xml,txt}` (tracking, required)
- `data/challenge/tactical/BAR-ATH_tactical_smart_tagging.xml` (kick-off and set-piece tags)

Then, with Python 3.12 and `uv`:

```bash
uv sync
uv run possess --data data/challenge --out submissions/<team_name>.csv
uv run python vendor/mtah/validate_submission.py submissions/<team_name>.csv
uv run pytest        # tests need no match data
```

The submitted file was produced by the same command with `--out submissions/draft_2026-11-17f.csv`.
