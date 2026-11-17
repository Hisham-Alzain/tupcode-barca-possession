# Short possessions: what the research literature offers (L32, 2026-11-18)

Research only. No code was changed, no pipeline was run, no match data was loaded. Three Opus
readers searched arXiv (Semantic Scholar answered HTTP 429 and was dropped) and read 13 papers end to
end as extracted text; the lead spot-checked 20 numbers and constants against the text files
(ELASTIC ramps and weights, ASRF table, PathCRF table, MonoTrack table, CAD ablation, Bekkers
constants, Link and Hoernig look-ahead, Bauer and Anzer feature windows). Everything else is as the
readers reported it. Every "expected benefit" below is an expectation; nothing here is measured on
our data.

**The loss being addressed.** Whole possessions under 8 s given to the wrong team or merged into a
neighbour (about +0.17 spellF1 if the team were right on live frames). Typical case: a 50-50 duel,
then the ball travels 2-5 s with nobody near it and no detected touch. Switches supported by
neither a touch nor the judge are within 0.5 s only 25% of the time.

**Bottom line.** No paper detects who wins a duel from tracking data; Vidal-Codina lists duels and
tackles as unresolved, and the searches for duel / tackle / loose-ball outcome returned nothing.
No paper infers a possession change and its winner from team reaction with hand rules; the
ball-free results are all trained networks. What the literature does offer is three structural
ideas we have not tried: (1) a wider, score-based set of touch candidates (distance minima united
with acceleration peaks, plus "kick distance"), (2) using low-precision candidates as segment
boundaries and voting the team inside each segment, instead of using them as evidence, and
(3) explicit "ball in flight from team X" states with forbidden transitions in the decoder.
Nothing can be adopted before the deadline (section 3).

## 1. Papers read in full

| # | paper | id / link | topic |
|---|---|---|---|
| P1 | Kim et al., ELASTIC: Trajectory-Based Synchronization of Event and Tracking Data in Soccer (CIKM 2026) | arXiv 2608.30227 | e |
| P2 | Kim et al., PathCRF: Ball-Free Soccer Event Detection via Possession Path Inference (KDD 2026) | arXiv 2602.12080 | a, b, f |
| P3 | Ishikawa et al., Alleviating Over-segmentation Errors by Detecting Action Boundaries (ASRF, 2021) | arXiv 2007.06866 | d |
| P4 | Liu et al., MonoTrack (badminton hit detection, 2022) | arXiv 2204.01899 | c |
| P5 | Ee et al., Constraint-Aware Decoding (CAD, 2026) | arXiv 2605.10149 | d |
| P6 | Link and Hoernig, Individual ball possession in soccer (PLoS ONE 2017) | doi 10.1371/journal.pone.0179953 | b |
| P7 | Bekkers, Pressing Intensity (2025) | arXiv 2501.04712 | a, f |
| P8 | Vidal-Codina et al., Automatic event detection in football using tracking data (2022), re-read | arXiv 2202.00804 | b, c |
| P9 | Adams and MacKay, Bayesian Online Changepoint Detection (2007) | arXiv 0710.3742 | c |
| P10 | Bauer and Anzer, Data-driven detection of counterpressing in professional football (DMKD 2021) | Springer open access | f |
| P11 | Biermann et al., Synchronization of passes in event and spatiotemporal soccer data (Sci Rep 2023) | Sci Rep 13:15878 | e |
| P12 | Kim et al., Ball Trajectory Inference from Multi-Agent Sports Contexts (KDD 2023), re-read | arXiv 2306.08206 | b, f |
| P13 | Farha and Gall, MS-TCN (2019) | arXiv 1903.01945 | d |

**P1 ELASTIC.** Defines the frame of a kick or reception from tracking alone, then aligns event
data to it. Candidate touch frames (§2.2) are the union of (a) local minima of player-ball
distance, (b) local minima of ball-to-touchline distance, (c) local maxima of ball acceleration
paired with the nearest player; a candidate is dropped if the player is more than 3 m from the
ball. Each candidate is scored with clipped linear ramps (§2.3): acceleration over 0-30 m/s²;
player-ball distance over 0-3 m; "kick distance" = the largest player-ball distance before
(reception) or after (kick) the candidate, over 0-3 m; slope of player-ball distance over 0.2 s,
up to 7 m/s; nearest opponent distance over 0-3 m for tackles. All weights 0.25; matches under 0.5
rejected; global Needleman-Wunsch alignment. Sportec, 3 matches: 95.2% of events within 2 frames;
tackles / dispossessions 87.9%. Candidate coverage: all three sources 98.4%, acceleration only
88.5%, distance only 77.2%. Global alignment against greedy: 96.5% against 84.1%. Needs IDs and
events for the alignment; the candidate set and scores do not. Ball height is only a gate and
removing it loses nothing.

**P2 PathCRF.** The state at each step is one edge among players: a self-loop is control, u→v is
the ball in flight from u to v. Only four transitions are legal (stay; control → kick; flight
u→v → control or kick by v; out of play); the rest are masked at −1e4; Viterbi decoding. Seven DFL
matches, no ball used: event F1 75.7%, frame-level team possession 92.4%. Adding the legal-path
Viterbi to an untrained-for-it classifier lifts event precision 58.8% → 69.5% (F1 65.3 → 73.1).
A CRF with fixed transition scores over-segments (2,135 events against 1,831 true); context
dependent transition scores fix that. Their training labels add missed touches as
Ramer-Douglas-Peucker corners of the ball path (tolerance not given). Needs a trained network and
per-window player tracks.

**P3 ASRF.** Boundaries are local maxima of a boundary probability above 0.5; every segment between
two boundaries takes the majority class of its frames. F1@10/25/50 on 50 Salads rises from
76.1/74.5/66.7 to 84.9/83.5/77.3 although boundary precision is only 37%: a spurious boundary
between two segments that vote the same class disappears. The alternatives were worse: relabelling
segments shorter than a threshold 81.5-82.4 (and it deletes true short segments), Gaussian smoothing
80.7. The threshold barely matters between 0.1 and 0.9. The refinement step itself needs no
training.

**P4 MonoTrack.** A thresholded-derivative hit detector reaches F1 0.699; a small learned scorer
0.866; the scorer plus a dynamic program that picks hit frames to maximise Σ(score − τ) with hits
at least 0.5 s apart and alternating sides reaches 0.946 (recall 78 → 94%). τ, the cost per event,
is the mean frame score. Video and pose inputs; the selection step is input-agnostic.

**P5 CAD.** Viterbi with a duration counter per state: a transition is legal only after the
segment's minimum duration, a self-loop only under its maximum; bounds counted from labels, no
retraining. Breakfast: F1@10 74.1 → 78.8; without the duration bounds 75.0; soft bounds 75.6
against hard 78.8.

**P6 Link and Hoernig.** A ball action starts when a player is nearest and within a distance
threshold, optionally only at a local maximum of ball acceleration of at least 4 m/s²; it ends
when the ball extrapolated 1 s ahead at its current velocity is beyond a second threshold from the
player. F (0.6 s tolerance, right player): proximity 0.77, plus acceleration 0.83, plus the 1 s
look-ahead end 0.88. Threshold values are not printed. Control / no-control classification is 50%
right on no-control intervals. Team possession is defined as unbroken during play, with "ping-pong"
phases a named subtype. Needs IDs.

**P7 Pressing Intensity.** Time for each player to reach a target extrapolated 1 s ahead:
reaction time + distance / top speed + a penalty for facing away; turned into a probability by a
logistic (σ 0.45, T 1.5 s); team value 1 − Π(1 − p); players under 2 m/s ignored. Anonymous, team
level. No accuracy evaluation. Reaction time and top speed values are not in the text.

**P8 Vidal-Codina (re-read for the touch rule).** A gain needs, over one player's run of frames
inside the possession zone, a change of ball direction or of speed; otherwise the ball "passed by".
The tolerances are in a supplement we have not read. Already covered by T74 (rejected).

**P9 BOCPD.** A recursion over the time since the last change point gives a change probability per
step from a predictive model per segment and a hazard 1/λ. No accuracy numbers.

**P10 Bauer and Anzer.** Takes the turnover as given (operator possession flag) and classifies
counterpressing from 134 features at 0, 1, 2 s after it (nearest-player distance and speed per team,
players within 10/20/30 m, stretch of the 3-5 players nearest the ball, team speed): F1 0.67-0.69,
AUC 0.874. Windows over 2 s were unusable because possession changes again too often. It does not
detect that a change happened.

**P11 Biermann.** 1 s windows of player-ball distance and ball acceleration, naive Bayes; passes
within 0.48 s rise from 14% to 69-71%; exact frame 5-6%. Even with events and IDs, 0.5 s timing
tops out near 70%, the level of our "touch and judge agree" case.

**P12 Ball Radar (re-read for partial ball).** Team possession 85.85% without the ball; with 5 / 10
/ 20% of ball frames given, 91.5 / 95.0 / 97.4% per frame (Table 3). Flight frames are labelled
with the next controller. Trained network.

**P13 MS-TCN.** Truncated smoothing loss; a training device, not a post-process. Nothing to use.

Not read (not found or not fetched): ETSY (known only through ELASTIC's benchmark of it, 54.4%
within 2 frames), sync.soccer, Anzer and Bauer pass sync, Khaustov and Mozgovoy, Richly, TrackNet
bounce detection, BCN, Kalman / IMM innovation tests, any expected-F1 decoding paper. Abstract
only: arXiv 2301.10052, 2410.17785, 2512.10355, 2606.09452, 2411.01122, 2405.15995.

## 2. Ranked methods for our problem

Inputs assumed: ball x, y, speed at 25 fps when visible (smoothed); anonymous positions per team;
no height; no IDs. Common gate for every item: leave-one-match-out on the three external matches;
real2 / real1 / orig benches with ctl T=2 and a bootstrap CI; spellF1 split by spell length (under
4 s, 4-8 s, over 8 s); public 5 min reported only. Items 1-3 are one line of work.

**1. Team-level touch candidates from the union of distance minima and acceleration peaks, scored
with kick distance (P1 §2.2-2.3).**
- Mechanism: per team, d_T(t) = distance from the ball to the nearest player of team T. Candidates
  = local minima of d_H and of d_A, united with local maxima of ball acceleration, kept when
  d ≤ 3 m. Score per candidate and team = mean of four ramps: acceleration 0-30 m/s²; 1 − d/3 m;
  kick distance (largest d_T in the next visible stretch, 0-3 m: did the ball actually leave this
  team?); slope of d_T over 0.2 s, up to 7 m/s. In a duel both teams produce a candidate on the
  same frame; the outgoing side is separated by the post-touch slope and by our existing flight
  direction term. Optional third source: RDP corners of the visible ball path (P2).
- Differs from: speed-jump kick detection (precision 0.15-0.50), which used one cue everywhere.
  Here the distance minimum does not depend on the smoothed acceleration at all (the paper loses
  10 points of coverage with acceleration alone), and precision is not required by items 2 and 3.
- Expected benefit: none by itself; it is the input of 2 and 3. Measurable alone.
- Data needs: none; the paper's constants as they are.
- Validation: recall of true switch frames within 0.5 s by any candidate, per match, for the
  current impulse touches against the union (target: clearly above the current detector on switches
  with "no touch"). No decoder run needed. If recall does not rise, stop the whole line.

**2. Candidates as boundaries, team by vote inside each segment (P3 §3.4).**
- Mechanism: cut every live stretch at the candidates of item 1; inside each piece sum the
  decoder's existing per-frame team evidence (H minus A) and give the whole piece to the winner;
  merge equal neighbours; then the existing ctl passes. A piece of 2-5 s gets its own decision and
  its edges sit on contact frames.
- Differs from: touch evidence as an emission, switch dating rules and snapping (T139, T140), which
  only moved existing switches, and "insert short spells from a cheaper-switch decode", which
  created spells from a second decode with no boundary support. Here spells can appear or vanish,
  but only between two contact candidates, and false boundaries cost nothing when both sides vote
  alike (37% boundary precision still gained 8-10 F1 points in the paper).
- Expected benefit: the timing of unsupported switches and some merged short spells. Size unknown;
  the paper's gain is on trained video features. Main risk: pieces with weak evidence flip and
  invent spells; a margin rule (keep the Viterbi label unless the vote margin exceeds m) needs one
  constant chosen on external data.
- Data needs: none for the rule; external matches for m.
- Validation: common gate, plus counts of invented and recovered spells under 8 s per match.

**3. Flight states with forbidden transitions in the decoder (P2 §2.3).**
- Mechanism: split H and A into control and flight: H-control, H-flight, A-control, A-flight (plus
  DH, DA). Legal: control → own flight (a kick, at a candidate of item 1); flight → control of
  either team (reception or interception, at a candidate); control H ↔ control A only on duel
  frames (both teams within 1 m). Illegal: flight → the other team's flight, and any change of
  team while the ball is away from everyone. Flight emission: nobody within about 1-3 m and the
  ball moving; flight frames are labelled with the kicking team. Team changes can then only be
  dated at contact frames, and a one-kick spell is "duel → H-flight → A-control", not a free choice
  on evidence-free frames.
- Differs from: measured transition costs and posterior placement (same topology, other numbers);
  this changes the topology. Hidden-ball blocks need a rule (allow the old 4-state behaviour inside
  them), since 41% of frames have no ball.
- Expected benefit: the largest of the list if item 1's recall holds, because it attacks both
  wrong-team flights and switch timing; also the largest build (decoder refactor, every weight
  re-checked).
- Data needs: none new; flight radius and speed chosen on external data.
- Validation: common gate; before it, on external truth, the share of true switches that fall
  within 0.5 s of a legal transition point (if under about 80% the masks would forbid the truth).

**4. Event selection with a cost per event and a minimum gap (P4 §4.4).**
- Mechanism: choose the switch frames among candidates by a dynamic program maximising
  Σ(score − τ) with a minimum spacing, τ = mean candidate score. spellF1 charges a flat cost per
  invented spell; τ is that cost made explicit. In practice this is the selection rule inside
  item 2 or 3, not a separate build.
- Differs from: the 0.6 s / 2 s deletion passes, which act after the path is fixed.
- Expected benefit: small alone. Data: none. Validation: as item 2.

**5. Minimum duration inside the decoder instead of deletion after it (P5 §3.2).**
- Mechanism: a duration counter so that an H or A spell cannot be left before 0.6 s (or 2 s without
  a clean touch); the path then chooses the best alternative, where the post-pass just hands the
  frames back to the previous team.
- Differs from: ctl post-passes; the D-only dwell model of the HMM book review.
- Expected benefit: small; risk that it hides true spells under the bound (the paper's hard bounds
  come from labels with no such spells). Our own notes (T58, Richard 2018) warn of this.
- Validation: common gate, short-spell recall watched separately.

**6. One-second look-ahead for the end of control (P6).**
- Mechanism: a team's control ends at the first frame where the ball, extrapolated 1 s at its
  current velocity, is beyond the control radius of that team's nearest player; optional start
  condition: acceleration peak ≥ 4 m/s². Gives the kick frame of a one-kick spell without an
  impulse threshold.
- Differs from: the 5-frame retention rule and speed-jump kicks. Paper gain: F 0.83 → 0.88.
- Expected benefit: small, on switch timing. Data: radius chosen on external data.
- Validation: timing of true spell ends within 0.5 s, before and after; then the common gate.

**7. Team time-to-reach the ball's future position as a judge feature (P7).**
- Mechanism: for each team, the smallest time for any player to reach the ball position 1 s ahead
  (reaction time + distance / top speed + facing penalty), home minus away, added to the judge.
  By name it is not among the 21 features in `ballteam.FEATS` (name check only).
- Differs from: plain proximity. Caution: after a kick it favours the receiving team, which is the
  rejected receiver-based attribution; it should only be used before or at a contact (who gets to a
  loose ball first), not during a flight.
- Expected benefit: small. Data: refit of the judge, leave-one-match-out.
- Validation: judge AUC on frames within 1 s before a true switch; then the common gate.

**8. Reaction check 0-2 s after a proposed switch (P10).**
- Mechanism: for each switch the decoder proposes without touch or judge support, compute both
  teams' speed toward their own goal, stretch of the 3-5 players nearest the ball and nearest-player
  closing speed at 0, 1, 2 s; a small external-fitted model says whether the switch is real.
- Differs from: the shape model (per frame, no alignment to a proposed switch).
- Expected benefit: low. The paper shows these features describe a known change, not that they
  detect one, and windows over 2 s fail. Lowest rank for that reason.
- Validation: AUC for "true switch within 0.5 s" on external proposals; build only above 0.75.

Not ranked: change-point probability on ball velocity (P9) is one more candidate source for item 1
at best, and our smoothing spreads a kick over the window; a trained set network with partial ball
(P12, 95-97% per frame with 10-20% of the ball) is the long-term route but cannot be trained and
gated on three matches.

## 3. What to build first, and tonight

**First: items 1 and 2 together.** Item 1 has a pass/fail check that needs no decoder run (does the
union of distance minima and acceleration peaks cover the true switch frames that our impulse
touches miss?). If it passes, item 2 is a post-process on top of the existing path and evidence,
about a day of work with a clear failure signal (invented spells). **Second: item 3**, only if
item 1's coverage is high, because its masks are only as good as the candidate set.

**Tonight: nothing.** None of these is a trivial, already-validated change. Every one needs a new
detector or a decoder change, a leave-one-match-out run and a cold QA. The submission stays as it
is.
