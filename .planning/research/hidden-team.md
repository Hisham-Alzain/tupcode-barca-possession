# Team in possession while the ball is not tracked (L5-hidden-team, 2026-11-15)

Scope: team-level, ID-free, offline labelling of H / A on hidden-ball frames. External tune matches
only (metrica1, metrica2, idsse1); nothing here was chosen on the provided match.

## 1. What the literature offers

Sources read: the repo notes in `docs/research/literature/` (B_segmentation: kim2023-ballradar,
capellera2024-transportmer, kim2026-pathcrf, bauer-anzer2021-counterpressing, bialkowski2014,
kim2022-soccercpd; A_control: wei-2015, lang-2022, vidal-codina-2022) and the full text of the local
Vidal-Codina PDF (searched for missing / occluded / smoothed ball). No web fetch was made; items
marked (memory) are from my own recall and are not checked against the papers.

| Line of work | What it does | What transfers to us |
|---|---|---|
| Ball / possessor inference from players only: Ballradar (Kim et al., KDD 2023), TranSPORTmer (Capellera et al., ACCV 2024), PathCRF (Kim et al., KDD 2026), Wei et al. 2015 | Deep set / sequence models over all players, trained on 18+ matches with ball and event labels. Repo notes: attacking team right on ~86% of frames (Ballradar), game state ~80% (TranSPORTmer); every one of them adds temporal decoding (CRF / Viterbi / post-processing) | A reference ceiling, not a method: our leave-one-match-out logistic names the right team on 0.87-0.93 of hidden live frames when it names one (section 2), i.e. already at the published frame level. The models are not trainable on 3 matches without IDs. PathCRF's context-dependent stay cost is the design idea behind candidate 2 |
| Vidal-Codina et al. 2022 (§2.1, §2.3) | Possession needs the ball inside a possession zone on a control frame. For two of the three providers "missing ball data" IS the dead-ball flag; one listed error class is "the ball goes missing after the assist" | Nothing for hidden frames: the paper never infers possession without the ball (stated after searching the full text) |
| Reaction to a turnover: counter-pressing (Bauer & Anzer 2021), recovery runs, team centroids moving together and following the ball (Frencken et al. 2011, memory), wider / longer shape in attack and compact in defence (Bialkowski 2014; Moura et al. 2012, memory) | Classify or describe what teams do AFTER a known turnover; supervised versions need 20,000 labelled transitions and the event feed | The reaction exists but follows the turnover (measured: 1-2 s) and is not specific to turnovers. Already in the shape model as linear columns (vu, vu_lead, vu_lag, dsy, dsu: T34). New here: the reversal read as an event (candidates 1 and 2) |
| Pitch / space control (Spearman 2018; Fernández & Bornn 2018, memory) | Who reaches a location first; possession value needs the ball position and its travel time | Without the ball it reduces to territory and enclosure, which the shape model has (cu, enc, box). Not built |
| Change-point detection (PELT, SoccerCPD) | Date a regime change in a 1-D series | Tested directly as candidate 1 (change point of the player flow inside a hidden block) |
| Ball imputation between sightings (T03-T10, Ballradar with 20% ball) | Fill short gaps | Our hidden blocks are 5-60 s; closed in T10 |

Takeaway: nothing published infers a team switch inside a multi-second blackout better than a
per-frame players-only classifier plus sticky decoding, which is what the pipeline already does.
The only unexploited signal class is the reaction dynamics as an *event*.

## 2. Diagnosis (scripts/a7_hidden.py diag; default decoder; ctl T=2 truth unless noted)

Canonical real2 build: the saved a4 seed-0 inputs = `evaluate.py` (old truth 0.5589, ctl 0.6333).
`scripts/diagnose.py` gives 0.5466 because its `default_run` hands no player positions to the
decoder (vertex touch ADR-020 silently off) and fits real2 shape features on the full positions:
its numbers describe an older decoder, not the default.

| | real1 | real2 s0 |
|---|---|---|
| hidden live frames (share of play) | 0.23 | 0.07-0.12 |
| right team when a team is named | 0.89-0.93 | 0.89-0.92 |
| hidden live frames predicted D | 0.05-0.13 | 0.19-0.35 |
| as is | 0.6352 | 0.6333 |
| oracle: true team on hidden live frames | 0.7232 (+0.088) | 0.6626 (+0.029) |
| ... only inside true possessions < 4 s | +0.061 | +0.020 |
| ... only inside true possessions >= 4 s | +0.031 | +0.009 |
| oracle: found switches moved to the true frame | +0.015 | +0.003 |
| all found hidden switches 0.5 s earlier | +0.0020 | -0.0003 |

- The loss is **possessions that are never named**, not timing. Of the true hidden turnovers whose
  shorter neighbour lasts 2-4 s, 15 of 62 (real1) have a predicted switch within 6 s; for 4-10 s it
  is 49 of 66, for >= 10 s 18 of 21. A 3 s hidden possession changes the shape log-odds by a bump
  that never pays two switch costs (2 x 4.0 against <= 0.2 per frame).
- Timing of the switches that are found: predicted - true median +0.5 / +0.6 / +0.6 s (IQR about
  -0.3 .. +1.3 s), 46-65% within 1 s. The shape log-odds cross zero 0.25 s after the true turnover
  on average, so the decoder's own change point is nearly centred.
- What the players do: the flow (mean velocity of all tracked players along the pitch) points to
  the old team's attack until +2 s and only then reverses (pooled, 112 turnovers with >= 3 s on
  both sides: it points to the NEW team's attack on 0.15 of them 1 s before, 0.49 at +2 s, 0.75 at
  +4 s; the shape log-odds: 0.50 at +0.5 s, 0.81 at +2 s). It reacts **later** than the shape model,
  which already reads the velocity 2 s ahead (vu_lead).
- As a detector the reversal is unspecific: at |reversal| >= 1 m/s it covers 0.46-0.80 of the hidden
  turnovers, but only 8-17% of its peaks on hidden frames lie at one (back passes, long balls,
  pressing and offside lines reverse the flow as well).
- On real2 (hiding coupled to the game state, the closest to the provided match) hidden live play
  is 7-12% of the frames and the whole team oracle is +0.029; a hidden-team mechanism cannot reach
  the +0.005 median seed gate there unless it recovers a sixth of everything that is recoverable.

## 3. Candidates (both opt-in, defaults unchanged; `src/possess/react.py`)

`react.flow`: mean x-velocity of all tracked players, signed towards home's attack; players matched
over 0.8 s by nearest neighbour inside their team (no IDs; a vanishing player has no velocity).

**Candidate 1, `gap_cp=1` (switch placement).** A predicted H<->A switch on a hidden frame moves to
the change point of the flow inside its block (+-4 s), minus a reaction lag `cp_lag`.
new_angle vs T20 / T77: T20 put the switch at the block edge, T77 moved visible switches towards
the preceding duel; this dates a hidden switch from the players' reaction.
Result: **rejected.** ctl T=2: real1 -0.0235 / -0.0136 / -0.0123 (lag 0 / 1 / 2 s; CI90 below 0),
orig -0.0122, real2 seeds 0-3 at lag 2 s: -0.0019 / -0.0004 / -0.0071 / -0.0083. Why: the flow change
point lands within 1 s of the true turnover for 21-29% of the found switches (after removing its
median lag 33-48%), the decoder's own placement for 46-65%.

**Candidate 2, `w_rev=2.0` (reaction as a transition term).** On hidden frames the switch to a team
costs `c_switch - w_rev * max(0, reversal - rev_thr)` where the flow reverses towards that team's
attack (`rev_lag` = 1.0 s later, `rev_thr` = 1.0 m/s, reversal over +-1 s); directional, so a reversal
never pays for the switch back. new_angle vs T34 / T35 / T62: T34 holds the velocity as linear
emission columns, T35 raised the emission weight, T62 used a cue on D transitions; this is a
directional H<->A switch discount, the team-level form of PathCRF's context-dependent stay cost.

| bench (ctl T=2 truth) | default | w_rev=2.0 | gain | paired bootstrap CI90 | old truth |
|---|---|---|---|---|---|
| real1 | 0.6352 | 0.6426 | +0.0074 | [+0.0015, +0.0141] | 0.5941 -> 0.6018 |
| orig | 0.6561 | 0.6638 | +0.0077 | [+0.0021, +0.0142] | 0.6113 -> 0.6196 |
| real2 seed 0 | 0.6333 | 0.6369 | +0.0036 | [+0.0006, +0.0075] | 0.5589 -> 0.5623 |
| real2 seed 1 | 0.6244 | 0.6258 | +0.0014 | [-0.0029, +0.0061] | +0.0013 |
| real2 seed 2 | 0.6415 | 0.6454 | +0.0039 | [+0.0003, +0.0084] | +0.0034 |
| real2 seed 3 | 0.6244 | 0.6246 | +0.0002 | [-0.0024, +0.0031] | +0.0016 |

Per match (ctl T=2, predicted / true possessions): real1 metrica1 0.686 -> 0.687 (293 -> 293 / 296),
metrica2 0.619 -> 0.631 (260 -> 267 / 276), idsse1 0.600 -> 0.609 (326 -> 332 / 303); real2 seed 0
0.637 -> 0.641, 0.621 -> 0.622, 0.641 -> 0.647 (250 / 263 / 274 -> 250 / 264 / 276).

Decision rule: no bench drops, real1 gains with CI90 above 0, orig gains; **the real2 seed gate
fails** (median of seeds 0-3 +0.0025, of seeds 1-3 +0.0014, needed +0.005). Not adoptable; left
opt-in. The gate is out of reach by construction on real2 (section 2). Grid: 15 configs
(w_rev 2 / 3 / 4, rev_lag 0 / 1 / 1.5 / 2, rev_thr 0.5 / 1 / 2); lag 1-2 s at w_rev 2 all gain
(+0.0063 .. +0.0089 real1), a higher weight or lower threshold adds possessions and loses on ctl
truth, so the real1 figure carries some selection.

Transfer (provided match INPUT only, `a7_transfer.json`): the flow exists on 0.83 of the hidden
frames, |reversal| >= 1 m/s on 4.7% of them; the default has 10 of its 270 team switches on hidden
frames; `w_rev=2.0` changes 0.03% of the frames (hidden switches 10 -> 14, possessions 378 -> 378,
107.1 per half hour both); `gap_cp` changes 0.12%. Expected effect on the provided match: none.

Reproduce: `uv run python scripts/ramlock.py -- uv run python scripts/a7_hidden.py run bench=real1`
(`bench=real2 seed=0..3`, `bench=orig`; `diag`, `transfer`), or
`... scripts/evaluate.py --variant x --bench real1 --no-public --dec w_rev=2.0` (old truth 0.6018).

## 4. Rows proposed for TRIED.md (not written: the file is not owned by this envelope)

| T88 | Hidden H<->A switch moved to the change point of the all-player flow (react.place_switches, gap_cp=1, lag 0-2 s) (L5) | decoder | ctl T=2: real1 -0.012..-0.024, orig -0.012, real2 seeds -0.000..-0.008 | REJECTED | the flow reverses ~2 s after the turnover with a wide spread; the decoder's own placement is better (46-65% vs 21-29% within 1 s). agent/a7 |
| T89 | Directional switch discount on hidden frames where the flow reverses towards the new team (w_rev=2.0, rev_lag 1 s, rev_thr 1 m/s) (L5) | decoder | ctl T=2: real1 +0.0074 CI90 [+0.0015,+0.0141]; orig +0.0077; real2 seeds +0.0036 / +0.0014 / +0.0039 / +0.0002; old truth real1 +0.0077 | NEUTRAL (not adopted) | passes the drop gates, fails the real2 seed median (+0.0025 < +0.005); ~0 on the provided match (0.03% of frames). Candidate to bundle with T79. agent/a7 |
| T90 | All found hidden switches 0.25-1.0 s earlier (offline post-pass, diag only) (L5) | decoder | ctl T=2: real1 +0.0006 / +0.0020 / +0.0008 / -0.0017; real2 +0.0002 / -0.0003 / -0.0015 / -0.0030 | REJECTED | timing-only oracle is +0.015 (real1) / +0.003 (real2); the loss is unnamed 2-4 s possessions. agent/a7 |
