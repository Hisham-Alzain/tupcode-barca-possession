# Papers read in full: what they give us for spellF1 (L16-papers, 2026-11-17)

Seven PDFs were read end to end by three Opus readers (methods, thresholds, tables, appendices) and
compared with `src/possess/`, `.planning/TRIED.md` (T01-T90) and `autonomous-L8.md` (T91-T141).
This replaces the abstract-level notes in `docs/smart-tagging-definitions.md` section 5.

How it was read: the Read tool could not render PDF pages on this machine (pdftoppm missing), so
the text was extracted locally (pdftotext / pypdf / PyMuPDF) and read as text. Nothing from the
PDFs was executed. Page numbers are PDF pages of the arXiv / journal file. The lead spot-checked
eleven of the quotes below against the extracted text; the rest are as reported by the readers.
Not read: Khaustov and Mozgovoy 2020 (not downloadable), Ganchev 2010 (not in the set), and the
online supplements of Vidal-Codina (S3-S4, threshold values) and Lang (hyperparameters).

**Bottom line.** Nothing in these papers addresses our largest loss (wrong team on live frames,
short possessions). The untried material is all on the stoppage side: goal-area geometry for the
visible dead ball, a nonlinear stoppage model, and a length model for D. None is likely enough to
pass an external gate tonight to be flagged without reservation; see section 2.

## 1. Corrections to the abstract-level notes

| paper | what the note said | what the paper says |
|---|---|---|
| Vidal-Codina 2022 | rules usable as they are; set pieces 82-100% | Every rule is per player and needs player IDs. Dead ball is an **input**, not detected: "the ball is dead, signaled by either a binary boolean or by the absence of ball tracking data" (p3, p8); the paper names the lack of that flag a main limitation (p16). The values of the tolerances (eps_s, eps_theta, eps_v, eps_k, eps_c ...) are not in the paper, only Rpz and Rdz (p12). Possession itself is not benchmarked (p12); shot recall 53%, free kick precision about 82%. The restart team is "the closest player to the ball" on the first live frame, and a trigger must hold from the first to the last dead frame (p9): the note's "team of the player at the corner mark takes it" is a simplification. |
| Morra 2020 | atomic events incl. ball out from distances and speeds; thresholds unconfirmed | Synthetic data (game engine, 30 fps, 22 IDs, perfect ball). **No rule is given for BallOut or Goal** (p7). Tuned threshold values are never reported, only search ranges ("0.1-2.0 meters with step 0.1", speeds "1-15", p9-10). "Higher than 80%" excludes BallDeflection (F about 0.31) and Tackle (recall about 0.61) (p10). The restart interval rule in the note is ours, not the paper's. |
| Sanford 2020 | self-attention over players and ball; events within 0.5 s | Median timing error about 0.2 s. Ball only is almost as good as ball + 5 nearest players (pass 0.78 vs 0.87); **all players is worse** (0.80). Everything depends on the ball track; nothing for hidden frames. |
| Lang 2022 | 92% of frames; "stoppages longer than 2 s found with 81% precision"; rule "no stoppage under 2 s" | 92.0% frame accuracy (AdaBoost) against 86.7% for logistic regression; **stoppage F1 79.8 against 61.2** (p4), counted on matches with IoU >= 0.5. 78% of starts and 80% of ends within +-2 s (text p5; abstract says 77 / 81). Stoppages under 2 s were **excluded from the evaluation** ("only 1.3%", p4): this is not a finding that they should not be produced. Input is 22 player slots in match-sheet order (needs IDs), with x, y, velocity, acceleration and shifted copies at "0.4, 0.8, 2, 4, 20, 40, and 80 s ... from both the past and the future" (p3); a 3.64 s median filter after it (p4). It fails exactly where we do: "a very short sequence of ball in-game lies between two stoppages, so the interruptions are stitched together", and false live "within long interruptions where the players show more activity" (p7). No remedy offered. |
| Kim 2023 | player-only possession right two times in three: a prior only | 64.7% is the **player**-level accuracy. **Team**-level is 85.85% (Table 2, p7), ball error 3.66 m, but on in-play 10 s windows only: "we only take in-play situations into account" (p6). No dead-ball handling. Team accuracy falls to 72.61% without the set encoders (Table 8). With 5% of ball frames given it reaches 91.54%. Our hidden-frame team call is about 0.90 (T105), so even the full network is not above us. |
| Richard 2018 | coarse labels as soft terms in the Viterbi path score | The useful part is an **explicit class length model** inside the Viterbi: "p(xt|cn(t)) · p(ln|cn) · p(cn|c1n-1)" with a Poisson length per class (p4, Eq 7-9) and a segment-level recursion capped at L = 2000 (p5). No frame subsampling. Only frame accuracy is reported, no segment metric. The paper itself warns of a "bias towards uniform lengths, which prevents short" actions from being found (p7). |
| Li 2021 | same | One labelled frame per segment is expanded to dense labels by a boundary search on the **trained network's features** (Eq 1, p4), plus a confidence loss and a truncated-MSE smoothing loss (tau 4, lambda 0.15, beta 0.075). Not a fixed rule. Does not apply: our external truth is already dense (`scripts/bench.py:build_truth`, DFL ball possession flag). Using the segment start as the timestamp is the bad case (F1@50 28.0 against 47.4, Table A p9). |

## 2. Usable ideas not in the system and not rejected, ranked

Honest flag: **no idea is marked "worth building tonight" without reservation.** If one thing is
built, it is idea 1; my estimate that it passes an external gate is about one in three, mainly
because the external benches barely contain the situation (T36, T116, T120: dead balls are frozen
or hidden there). Idea 2 is the only other one with a measured result behind it (in the paper),
about one in four. With four hours to the deadline and a cold QA still needed, not building is the
defensible choice.

| # | idea | source | mechanism | targets | expected size | external check under an hour |
|---|---|---|---|---|---|---|
| 1 | **Goal-kick placement is D** (conditional top pick) | Vidal-Codina p8: "goal kick trigger: at least one player is within their own goal area"; p9: trigger active until the last dead frame | New pass in `edges.py`: visible ball slower than about 1 m/s inside a goal area for >= 2 s, at most one player within 2 m, and (our addition, Law 16) no point of the other team inside that penalty area -> D until the first fast ball. `still_dead` (T120) skips exactly this case because a player is within 1 m; `taker_frames` leaves goal kicks out | visible dead ball labelled as a team (keeper placing the ball) | up to one false team spell per goal kick, 8-10 per match: at most +0.01; provided input goal-kick D agreement is 45% (definitions doc 3.5) | IDSSE only (real ball during stoppages): count firings against DFL ball status (precision gate >= 0.9), then ctl T=2 delta on idsse1/idsse2. Metrica benches will show ~0 by construction; say so |
| 2 | **Nonlinear stoppage model** | Lang p4: stoppage F1 79.8 (AdaBoost) vs 61.2 (logistic) | Replace the logistic in `shape.py` dead model by boosted stumps on the same `DEAD_FEATS` + `FAST_FEATS` (numpy, no new dependency), leave-one-match-out, same log-odds slot in the decoder | stoppage edges; interludes | paper gap is large but on 45 training matches with IDs; ours 5. T128 found +0.002 for nonlinearity on the team side | LOMO AUC and hidden-D precision first (10 min); only if AUC gains > 0.01 run real1 / real2 ctl T=2 |
| 3 | **Length model for D only** (T54, never run) | Richard p4 Eq 9 "class-dependent Poisson"; p5 Eq 10-12 | Cheap form: split DH / DA into a chain of k sub-states with a minimum stay, giving a negative-binomial stoppage length; fit from external D run lengths (histogram, not Poisson). H / A lengths stay flat (the paper's own warning, p7) | edges; interludes | small; T58 / T99 / T130 say costs and durations are flat | real1 + real2 ctl T=2 with k in {2, 4}; hidden-D precision gate |
| 4 | **Long-range context in the dead model** (+-20, +-40 s copies of mean and fast speed) | Lang p3 context list; p4 "no particularly strong weight on ... time shift = 0" | Add shifted copies with NaN fallback at period breaks. Only the long shifts are new: +-2 / +-4 s was T40 (AUC +0.003, rejected) | long hidden stoppages | small; may smear interludes further | LOMO AUC, then real1 |
| 5 | **Formation by the Laws as ball-free stoppage evidence** | Vidal-Codina p8: "neither within the penalty area nor within 9.15 m" | Feature: one team entirely outside a penalty area that holds a lone opponent near its goal area, held for >= 2 s -> D evidence on hidden frames; its break dates the exit | goal-kick stoppages with the ball hidden; exits | rare; deep defending gives the same picture less often than it seems (attackers leave the box only at goal kicks) | IDSSE: firing precision against ball status |
| 6 | **Ball-speed gate on the distance term** | Morra p6: possession needs "speed(b,t+k)<Ts2" | Above 8-10 m/s with no impulse, drop the nearest-player emission and let `w_fly` / `w_bt` decide | wrong team during passes | near zero or negative: cousin of T74 (rejected), and speed is already a `w_bt` feature | real2 4 seeds |
| 7 | **Prior correction of learned stoppage log-odds** | Richard p4 Eq 8: "p(xt|c) ∝ p(c|xt)/p(c)" | Subtract the training log prior odds from the `w_bd` term if it is not already removed (not checked) | visible dead ball | a re-scaling of one constant | real1 |
| 8 | **Set-encoder sequence model for team / D** | Kim App. A.1, Table 8 (72.61 -> 85.85% with set encoders) | Permutation-invariant team embedding + Bi-LSTM as one more emission | hidden frames | cannot be trained and gated today; 85.85% in play is below our 0.90 | not feasible tonight |

## 3. Ideas from the papers already in the system

- Possession zone 1 m and duel zone 1 m (Vidal-Codina p6, p12): `team_control`, T13; duel prevails in the touch hold, T41.
- Loss only when another player controls next, not when the ball drifts away (Vidal-Codina p7): team hold with k-frame retention, T13.
- Savitzky-Golay smoothing (Vidal-Codina p6): `decode.py`, `stoppage_ml.py`.
- Kick-off formation (Vidal-Codina p8): `kickoff.py`.
- Kick = ball leaving a near player with acceleration; pass = kick then possession by the same team (Morra p6-7): touch impulse and hold, T29, T66, T81.
- Ball out at the line (Morra p7, no rule given): `edges.cross_entry`, `cross_dead`, T112, T113.
- Ball-free in-play model from player speeds, home / away augmentation, second-half flip (Lang p3): `shape.py` (T25, T55, T34).
- Median filter and no sub-2 s stoppages (Lang p4): Viterbi switch costs, `ed_xdead_min`.
- "Controller or next controller" as the label (Kim p3): flight-direction term, T100.
- One touch per score peak (Kim p5): `smooth_touch`, T66.
- Emission x context model in a Viterbi with a restart grammar (Richard p4): `decode.viterbi`, DH / DA states.
- Label change at the midpoint between two anchors (Li p5, baseline): `place_gap_switches`, gap_pos 0.5.

## 4. Ideas the papers suggest that our tests already rejected

- Gain needs a change of ball direction or speed (Vidal-Codina p7): T74.
- Duel prevails in the distance emission: T75. 0.5 m possession zone: T76.
- Corner / throw-in taker geometry as stoppage evidence or exit date: T64 (M5), T116.
- Restart team from the taker: T16 kept, oracle +0.002; T97, T98.
- Loss when the ball enters the opponent zone: T78, T83.
- Deflection from deceleration (Morra, F 0.31 in the paper itself): T81 covers the contact; T109 rejected.
- Lead / lag +-2, +-4 s in the stoppage model (Lang): T40. Context of the hidden block: T138.
- Stoppage duration statistics as clean-up rules (Lang p8): T58.
- Ball imputation from players (Kim): T03-T10. Reality loss / pass-by: T74. Snap to touches: T140.
- Ball-player features, nearest players (Sanford, Kim Table 7): T134 flat; nonlinear team model T128 neutral.
- Boundary re-dating by local features (Li Eq 1): T139, T140 for team boundaries; exit + entry snap T60, T61.
- Smoothing / minimum segment rules (Li smoothing loss in kind): T58.
- Threshold search by genetic algorithm (Morra p9): not run, but constants are flat under scans T45, T99, T103, T131.
