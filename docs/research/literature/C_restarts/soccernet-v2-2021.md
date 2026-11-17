# SoccerNet-v2: A Dataset and Benchmarks for Holistic Understanding of Broadcast Soccer Videos

- **Authors:** A. Deliège, A. Cioppa, S. Giancola et al.
- **Year / venue:** 2021, CVPR Workshops (CVSports)
- **URL:** https://arxiv.org/abs/2011.13367 · supplement on openaccess.thecvf.com
- **Verified from:** search snippets plus a survey (arXiv 2309.12067).

## Key facts
- Action spotting from video, 17 classes: Penalty, Kick-off, Goal, Substitution, Offside, Shot on/off target, Clearance, **Ball out of play**, **Throw-in**, Foul, Indirect/Direct free-kick, **Corner**, cards.
- Definitions: **Ball out of play** = the moment the ball crosses one of the outer field lines. **Throw-in** = the moment the player releases the ball. **Kick-off** = the first pass between the two players in the centre circle.
- Counts: ball out of play 31,810, throw-in 18,918, kick-off 2,566, corner 4,836.
- The 2024 ball-action-spotting task (12 classes, with "Out") is a different label scheme.

## Relevance to us
- Gives the convention for a stoppage's **start = the line crossing** and its **end = the restart action**. This matches what an operator flag (IDSSE) is likely to mark.
- Rates: about 32k ball-outs and 19k throw-ins in 500 games, roughly **64 ball-outs and 38 throw-ins per game**. Throw-ins are about 60% of line exits, so throw-in team inference matters most. Smart Tagging has only 26 throw-ins, so it misses many.
- Video only. No tracking method to reuse.
