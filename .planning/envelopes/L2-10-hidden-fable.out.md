```yaml
id: L2-10-hidden-fable
status: win   # default switched (shape.TEAM_CTX=1); tune only; held-out / public window NOT looked at; not committed
matchlens_delta: nothing in MatchLens gives ball-free team evidence (L2-7 check stands); change is inside our shape.py team model
```
**Diagnosis (tune, default 0.5679):** 375 hidden blocks, 250 contain a true stoppage; only 111 of the 306 hidden turnovers lie in all-live blocks. Block edges are already right (92% / 85%). On hidden live frames the shape model names the right team 0.854, but by true possession length: < 2 s 0.45, 2-4 s 0.55 (chance), 4-10 s 0.76, >= 10 s 0.93. Oracle, team perfect on hidden frames: all 0.7178, possessions >= 4 s only 0.6020, >= 10 s 0.5745. More shape weight on hidden frames only hurts (x2 0.5605, x3 0.5523): the evidence, not the switch cost, is the limit.

**Change:** `shape.team_context`: 4 columns added to the team model from the cached columns (no cache rebuilt): centroid-velocity difference 2 s ahead and 2 s behind, change of width and of length across +-2 s (players react 1-2 s after a turnover). Model `team_ctx`, fitted inside `fit_models`; base model where the +-2 s rows are missing. Hidden frames right 0.854 -> 0.886 (4-10 s: 0.759 -> 0.824); nearest-opponent features added nothing. Decoder constants unchanged.

**Tune LOMO:** 0.5679 -> **0.5845** (m1 0.630 325/357, m2 0.585 298/330, i1 0.539 367/511). Hidden-turnover recall 0.08 -> **0.18**, precision 0.19 -> 0.36. `w_shape` 0.075 gave 0.5901: not adopted (noise). 25 decoder configs, 15 feature sets.

**Noise gate (models fitted clean), off -> on:** sigma 0.10 m 0.5675 -> 0.5835; 7-12 players 0.5330 -> 0.5487; both 0.5327 -> 0.5489; both + ball 0.10 m 0.5279 -> 0.5431. All three matches up each time.

**Reproduce:** `uv run python scripts/evaluate.py --variant L2-10-hidden --no-public` -> 0.5845; `scripts/diagnose.py [ctx=0]`; `uv run pytest` (100 pass).

**Risks:** lag 2 s set a priori, feature set picked on LOMO frame accuracy over the same three matches. Possessions < 4 s stay invisible without the ball. A lag across the half-time break reads the other half for 2 s. Hidden stoppages are now the larger loss (oracle +0.04). Shipped `MODELS["team_ctx"]` = fit on all three tune matches; a new submission draft is needed.
