id: L3-A4-short-possessions — DONE, adopted as default (all gates passed)

**Diagnosis (real2 s0, `scripts/a4_short.py`).** 286 short visible true possessions are missed, and 215 of them have no touch hold for their team. On a low-passed track the impulse peak sits about 2 m off the toucher, so the team test fails or becomes a contest. The rest are lost in merged flurries: the predicted spell runs 2-10 s too long.

**Paper §2.3.3.** A gain is judged by incoming vs outgoing direction or speed across the whole control interval.

**Change.** `decode.touch_vertex` rebuilds the contact point from straight incoming and outgoing flight. The touch team is read there, on low-passed tracks only. The radius is 2 m, the same as SMOOTH_TOUCH, so no new constant. On live peaks the touch team is right 0.98-0.99 of the time instead of 0.94-0.97. Default: `DEC_SHAPE vtx=2.0`. `--dec vtx=0` gives L2-22.

**Results (spellF1, pred/true):**
- real2 0.5326 -> **0.5618**: metrica1 0.604 (271/357), metrica2 0.582 (278/330), idsse1 0.499 (302/511). Bootstrap +0.029, CI90 [+0.013, +0.046].
- Seeds 1, 2, 3: +0.033, +0.031, +0.026, every CI90 lower bound > 0. metrica1 loses 0.006-0.010 on these seeds.
- real1 0.6068 and orig 0.6283: unchanged (vtx does not run on those tracks).
- Provided input: 3.2% of frames change; possessions per half hour 120.1 -> 116.7.

**Negatives.** Touch window: neutral. Hold release in the opponent's zone: -0.013 to -0.073 (T74-T75). 11 configs.

**Risks.** Possession count drops further (under-segmentation stays). Inherits ADR-017's compliance status.

**Reproduce.** `ramlock.py -- uv run python scripts/evaluate.py --variant a4_vtx2 --bench real2 --no-public`. Seeds: `a4_short.py save seed=N`, then `run seed=N vtx=2.0`.

**Files.** possess/decode.py, pipeline.py, shape.py; scripts/a4_short.py; tests/test_vertex_touch.py (+2 default tests); TRIED.md.
