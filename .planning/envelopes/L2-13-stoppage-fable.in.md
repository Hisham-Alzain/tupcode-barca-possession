```yaml
id: L2-13-stoppage-fable
role: Fable designer/builder (hard task)
goal: Improve stoppage detection when the dead ball MOVES (carried, thrown back, rolled to the spot), the largest exposure found by the L2-12 audit, so that the default's tune mean on the REALISTIC bench (`real1`) beats 0.5276 without losing on the original bench (0.5845 +- 0.005).
matchlens_start:
  files: [src/possess/stoppage_ml.py (MatchLens deadball port: OOB, placed ball, hysteresis), MatchLens detectors/set_piece_detector.py ideas (restart formations, Laws 8/15/16/17), src/possess/shape.py stoppage model (player-only)]
  baseline: "default on real1 0.5276 (orig bench 0.5845); stoppage D recall on visible frames real1 0.40/0.43/0.57 vs orig 0.70/0.72/0.95; on the provided match the detector marks only 7.9% of visible frames dead (realistic bench 18-24%, true dead share there 35-41%)"
context_refs:
  - .planning/envelopes/L2-12-audit-fable.out.md (realistic bench design, transfer table), L2-11 envelopes + ADR-010 (why ball-still rules fail to transfer)
  - .planning/LOOP.md protocol incl. the 2026-11-14 transfer-check amendment
  - docs/research/literature/C_restarts/survey.md, docs/research/swarm-findings.md (dead ball = ball far from players + near a line + collective slowdown; ball still visible on ~72% of D frames)
  - scripts/{bench.py (variant=), evaluate.py, diagnose.py, transfer_check.py, audit_bench.py}
owned_files: [src/possess/deadball2.py, src/possess/shape.py, src/possess/decode.py, src/possess/pipeline.py (DEC/DEAD2 constants only), scripts/evaluate.py (add `--bench real1|orig`, default orig), scripts/diagnose.py, scripts/tune_shape.py, tests/*, .planning/envelopes/L2-13-stoppage-fable.out.md]
ideas (not orders): player-formation evidence of a restart (one player at the ball near a line/corner/spot, others set; distances per Laws), collective slowdown + ball near a line, player-only stoppage model retrained to be robust to moving dead balls (it uses positions only, so it should transfer), duration priors for stoppages in the decoder.
constraints:
  - Decide on `real1` (tune split, LOMO) AND report the original bench; run scripts/transfer_check.py for any new signal: its rate on the provided match's input must resemble the realistic bench. Never read the public ground-truth CSV; never tune on the provided match.
  - RAM: separate-command check >= 3 GB before every run; foreground; < 10 min per command; no recomputation of existing caches (new caches under new names only); grids <= 30.
  - No held-out. evaluate.py must reproduce any default you set with no extra flags (plus --bench real1). Do not commit.
deliverable: code + tests + output envelope (<= 300 words): real1 and orig tune per match, D recall/precision on visible frames, transfer-check numbers, reproducing commands, risks.
done_when: real1 tune > 0.5276 with orig >= 0.5795 and a passing transfer check, or a clear negative.
```
