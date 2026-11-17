```yaml
id: L2-9-ballclean-fable
role: Fable designer/builder
goal: Make the touch emission robust to real tracking faults by cleaning the ball track before decode.touch_hold (plan step 2 "clean the ball"), without losing tune score; then, if time allows, attack the next cause (turnovers inside hidden-ball blocks, recall 0.08).
matchlens_start:
  files: [MatchLens events/impulse.py + kinematics.py (ported: decode.touch_hold, stoppage_ml.BallSpeed); MatchLens core ball tracker coasting idea]
  baseline: "default tune 0.5657 (m1 0.603, m2 0.564, i1 0.530); held-out 0.5584 (3 looks). Fable QA L2-8: a 1-frame 5-10 m teleport adds ~3.7 false touch frames; teleports 1 per 2 s -> 0.5376; stale 3-8 frame repeats 1 per 2 s -> 0.5371"
context_refs:
  - .planning/envelopes/L2-8-segment-fable.out.md and the QA findings above
  - docs/research/swarm-findings.md "Data problems" (challenge ball: jumps up to 33-40 m/s per 0.2 s, stale repeated values like x = -51.8 m, readings 4-8 m off the pitch)
  - src/possess/{decode.py, pipeline.py, features.py}, scripts/{evaluate.py, diagnose.py}
owned_files: [src/possess/ballclean.py (new), src/possess/decode.py, src/possess/pipeline.py (only to call the cleaner on the ball passed to touch_hold), tests/test_ballclean.py (new), tests/test_decode.py, scripts/diagnose.py, .planning/envelopes/L2-9-ballclean-fable.out.md]
design_hints:
  - Remove (set NaN, i.e. treat as hidden) single/short teleports (implied speed > ~35 m/s that returns), stale repeats (identical position held while players move), and far off-pitch readings; apply only to the ball fed to touch_hold first (do not change distance features/caches unless measured to help).
  - Acceptance test (report all): clean tune score (must not drop by more than 0.003 vs 0.5657); tune score with injected faults at 1 per 10 s and 1 per 2 s for teleports and for stale repeats (should recover most of the loss); noise 0.05 m.
  - Optional second hypothesis: hidden-block turnovers (team shape / player-anchored change across the block) — one hypothesis at a time, each measured.
constraints:
  - Tune ONLY on metrica1, metrica2, idsse1; never public5/challenge data for choices; no held-out.
  - RAM: separate-command check >= 3 GB before every run (powershell ... FreePhysicalMemory, or grep MemFree /proc/meminfo); foreground; < 10 min per command; no cache recomputation; grids <= 30.
  - evaluate.py must reproduce any default you set with no flags. Repo self-contained. Do not commit.
deliverable: code + tests + output envelope (<= 300 words) with clean and fault-injected numbers before/after, reproducing command, risks.
done_when: fault-injected scores recover with clean tune >= 0.5627, or a clear negative.
```
