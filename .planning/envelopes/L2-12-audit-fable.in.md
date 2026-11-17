```yaml
id: L2-12-audit-fable
role: Fable auditor/builder (hard task: validation)
goal: Find out how much of the current default's tune/held-out gain depends on bench-specific ball behaviour that the real match does not have, and make the bench more realistic so future choices transfer. Deadline pressure: submission 17 Nov.
background: |
  L2-11 was rejected (ADR-010): its +0.028 came from the bench's dead ball (IDSSE freezes it at speed 0; Metrica's is a drawn straight fill line), while the real match has a still ball on only 4.5% of visible frames vs 21-42% on the bench. Held-out matches share the same bench, so they share the blind spot. Touch detection (ADR-007) took ~45% of its tune gain at stoppage boundaries.
context_refs:
  - .planning/LOOP.md (all rows, protocol amendment after L2-11), .planning/DECISIONS.md ADR-005..010
  - scripts/bench.py (get_case: show_dead_ball, hide_mask), scripts/evaluate.py, scripts/diagnose.py
  - src/possess/* (current default pipeline)
  - challenge INPUT only: data/challenge/_runs/features_v1.npz and positions_v1.npz, Smart Tagging XML (automatic). NEVER read the public ground-truth CSV for this task.
owned_files: [scripts/bench.py (new degradation options only; keep default behaviour reproducible), scripts/audit_bench.py (new), scripts/transfer_check.py (new), .planning/envelopes/L2-12-audit-fable.out.md, eval/results/audit_*.json]
tasks:
  1. Input-level transfer table: for every ball-based signal the default uses (visibility, gap lengths, still-ball share, speed distribution, touch rate per minute, distance-to-nearest-player distribution, dead-share of the stoppage detector), compare the provided match (input stats only) with each tune/held-out match. Flag every signal whose distribution differs materially.
  2. A "realistic bench" option in bench.get_case (new cache name, default unchanged): e.g. during true stoppages, a ball that drifts / is carried / is hidden ~28% of the time instead of a straight fill line or a frozen ball; challenge-like faults (sample-and-hold at ~5 Hz, long freezes, jumps) calibrated on provided-match INPUT statistics only.
  3. Re-score the ADOPTED steps (ADR-005..009) one by one on the realistic bench (LOMO where learned; refit shape models only if the bench change requires it, report both) — does each step still help? Report a table: step, original bench tune, realistic bench tune.
constraints:
  - Do not change pipeline defaults; this is an audit. No tuning of pipeline parameters on the provided match. No held-out on the new bench unless all tune numbers are ready (and then one combined look, report it as look 5).
  - RAM: separate-command check >= 3 GB before every run; foreground; < 10 min per command; new caches only under new names.
  - Do not commit.
deliverable: output envelope (<= 350 words) with the transfer table, the realistic-bench design, the per-step re-score table, and a recommendation: which steps are safe, which are at risk, and whether any default should change before submission.
```
