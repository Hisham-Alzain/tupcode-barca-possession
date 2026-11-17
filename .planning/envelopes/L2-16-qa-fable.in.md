```yaml
id: L2-16-qa-fable
role: Fable cold QA (correctness, not style)
goal: Decide whether the uncommitted L2-16 change (fast-player speed features ps90/ps90_5 in the LOMO stoppage logistic, model dead_fast, DEAD_FAST=1, refit shape.MODELS) is correct, leak-free and safe to make the default. Verdict PASS / FAIL with findings.
context_refs:
  - git diff (src/possess/shape.py, scripts/tune_shape.py, scripts/diagnose.py, tests/test_shape.py), new eval/results/eval_L2-16*.json, transfer_hidden_stop.json
  - .planning/envelopes/L2-16-hidden-stop-fable.in.md and .out.md (builder's claims to verify)
  - CLAUDE.md, .planning/LOOP.md, .planning/TRIED.md, ADR-005..013
rules_reminder: Agents MAY read the provided match's INPUT under data/challenge (ADR-001) for aggregates; NEVER open the public ground-truth CSV (third_party/mtah/evaluation/gt/*), run evaluate.py with --no-public; never tune on the provided match. No held-out runs.
owned_files: [.planning/envelopes/L2-16-qa-fable.out.md]
checks:
  - LOMO integrity: the tune number for each match must use stoppage models fitted without that match (incl. the new dead_fast model and any standardisation/scaling stats). Shipped shape.MODELS refit only on the three tune matches — verify nothing from held-out/challenge enters the fit.
  - ps90 / ps90_5 computation: NaN/untracked players, period boundaries, frame rate, window alignment (causal vs centred), team-agnostic; same code path on the challenge match (positions cache) as on the bench.
  - `fast=0` reproduces 0.5935 on real1 (refactor inert); default reproduces 0.6061 real1 / 0.6283 orig with no extra flags; `uv run pytest` passes.
  - Transfer finding: hidden D share on the provided match rises 0.673 -> 0.711 while real1 stays flat — is this a red flag (over-calling D on the real match) or consistent with the dhide sensitivity runs? Read the transfer JSON; you may compute input-only aggregates on the provided match.
  - Selection bias (~20 features screened, 13 decodes): is the gain robust (p75/top-3/max variants 0.602-0.606)?
constraints:
  - RAM: check >= 3 GB free before every run (python ctypes GlobalMemoryStatusEx); foreground; one run at a time; < 10 min per command.
  - Do not edit source files. Do not commit.
deliverable: output envelope <= 300 words: verdict, findings with file:line, commands run.
```
