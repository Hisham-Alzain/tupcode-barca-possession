# L3 — common rules for every independent solver (read fully before anything else)

You work ALONE in your own git worktree (your current directory) on your own branch. Do not look for or coordinate with other agents. Do not touch the main checkout.

## Task context
Hackathon: label every frame of a football match H / A / D (home, away, dead ball) from tracking data only (22 player dots + a ball that is missing ~41% of frames). Score = spellF1 (possession spells matched one-to-one by temporal overlap). Pipeline: `src/possess/` (features, stoppage detection, team-shape models, touch detection, Viterbi decoder, post-passes). Read `CLAUDE.md`, `.planning/STATE.md`, `.planning/TRIED.md` (every idea already tried: do NOT repeat a row without a stated new angle), `.planning/DECISIONS.md` ADR-013..019, `docs/TECHNICAL.md`.

## Mandatory reading
The Vidal-Codina et al. paper (automatic event detection in football from tracking data): `D:\ITE\year5\graduation project\event detection\automatic event detection research paper.pdf` (Read tool, `pages` parameter, ≤ 20 pages per call). Use its possession / duel / pass / set-piece definitions and thresholds as the reference; cite section or table when you port something. You may also read `docs/research/**`.

## Hard rules (violation = disqualification of the whole entry)
- Decide and tune ONLY on the external tune matches (metrica1, metrica2, idsse1) via `scripts/evaluate.py --bench real2|real1` and the original bench (no `--bench`). ALWAYS pass `--no-public`. NEVER run `--split heldout`.
- NEVER open `third_party/mtah/evaluation/gt/*` (public ground truth). NEVER fit, tune or choose anything using the provided match (`data/challenge`); you MAY read its INPUT for aggregate transfer checks only. Never write under `data/challenge/tactical/`.
- Any switch that behaves differently by match type must be justified from external data, not from the provided match's statistics.
- Repo stays self-contained: no imports or paths outside the repo (the paper path is for reading only).

## Decision rule (ADR-016)
A change is adoptable only if: neither real2 nor real1 tune mean drops by more than 0.003 AND at least one gains; original bench drop ≤ 0.005; paired 2-minute block bootstrap CI90 lower bound > 0 on the gaining bench; ≥ 3 real2 seeds (built in memory, no cache) with median gain ≥ +0.005. Baseline at start (commit on `main`): real2 0.5326, real1 0.6068, orig 0.6283 — re-measure first. A clear, diagnosed negative is a valid result. Never relax a gate.

## Machine limits (16 GB laptop, several agents share it)
- EVERY command that loads match data (evaluate.py, diagnose.py, bench, pipeline, anything reading `data/external`) MUST be run through the lock: `uv run python scripts/ramlock.py -- <command>`. Exit code 75 = busy, nothing ran: call it again. Never bypass it. One heavy command at a time, foreground, < 10 minutes each, grids ≤ 30 configs in total.
- New cache files only under new names that start with your agent id. Never overwrite an existing cache.
- `uv run pytest -q` (light) may run without the lock.

## Deliverable
- Code + tests on your branch; `evaluate.py` must reproduce your result (state whether you made it the default in `shape.DEC_SHAPE` or left it opt-in via `--dec key=value`; prefer opt-in unless every gate passed).
- Commit on YOUR branch only (`git add` specific files; `git commit`). Never push, never switch branch, never merge.
- Output envelope `.planning/envelopes/<your id>.out.md` (≤ 300 words): diagnosis, what the paper says, the change, real2 / real1 / orig per match with possession counts, bootstrap + seeds, transfer aggregates on the provided match's input, exact reproducing commands, risks, files changed. Return the same as your final message.
