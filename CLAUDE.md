# barca-possession — project rules

Hackathon (More Than A Hack 2027): per-frame team possession H/A/D/X, scored by spellF1. Deadline 2026-11-17 23:59 CET. Plan: `docs/PLAN.md`. Rules: `docs/RULES.md`. Operating model: `docs/AGENTIC_PLAYBOOK.md`.

## Session start
1. Read `.planning/STATE.md` (incl. "Open requests"), then only the `docs/PLAN.md` section you need.
2. Before touching data, re-read `docs/RULES.md`.

## Hard rules (violation = disqualification)
- `data/challenge/` may be read by agents and scripts (organisers' second clarification, ADR-001), but never written under `data/challenge/tactical/`, never committed, and never used to train, fit or tune anything.
- No tuning, fitting or calibration on challenge data or ground truth. Tune on `data/external/` only; record the run that chose each parameter.
- Public 5 min: scored and reported for every version (user request, LOOP.md amendment 2026-11-13), never used to choose or tune.
- Never lower a gate silently. Report real numbers, including failures.
- Never push without asking. Commit locally at the end of each work package.

## Start from where MatchLens ended (every feature, no exceptions)
MatchLens (original, read-only, outside this repo: footballtactics `backend/events/`) is the starting point for every component, not an optional reference. **The repo must stay self-contained**: never import or path-reference MatchLens; port only the pieces a component needs into `src/possess/` with a source citation (as `stoppage_ml.py` does). Before building anything:
1. **Locate** the MatchLens code that does the closest thing (`possession.py`, `impulse.py`, `kinematics.py`, `deadball.py`, `detectors/set_piece_detector.py`, `attacking_direction.py`, `eval/metrica.py`, `eval/sportec.py`). Name the file and functions in the envelope's `matchlens_start` field.
2. **Measure it first**: run the MatchLens version on the external eval set and record its number (v0 baseline: `eval/results/matchlens_v0_public5.json`, `matchlens_replay_metrica_clean.json`). The new component must beat that number on the same data, or the envelope says why it was replaced.
3. **Port, then change**: keep its constants, thresholds and kinematics unless a measured run shows a better value; cite `source_file:function` in a docstring at every port. Rewrite the state logic only where the task differs (offline, team-level, no IDs, ball missing 41%).
4. "Nothing in MatchLens covers this" is a valid answer, but it must be stated in the envelope after looking, with the files checked.
The footballtactics repo is read-only; never edit it.

## Context engineering
- Every subagent handoff = envelope in `.planning/envelopes/<id>.in.md` / `.out.md` (format in playbook §3). Out ≤ 300 words. Workers read only `context_refs` and change only `owned_files`.
- Raw logs and benchmark dumps go to files, never into the orchestrator's context.
- Independent packages build in parallel worktrees (`isolation: "worktree"`); shared schema lands on main first.
- Each builder branch gets a cold QA agent (correctness, not style) before merge.

## Models by role
Orchestrator / routine builders: Opus. **Hard tasks (design, diagnosis of a failed idea) and all validation / cold QA: Fable** (user, 2026-11-13); councils via `ai-council` on Fable. Mechanical work: Sonnet.

## Recording
- `.planning/STATE.md` — ≤ 1 page, overwrite; gate table with real numbers; "Open requests" list (write user asks there immediately).
- `.planning/DECISIONS.md` — ADR per non-trivial choice.
- `.planning/LOG.md` — one line per action.
- `eval/results/*.json` — every scored run with commit hash + config.
- Scope creep → `.planning/BACKLOG.md`.

## Stack
Python 3.12, `uv` (pinned in `pyproject.toml` + `uv.lock`). Setup `uv sync`. Test `uv run pytest`. Submission `uv run possess --data data/challenge --out submissions/<team>.csv`. Bench `uv run python scripts/bench.py --pipeline possess --split tune`. Official scorer vendored in `vendor/mtah/`. Pipeline in `src/possess/`.
