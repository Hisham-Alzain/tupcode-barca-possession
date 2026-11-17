# Agentic Playbook — lessons from Padel Lens, adapted to barca-possession

Source: 4 Claude Code sessions in `D:\TupCode\Padel Lens` (2026-10-31 → 11-11, 110 user prompts, ~100 subagents, 50 commits, 42 ADRs) plus its `CLAUDE.md`, `PROMPT.md`, `KICKOFF.md`, `.planning/`.

---

## 1. Metacognition — the thinking habits that made it work

These are *how* the work was reasoned about, independent of padel.

1. **Define "done" as numbers before building.** PROMPT.md §2 had a gate table (recall ≥ 0.95, ID switches ≤ 1/min…). Every loop iteration was judged against it. Rule: *never lower a gate silently; report real numbers, including failures.*
   → Here: spellF1 on degraded external matches, with bootstrap CI, beats baseline 0.24 by a stated margin.
2. **Measure before choosing.** "No model locked without benchmark numbers on our clips." Phase 1 was *only* benchmarking; picks came after, via council, one ADR each.
3. **Protect held-out data like money.** Reserved split (client frames ≥ 33000) needed an explicit `--allow-gate` flag; Barcelona match M was held out and *every look was counted* ("M has been looked at 4 times: closed for tuning"). → Here the stakes are higher: challenge data is off-limits to AI entirely (RULES.md).
4. **Pre-register, then run.** Commits like `Pre-register hierarchical shot types…` landed *before* results; the result was then accepted or rejected against the pre-registered rule ("lost to ensemble 0.600 vs 0.735 — kept as option, not adopted"). Kills p-hacking and wishful merges.
5. **Suspect circularity.** STATE flagged "labels were pre-filled from model output: metrics partly circular"; QA flagged "pseudo-hitter metric is circular". Always ask: *could this number be high because the test shares a source with the thing tested?* → Here: degradation must be calibrated on challenge *input* stats only, never on outputs.
6. **Find the real lever, not the fun one.** Fable council: "Stop treating accuracy as a modelling problem first" — the bottleneck was labels and upstream ball recall (28 % visible). Ask "what upstream signal throttles everything downstream?" before tuning the model.
7. **Honesty in output.** UI hid momentum below 50 % known winners; uncalibrated estimates behind a toggle; a test asserted no UI text claimed "accuracy". Say what you don't know.
8. **Scope discipline.** Milestone 1 = detection + tracking only; everything else → `BACKLOG.md`. Scope widened only by explicit ADR (ADR-023/024) when the user asked.
9. **Know when to stop.** Loop stop conditions: all gates pass, *or same gate fails twice*, *or a decision needs the user*.

## 2. Harness engineering — the files that carry the brain

| File | Role | Rule |
|---|---|---|
| `CLAUDE.md` | Project rules, read every session | Short. Points to STATE + plan, never duplicates them. |
| `PROMPT.md` (master prompt) | Inputs, mission, DoD gates, roles, envelope spec, phases, hard rules | Written once with the user, then referenced by § number. |
| `KICKOFF.md` | Copy-paste prompt per phase + `/clear` between phases | Turns the plan into runnable commands. |
| `.planning/STATE.md` | ≤ 1 page, overwritten: phase, gate table with real numbers, next action, waiting-on-user, risks | The answer to "what is remaining?" (asked ~10×). |
| `.planning/DECISIONS.md` | ADR per non-trivial choice: date · decision · options · why · who · reversible? | Includes user decisions and QA amendments. |
| `.planning/LOG.md` | One line per action: date · phase · action · envelope/commit · outcome | Append-only. |
| `.planning/envelopes/` | `<id>.in.md` / `<id>.out.md` per handoff | Out ≤ 300 words. |
| `.planning/council/` | Brief + synthesis per council | Blind positions kept on file. |
| `eval/results/*.json` | Every run with commit hash + config | Reproducibility. |
| `.claude/settings.json` | `deny` writes to reference repo, `deny git push`, allow test/eval commands | Rules enforced by harness, not by memory. |
| Memory | Durable facts (e.g. "academic, not commercial" → ADR-030) | Correct a wrong premise once, permanently. |

## 3. Agentic engineering — the operating model

**Roles.** Queen = Opus orchestrator (owns plan, never reads raw dumps). Builders = Opus, one component each, `isolation: "worktree"`. Cold QA = fresh Opus per branch. Advisor = Fable via `ai-council`. Delegates = Sonnet (README, fetch scripts).

**Envelope (input).**
```yaml
id: P3-ball-01
goal: one sentence
context_refs: [paths, not content]
owned_files: [only these may change]       # prevents worktree merge conflicts
constraints: [API signature, data limits, CPU/GPU, "never read X"]
deliverable: code + tests (named synthetic cases) + out envelope
done_when: measurable check
budget: 1 round
```
Builder prompt = 5 lines: "you're in a worktree; read your envelope; env rules (shared venv via `uv run --no-sync --project <main>`, data only in main checkout, never write cache); commit on your branch; reply only branch + status."

**Cold QA prompt pattern** (the highest-value agent): review branch vs main *for correctness, not style*; give it the spec and the author's report; list concrete suspect bug classes (off-by-one/stride, crossing boundaries, edge frames, unit mix-ups, tests that don't test their names); run tests; don't edit; reply ≤ 250 words: verdict (merge / merge after fixes / don't) + file:line + failing input + severity. Then a fixer agent applies, and QA findings become an ADR amendment.

**Council pattern.** Brief file → 4 blind Fable members with distinct lenses (domain expert, skeptic/red team, pragmatist, architect / thesis examiner) → each writes ≤ 350 words with fixed headings (Position, Reasoning, Risks, What would change my mind, Confidence %) → synthesis with consensus, dissents, aggregate confidence. Skip cross-examination when blind positions converge.

**Parallelism.** Phase 2: 5 work packages (contracts first, then 4 worktrees). "Work in parallel" was the user's most repeated instruction — default to it whenever `owned_files` don't overlap. Shared contracts/schemas land on main *before* the parallel fan-out.

**Session rhythm.** One phase per session → update STATE/LOG/ADR → commit locally → `/clear`. Resume = "read CLAUDE.md + STATE.md". `/loop` (self-paced) with explicit stop conditions for the build phase.

## 4. What went wrong (fix these here)

- **Request lost across compaction** — user had to ask "what happened with this prompt" (shot enhancement, 11-09). → On every multi-part request, write it into STATE.md "Open requests" immediately.
- **Resource kills** — renders/timing runs killed at < 3 GB free RAM; GPU contention between parallel workers. → Envelopes state CPU/GPU ownership; heavy runs in child processes; state RAM need in STATE.
- **Permission denials / logins** interrupted flow. → Pre-allow test/eval commands in settings.
- **Wrong premise** (commercial vs academic) excluded datasets for days. → Confirm project context in ADR-000 on day 1.
- **Labels pre-filled by the model** made metrics circular. → Never score on anything the pipeline produced.

## 5. Translation to barca-possession (hackathon, deadline 2026-11-17)

- **Start from where MatchLens ended.** Padel Lens ported Match Lens ideas file by file ("cite the source file in the envelope"); here the rule is stricter because MatchLens already solves the adjacent problem. Every feature begins with the MatchLens code that does the closest thing, measured first on the external set, then ported and changed only where the task differs. Envelopes carry `matchlens_start` (in) and `matchlens_delta` (out). See CLAUDE.md.

- **Hardest rule:** AI never reads `data/challenge/` or its outputs. Enforced in `.claude/settings.json` deny rules + CLAUDE.md; tell every subagent in its envelope constraints. Consider a PreToolUse hook that blocks any tool input containing `data/challenge` (deny rules don't cover every Bash form).
- **Gates:** spellF1 (organisers' scorer) on degraded Metrica/IDSSE held-out split, mean + bootstrap CI; fragmentation count; runtime; pytest green.
- **Held-out discipline:** split external matches tune/held-out; count held-out looks in STATE.
- **Parallel work packages** (after `load` + schema land): ball cleaning · touches · dead-ball · shape classifier · tags · degradation simulator · scorer harness. Then fuse + Viterbi.
- **Lighter process for a 7-day sprint:** envelopes + cold QA + ADRs stay; councils only for hard-to-reverse calls (segmentation model, degradation design).
