Envelopes: `<id>.in.md` (Queen → worker) and `<id>.out.md` (worker → Queen, ≤ 300 words). Format in `docs/AGENTIC_PLAYBOOK.md` §3. Every input envelope's constraints must include: "Never read anything under data/challenge/."

Every input envelope must carry a `matchlens_start` field (CLAUDE.md rule): the MatchLens file(s) and function(s) the work starts from, and the measured MatchLens number to beat. Every output envelope must have a `matchlens_delta` line: what was kept, what was changed, and the before/after number on the same data.

Template:
```yaml
id: D1-touches-01
goal: one sentence
matchlens_start:
  files: [matchlens_ref/impulse.py:detect_impulses, matchlens_ref/possession.py:PossessionTracker._on_touch]
  baseline: "touch recall 0.xx on degraded Metrica G1 (eval/results/<file>.json)"
context_refs: [paths, not content]
owned_files: [...]
constraints: ["Never read anything under data/challenge/", ...]
deliverable: code + tests + out envelope
done_when: beats the matchlens_start baseline on the same data, or states why not
budget: 1 round
```
