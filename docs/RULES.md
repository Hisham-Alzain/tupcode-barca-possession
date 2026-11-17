# Competition rules — the lines we do not cross

Source: organisers' "Clarification of the rules" email (takes precedence over all earlier docs/answers).

**Update 2026-11-11 (ADR-001):** a second "Clarification of the rules" email from hackathon.talentarena@mobileworldcapital.com (Wed 11 Nov 2026, 17:01) states: *"Yes you can use agents to read raw data"*, and that the rules in that email take precedence over earlier information. AI agents may now read the raw files under `data/challenge/`. Its rule list, verbatim:

> **What is not allowed**
> - Training, fitting or calibrating your algorithm on the ground-truth possession data or on the provided video.
>
> **What is allowed**
> - Running automatic checks using only the provided data.
> - Inspecting the provided data and the video to better understand what counts as possession.
> - Using as input the video, the automatic tracking data, the smart tagging or any other generated data, as long as your algorithm was not trained on the ground truth or on the provided video.
> - Providing files of intermediate steps.
>
> **Training and tuning** — If you need to train (or retrain), you MUST use only external data. If your algorithm has parameters that need to be tuned, you MUST tune them on external data, never on the ground-truth possession data or on the provided video.

The email no longer lists "uploading to AI services" or "checking output against the video". We still keep the official repository's RULES.md lines (no hand labels on the game, no frame-specific rules), because nothing in the email contradicts them.

## Not allowed
- Checking our output against the video.
- Hand labels or any manually created data on the provided game.
- ~~Uploading the provided data to AI services, incl. letting an AI agent read `data/challenge/`.~~ Superseded by ADR-001: agents may read the raw data.
- Training, fitting or calibrating on the ground-truth possession data or the provided video.
- Rules or constants tied to specific frames of the provided game (from the official RULES.md), even when an agent finds them.

## Allowed
- Automatic checks using only the provided data.
- Inspecting the provided data and video to understand what counts as possession.
- Using video, automatic tracking, Smart Tagging and other generated data as *input*.
- Any dataset we have the rights to (check licences).

## How this repo enforces it
- Every tunable parameter is tuned on `data/external/` only (Metrica, IDSSE, ...). Record the tuning run that chose it.
- The public 5 minutes are used **only** for format sanity (frame offset, H/A swap, frame count) — never to compare versions or pick thresholds.
- Agents may read `data/challenge/` raw inputs (ADR-001) to understand the problem. What they learn becomes general mechanisms, validated and tuned on external data; never a frame-specific rule or a constant fitted to this game.
- The public 5-minute ground truth may be read to understand what the labels mean (official RULES.md), never to tune or choose versions.
- Raw challenge files are read-only: nothing writes into `data/challenge/tactical/`.

## Dates
- Team registration: 2026-11-10 (all members, same team name).
- Submission instructions: 2026-11-14.
- Deadline: 2026-11-17 23:59 CET.
