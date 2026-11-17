# StatsBomb / Opta conventions for ball-out and restarts

- **Sources:** StatsBomb Live Data API event descriptions https://live-data-api-guide.statsbomb.com/api-reference/event-descriptions.html · StatsBomb glossary https://stats-portal.statsbomb.com/glossary · Opta Event Definitions https://statsperform.com/opta-event-definitions · throw-in study (Sheffield Hallam) https://shura.shu.ac.uk/35132/
- **Year:** current documentation. **Verified from:** search snippets only.

## Key facts
- StatsBomb has an `out` event ("Ball goes out of play"). Each event carries `play_pattern` (for example From Throw In, From Corner) and `possession_team`. A new possession starts at a restart. The **exact play-pattern and possession rules were not found**.
- Opta counts a throw-in as part of a set-piece pattern only until play returns to open play. Throw-ins are not counted as passes.
- A StatsBomb 360 throw-in study **assumed a ball kicked out always gives the opponent the throw-in**, the Law 15 last-touch rule.

## Relevance to us
- Event-data providers treat every restart as the start of a new possession, owned by the restarting team. Our "keep the pre-stoppage team" rule contradicts this whenever the team that sent the ball out is the one that had it. That is the usual case for an attacking team's ball going over the goal line (goal kick to the defenders) or a cleared ball (throw-in to the opponents). How often the restart flips the team: **not found** as a published number. Measure it on IDSSE or Metrica events.
