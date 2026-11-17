# Open-source tools: databallpy, kloppy, unravelsports (possession handling)

- databallpy (Alexander Oonk et al.; JOSS 10.21105/joss.10223): https://databallpy.readthedocs.io (features/team_and_player_possession)
- kloppy: https://pypi.org/project/kloppy
- unravelsports: https://unravelsports.readthedocs.io

## Key method
- databallpy `get_individual_player_possession`: port of Vidal-Codina 2022; docs state "the ball is only gained when the ball changes direction or speed, to correct for balls that fly over players". Parameter names/values: not found in search text.
- databallpy team possession: computed from SYNCHRONISED EVENT DATA, not tracking. For Metrica tracking, ball_possession is set to None ("the data does not indicate which team has possession").
- databallpy docs: Tracab provides team possession in the feed, and it differs by definition: Tracab assigns possession to the receiving team after an unsuccessful pass, while the event-based rule switches only after a successful event by the other team.
- kloppy: `ball_owning_team` = team in possession at the time of the event/frame; filled from provider when present.
- unravelsports: if the dataset lacks ball-owning-team info, it is inferred by distance to ball, with a warning that this gives unexpected results when the ball is contested or in the air.

## Relevance to us
- Confirms that provider truths differ in switch timing (Tracab-style immediate vs event-style on control). Our truths: Metrica corrected tracking (control-based, loose ball stays) and IDSSE DFL flag (unknown rule). Switch timing should be tuned per truth, or at least measured per truth.
- The distance-only inference we use (team_control.py) is the same as unravelsports' fallback and has the known failure (contested / aerial balls).
