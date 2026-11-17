# SoccerCPD: Formation and Role Change-Point Detection in Soccer Matches Using Spatiotemporal Tracking Data

- **Authors:** H. Kim et al.
- **Year / venue:** 2022, KDD
- **URL:** https://arxiv.org/abs/2206.10926 · code https://github.com/pientist/soccercpd
- **Verified from:** search snippets.

## Key method
- Role-adjacency matrices from Delaunay triangulation (10 Hz), segmented with discrete g-segmentation, first by formation and then by role.
- **Set pieces break the formation**: players leave their roles, so the frame-level **role-switch rate goes towards 1.0**. A threshold of about **0.9** on that rate finds set pieces without supervision (from one search summary, not checked in the paper or code).

## Relevance to us
- A **ball-free** D cue for corners and wide free kicks, the long stoppages where the box fills up. It needs players without IDs only to the extent that it needs stable roles inside a team. Our challenge has no player IDs, so a direct port is **not possible**. A proxy that needs no IDs: sudden growth of team overlap (players of both teams mixed in the penalty area, "Max Players in the box" style).
