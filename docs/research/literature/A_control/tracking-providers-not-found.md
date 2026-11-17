# Tracking providers: SkillCorner, Second Spectrum, TRACAB, Hawk-Eye - per-frame owning team

Searched; official definitions NOT found for any of them.

- SkillCorner: possessions are split into start and end moments (pressure metrics), "first touch to last" per player possession; phases of play (Build Up, Create, ...). No rule for how the possessing team per frame is set. https://www.skillcorner.com/articles/introducing-skillcorners-new-pressure-metrics
- Second Spectrum: optical tracking in MLS; no public possession definition found.
- TRACAB (ChyronHego): the feed carries a ball-owning-team flag and live/dead flag (databallpy: "Tracab provides info about which team has possession"); per databallpy's comparison Tracab switches to the receiving team after an unsuccessful pass, i.e. a faster/looser switch than event-based possession. Spec not found. TRACAB is also the position source of DFL/IDSSE.
- Hawk-Eye (SkeleTRACK): ball + 18-29 skeletal points; automated event detection and semi-automated offside; no possession definition found.
- DFL / Sportec BallPossession: see bassek-2025-idsse.md (rule not published).

## Relevance to us
- No provider publishes its rule, so the only way to learn each truth's switch convention is empirically from the bench: measure, per truth source, the lag between the first frame the new team is alone in the zone and the truth switch frame.
