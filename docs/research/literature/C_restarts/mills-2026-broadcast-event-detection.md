# Automatic event detection in association football using broadcast-derived tracking data

- **Authors:** K. Mills, H. Wang, Z. Crang, J. Billingham, G. Duthie, R. Johnson, S. Robertson
- **Year / venue:** 2026, Sports Engineering 29 (article 15)
- **URL / DOI:** https://doi.org/10.1007/s12283-026-00549-4 · open PDF https://shura.shu.ac.uk/37410/
- **Verified from:** search snippets only.

## Key method
- Runs the Vidal-Codina et al. (2022) algorithm with its **default possession and set-piece hyperparameters** on broadcast-derived tracking from three commercial providers (Camera 1, Tactical, PGM). Data: one 90-minute 2022 World Cup match.
- A set piece is detected when the ball status switches dead -> live, and its type is set by trigger and pattern criteria.
- A detected event matches the reference when it has the same type and lies within **3 s** of it. Score: F1 per event type.

## Reported results
- Throw-in F1: best **0.87** (Camera 1 and Tactical), 0.60 (PGM), against **0.97** for multi-camera tracking. Only 31 throw-ins in the reference set, so these numbers are noisy.
- Most errors happen around set pieces, from player visibility and ball-tracking errors.
- **Ball status switching to live too early caused the wrong set piece to be detected**, because the events before it were then in the wrong order.

## Relevance to us
- Closest published setting to ours (imperfect, non-optical-grade tracking). It shows that restart typing holds up (F1 0.87) once the dead/live flag is right. Most of the damage comes from **live-switch timing**.
- So the END frame of a D stretch should be set conservatively: wait for the executor's first kick or throw, and do not switch to live at the first ball movement.
