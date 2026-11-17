# SoccerTrack v2: A Full-Pitch Panoramic Video Dataset for Game State Reconstruction and Ball Action Spotting

- **Authors:** not captured in search text (SoccerTrack team)
- **Year / venue:** 2025, arXiv 2508.01802
- **URL:** https://arxiv.org/abs/2508.01802
- **Verified from:** one search snippet.

## Key facts
- Has a **per-frame ball-status flag**. It is left out of the ball-action-spotting task because it **nearly gives away the out-of-play label**.

## Relevance to us
- Confirms that once ball status is known, most stoppage-related labels follow. It is the hard part, and we must infer it. No method to reuse. A possible extra source of truth D if we need more external data (not checked for licence or format).
