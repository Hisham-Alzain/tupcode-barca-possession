# An integrated dataset of spatiotemporal and event data in elite soccer (IDSSE)

- **Authors:** M. Bassek, R. Rein, H. Weber, D. Memmert (German Sport University Cologne)
- **Year / venue:** 2025, Scientific Data 12, article 195
- **URL / DOI:** https://doi.org/10.1038/s41597-025-04505-y · PMC11787359 · data https://doi.org/10.6084/m9.figshare.28196177 · https://github.com/spoho-datascience/idsse-data
- **Verified from:** search snippets.

## Key facts
- 7 matches (Bundesliga 1 and 2, 2022/23), player and ball x/y at **25 Hz** from TRACAB / Sportec Solutions, following the DFL official match-data catalogue. Synchronised events (passes, shots, fouls, set pieces) and match information. CC BY 4.0.
- Per ball frame: `BallStatus` (0 = inactive, 1 = active), `BallPossession` (1 = home, 2 = away), plus Z.
- **How BallStatus is produced: not found.** The paper says possession "is defined by the player in control of the ball", which is a high-level description and no algorithm. Lang et al. (2022) describe Bundesliga in-game status as **manually annotated**, which suggests (unverified) that a data-collection operator sets it, not the tracker. A secondary source (a GitHub issue from another project) says `BallPossession` **stays set during dead-ball periods** and is never empty on live frames.

## Relevance to us
- This is our IDSSE truth D. If BallStatus is set by an operator, D starts at the **whistle or the ball crossing a line** and ends at the **restart touch**. That is the timing our detector must copy.
- `BallPossession` during D is the team that **takes the restart** or the team before it. Which one it is matters for the H/A/D truth mapping in our bench. **Check in code** whether BallPossession flips during D before the restart. If it does, IDSSE gives us restart-team truth for free (to measure candidate C1).
- IDSSE events include set pieces with team: a **labelled set** to tune restart-team rules on.
