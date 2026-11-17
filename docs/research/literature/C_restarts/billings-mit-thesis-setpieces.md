# Using Machine Learning to Differentiate Set Pieces in Football via Tracking Data

- **Author:** J. A. Billings (MIT thesis; also co-author of Vidal-Codina et al. 2022)
- **Year / venue:** MIT DSpace, year not captured (about 2024)
- **URL:** https://dspace.mit.edu/handle/1721.1/156805
- **Verified from:** catalogue abstract only. Results and features **not found**.

## Key method
- A learning approach that uses differences in **ball and player motion** to tell which set piece ends a dead-ball interval (corner, free kick, throw-in). This is the learned counterpart of the Vidal-Codina triggers.

## Related: Classifying Corner Kicks in Football (Aalborg Univ. MSc, 2024)
- https://projekter.aau.dk/projekter/files/715590408/Classifying_Corner_Kicks_in_Football.pdf
- 2,132 corners at 25 fps. Six corner types; MLP / SVM / RF / XGBoost compared by F1. It classifies the type of corner, not whether a stoppage happened. Not relevant beyond confirming that the corner-arc geometry is easy to find.

## Relevance to us
- Says the stoppage type, and so the restart team, can be read from motion during D. We have no results to anchor a prediction. Treat as support for C1/C2 in survey.md, not as a recipe.
