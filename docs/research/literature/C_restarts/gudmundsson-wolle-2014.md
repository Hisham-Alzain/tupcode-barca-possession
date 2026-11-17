# Football analysis using spatio-temporal tools

- **Authors:** J. Gudmundsson, T. Wolle
- **Year / venue:** 2014, Computers, Environment and Urban Systems 47, 16–27. Earlier version: "Towards Automated Football Analysis: Algorithms and Data Structures", 10th Australasian Conf. on Mathematics and Computers in Sport, 2010.
- **URL:** https://researchr.org/publication/GudmundssonW12 (bibliographic record)
- **Verified from:** search snippets. This paper is known only through Lang et al. (2022), which cites it.

## Key method
- Extracts "ball-in" and "ball-out" events from **ball and player** positions. Method details: **not found**.

## Reported results
- **F1 0.94** for ball-out detection, as cited by Lang et al. 2022. Not checked against the original.

## Relevance to us
- Shows that ball-out detection is easy when the ball is tracked all the time. Our problem is the 41% of frames without a ball, plus restarts with the ball still in the field (fouls, offside). Low priority. Only the upper bound is useful.
