# IFAB Laws of the Game: Laws 8, 9, 13, 15, 16, 17 (restart geometry)

- **Author / venue:** The IFAB, Laws of the Game (yearly, 2024/25 to 2026/27). https://www.theifab.com · https://downloads.theifab.com/downloads/laws-of-the-game-2024-25?l=en · FA mirror https://www.thefa.com/football-rules-governance/lawsandrules/laws/football-11-11
- **Verified from search text:** Law 9 wording; Law 8 kick-off after a goal; Law 15 throw-in to the opponents of the last toucher; the 9.15 m corner mark. Parts marked (memory) are standard Law text I know but could **not** confirm from search snippets.

## Rules with geometry
| Law | Rule | Geometric cue in tracking |
|---|---|---|
| 9 | Out of play when the **whole ball** has passed over the goal line or touchline, on the ground or in the air, or the referee stops play. | Ball centre beyond the line by > 0.11 m (ball radius) plus tracking noise. Referee stops (fouls, offside, injury) have **no line cue**. |
| 8 | Kick-off: start of each half and **after a goal, taken by the team that conceded**. All players in **their own half**; opponents at least 9.15 m away (centre circle) (memory). | Ball at the centre spot, both teams split by the halfway line. Restart team after a goal = the team that conceded. |
| 15 | Throw-in to the **opponents of the player who last touched** the ball. Thrower on or behind the touchline. Opponents at least 2 m from the point (memory). | One player at \|y\| near or beyond the touchline next to the ball. Their team is the restart team. |
| 16 | Goal kick when the ball crosses the goal line (not a goal) and an **attacker** touched it last. Kicked from anywhere in the **goal area** (memory). Opponents outside the penalty area until the ball is in play (memory). | Ball still in the goal area (5.5 m box). **Restart team = the team defending that goal.** Opponents out of the box. |
| 17 | Corner kick when the ball crosses the goal line and a **defender** touched it last. Ball in the corner area (1 m arc) (memory). Opponents at least 9.15 m (corner mark). | Ball within about 1 m of a corner flag. **Restart team = the team attacking that goal.** |
| 13 | Free kick: opponents at least 9.15 m away (memory). | One team inside a 9.15 m ring around a still ball, the other outside. Restart team = the team inside. |

## Relevance to us
- For corners, goal kicks and kick-offs, the **restart team follows from where the ball is plus which goal each team defends** (we already estimate each team's side per half). No last touch needed.
- For throw-ins and free kicks, the **exclusion zone** (opponents at 2 m or 9.15 m) means the team with players near the still ball is the restarting team. This rule needs no player IDs.
- After a goal, the restart team is always the team that conceded, so the possession flips to them.
