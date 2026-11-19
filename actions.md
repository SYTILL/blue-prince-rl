# PPO action definition

## At a glance
- Action space: **`Discrete(183)`**. The network output has a fixed size of 183, and actions that are not possible right now are blocked with a **mask** (sb3-contrib `MaskablePPO`).
- One game turn is **two** agent actions: ① pick a door → ② pick a blueprint.
- Walking and entering the Antechamber are not actions; they happen **automatically**.

| Number | Meaning | Phase |
|---|---|---|
| 0 – 179 | Open a door: one (cell, direction) | door phase |
| 180, 181, 182 | Pick the 1st, 2nd or 3rd drawn blueprint | draft phase |

## 1. Open a door (0–179)
```
index = ((r - 1) * 5 + (c - 1)) * 4 + d
r = row 1..9 (1 is the bottom), c = column 1..5 (A..E), d = N 0, E 1, S 2, W 3
```
Meaning: "open the d-side door of the room at cell (r, c)."

Example: the starting room, Entrance Hall, is C1 = (1, 3).
- North door = 8
- East door = 9
- West door = 11

**Legal when (mask = 1)**: all of the following hold.
- It is the door phase
- There is a room at (r, c), and that room has a door on side d
- The cell beyond the door is inside the grid and empty
- That room is not the Antechamber (it cannot be opened from outside)
- There is a path from the player's position to that room (BFS)
- Steps ≥ distance + 1 (walking there + entering the new room)
- If the door is locked, keys ≥ 1

**When taken**, it is processed in this order.
1. The player walks there along the shortest path. Steps go down by the distance.
2. If the door is locked, one key is spent.
3. Three blueprints are drawn and the game moves to the draft phase.
   - If the deck has no room that can be placed in that cell, the door becomes a dead door and the game stays in the door phase.

## 2. Pick a blueprint (180–182)
Meaning: place the `action − 180`-th drawn blueprint in that cell.

**Legal when**: all of the following hold.
- It is the draft phase
- There is a blueprint in that slot (fewer than 3 may be drawn)
- Gems ≥ the cost of that blueprint (the first slot costs 0)

**When taken**, it is processed in this order.
1. Pay the gems and remove the room from the deck.
2. Place the room. Its orientation is already fixed when it is drawn (L-shaped rooms have a fixed left/right turn, T-shaped rooms are random). That is why the door directions are part of the blueprint features in the observation.
3. Decide the locks of the new doors. A door that meets an existing door is open.
4. Enter (steps −1). Then receive the room's reward (steps, keys, gems).
5. If this is a row reached for the first time, give the milestone reward.
6. Run the automatic entry check and the end check, then go back to the door phase.

## 3. Handled automatically (not actions)
- **Movement**: picking a door walks to it automatically along the shortest path. There is no action for just walking around, because walking alone only costs steps.
- **Entering the Antechamber**: if a path is connected, steps ≥ distance and keys ≥ 3, the player enters at once and wins (+10). Entering is always the best choice, so there is no reason to make it an option.
  - In the web version a person clicks it, but the result is the same.

## 4. Ending
- **Win**: automatic Antechamber entry. Reward +10.
- **Loss**: in the door phase the mask is all 0 (out of steps, not enough keys, no door to open).
  - The cause of the loss is recorded in `info` for evaluation.
- **Truncation**: cut off at 300 actions as a safeguard. Normally the 99-card deck ends the game before that.

## Why it was designed this way
- **Fixed size + mask**: the network output size has to be fixed. Usually only 2–10 actions are actually possible, but keeping all 183 and masking the rest is the standard approach.
- **Numbering doors by cell + direction**: the same door always has the same number in every game, so the network can easily learn position rules such as "the top middle door is good".
- **Two phases**: choosing a "door × blueprint" combination in one step would mean choosing before seeing the blueprints, which does not match the original rules.
  - So the observation includes a **phase flag** (door phase / draft phase).
  - The number of actions per game doubles, so the discount factor γ is set to 0.99 or higher.

## Still to decide
- The value of the discount factor γ (0.99 vs 1.0).
- Whether to give a small penalty when a dead door appears (no room in the deck can be placed).
