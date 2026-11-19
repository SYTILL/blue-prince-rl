# Minimal Blue Prince: room list (draft)

Source: the Blue Prince wiki (Room Directory, Room shape, Gem cost, and the pages by colour). As of 2026-10.

## Rules summary
- Rooms differ only in **shape (door directions), gem cost, rarity, and the steps, keys and gems they give**. There are no special effects.
- Resources are received **once, when the room is placed** (re-entering does not give them again). ← needs confirming
- Shape, rarity and gem cost are **as in the original game**. The resources are simplified from the original effects.
  - `original`: the original game gives almost the same amount.
  - `simplified`: the original is a puzzle reward, a random item, a next-day effect, a conditional effect, etc., so it was turned into a number. The values can be tuned.
  - `empty room`: the original effect is hard to turn into a number, so **only the shape is kept and the room gives nothing**.
- Negative = a loss. Some Red Rooms hurt the player in the original game, so they were turned into rooms that take steps away.

## Placement rules by shape
| Shape | Doors | Where it can be placed |
|---|---|---|
| Dead End | 1 | anywhere |
| Straight | 2, facing each other | not in a corner. On an edge only when entering along the edge |
| L-shaped (L) | 2, at a turn | each room has a **fixed turn direction (left/right)**. It cannot be placed if that direction faces outside (see below) |
| T-shaped (T) | 3 | not in a corner. On an edge, the closed side faces outside |
| 4-way | 4 | middle only |

### Orientation of L-shaped rooms
- Each room has a fixed **left turn** or **right turn**. Looking into the room from the door you entered through, it is left if the other door is on the left and right if it is on the right.
- So **once the entry door is known, an L-shaped room has exactly one placement** (no random rotation).
  - Example: moving north to open a room, the entry door is south (S). A left-turn room is S·W, a right-turn room is S·E.
- If the other door would face outside the house, that room is **not offered as a draft candidate** for that cell.
  - Entering along an edge: only rooms that turn inward are possible.
  - Entering straight into an edge: both left and right are possible.
  - Corner: only one side is possible.
- Once placed, it is simply a room with two doors. Both doors can be passed in both directions.

```python
LEFT  = {"N": "W", "W": "S", "S": "E", "E": "N"}   # move direction -> left
RIGHT = {v: k for k, v in LEFT.items()}            # move direction -> right
OPP   = {"N": "S", "S": "N", "E": "W", "W": "E"}

def l_room_doors(move_dir, turn):                  # turn: "L" or "R"
    return {OPP[move_dir], (LEFT if turn == "L" else RIGHT)[move_dir]}
```

## Room list (99)
The fixed rooms (start: Entrance Hall, goal: Antechamber) are defined separately and are not in the deck.

### Dead End (29)
| Room | Rarity | Gem cost | Steps | Keys | Gems | Basis |
|---|---|---|---|---|---|---|
| Closet | Commonplace | 0 | 0 | +1 | 0 | simplified (random item) |
| Storeroom | Commonplace | 0 | 0 | +1 | +1 | original |
| Guest Bedroom | Commonplace | 0 | +10 | 0 | 0 | original (+10 on first entry) |
| Nursery | Commonplace | 1 | +5 | 0 | 0 | simplified (+5 for each bedroom placed) |
| Walk-In Closet | Standard | 1 | 0 | +1 | +1 | simplified (items, one gem guaranteed) |
| Terrace | Standard | 0 | 0 | 0 | 0 | empty room |
| Greenhouse | Standard | 1 | 0 | 0 | +2 | simplified (0–3 gem flowers) |
| Lavatory | Standard | 0 | 0 | 0 | 0 | empty room (Red Room) |
| Utility Closet | Standard | 0 | 0 | 0 | 0 | empty room |
| Coat Check | Standard | 0 | 0 | 0 | 0 | empty room |
| Solarium | Standard | 1 | 0 | 0 | 0 | empty room |
| Dormitory | Standard | 0 | 0 | 0 | 0 | empty room |
| Planetarium | Standard | 0 | 0 | 0 | 0 | empty room |
| Garage | Unusual | 1 | 0 | +3 | 0 | original |
| Wine Cellar | Unusual | 0 | 0 | 0 | +3 | original |
| Study | Unusual | 0 | 0 | 0 | +1 | original (1 gem in the safe) |
| Servant's Quarters | Unusual | 1 | 0 | +2 | 0 | simplified (keys equal to the number of bedrooms) |
| Sauna | Unusual | 0 | 0 | 0 | 0 | empty room |
| Mail Room | Unusual | 0 | 0 | 0 | 0 | empty room |
| Bunk Room | Unusual | 0 | 0 | 0 | 0 | empty room |
| Locksmith | Unusual | 1 | 0 | 0 | 0 | empty room |
| Clock Tower | Unusual | 1 | 0 | 0 | 0 | empty room |
| Master Bedroom | Rare | 2 | +10 | 0 | 0 | simplified (+10 next day → immediately) |
| Attic | Rare | 3 | 0 | +2 | +2 | simplified (many items) |
| Vault | Rare | 3 | 0 | 0 | 0 | empty room |
| Freezer | Rare | 0 | 0 | 0 | 0 | empty room |
| Her Ladyship's Chamber | Rare | 0 | 0 | 0 | 0 | empty room |
| Laundry Room | Rare | 1 | 0 | 0 | 0 | empty room |
| Furnace | Rare | 0 | 0 | 0 | 0 | empty room (Red Room) |

### Straight (14)
| Room | Rarity | Gem cost | Steps | Keys | Gems | Basis |
|---|---|---|---|---|---|---|
| Spare Room | Commonplace | 0 | 0 | 0 | 0 | original (no effect) |
| Corridor | Commonplace | 0 | 0 | 0 | 0 | in the original, "doors are never locked" (optional rule) |
| Rumpus Room | Standard | 1 | 0 | 0 | 0 | empty room |
| The Kennel | Standard | 0 | 0 | 0 | 0 | empty room |
| Tunnel | Standard | 0 | 0 | 0 | 0 | empty room |
| Workshop | Unusual | 0 | 0 | 0 | 0 | empty room |
| Foyer | Unusual | 2 | 0 | 0 | 0 | in the original, "unlocks the doors of hallway rooms" (optional rule) |
| Ballroom | Unusual | 2 | 0 | 0 | 0 | empty room |
| Secret Passage | Unusual | 1 | 0 | 0 | 0 | empty room (the wiki pages disagree on its shape; follows the Room shape page) |
| Veranda | Unusual | 2 | 0 | 0 | 0 | empty room (in the original one door always faces outside; here a normal Straight) |
| Locker Room | Rare | 1 | 0 | +3 | 0 | simplified (scatters keys around the house) |
| Gallery | Rare | 0 | 0 | 0 | 0 | empty room |
| Drafting Studio | Rare | 2 | 0 | 0 | 0 | empty room |
| Showroom | Rare | 2 | 0 | 0 | 0 | empty room |

### L-shaped (27)
Direction: left = turns left from the entry door, right = turns right. Left and right alternate within each rarity (14 left, 13 right).

| Room | Direction | Rarity | Gem cost | Steps | Keys | Gems | Basis |
|---|---|---|---|---|---|---|---|
| Bedroom | left | Commonplace | 0 | +2 | 0 | 0 | original (the original gives +2 on every entry) |
| Parlor | right | Commonplace | 0 | 0 | 0 | +1 | simplified (puzzle reward of 2 gems) |
| Billiard Room | left | Commonplace | 0 | 0 | +1 | 0 | simplified (puzzle reward of 2 keys) |
| Nook | right | Commonplace | 0 | 0 | +1 | 0 | original |
| Pantry | left | Commonplace | 0 | +3 | 0 | 0 | simplified (fruit = steps) |
| Kitchen | right | Commonplace | 1 | 0 | 0 | 0 | empty room |
| Boudoir | left | Standard | 0 | 0 | 0 | +1 | original (1 gem in the safe) |
| Office | right | Standard | 2 | 0 | 0 | +1 | original (1 gem in the safe, coins removed) |
| Laboratory | left | Standard | 1 | 0 | 0 | 0 | empty room |
| Observatory | right | Standard | 1 | 0 | 0 | 0 | empty room |
| Patio | left | Standard | 1 | 0 | 0 | +1 | simplified (scatters gems in green rooms) |
| Commissary | right | Standard | 1 | 0 | 0 | 0 | empty room |
| The Armory | left | Standard | 0 | 0 | 0 | 0 | empty room |
| Library | right | Unusual | 0 | 0 | 0 | 0 | empty room |
| Music Room | left | Unusual | 2 | 0 | +2 | 0 | simplified (1 key + a special key) |
| Pump Room | right | Unusual | 0 | 0 | 0 | 0 | empty room |
| Maid's Chamber | left | Unusual | 0 | 0 | 0 | 0 | empty room (Red Room) |
| Dovecote | right | Unusual | 0 | 0 | 0 | 0 | empty room |
| Classroom | left | Unusual | 1 | 0 | 0 | 0 | empty room |
| Casino | right | Unusual | 2 | 0 | 0 | 0 | empty room |
| Conservatory | left | Unusual | 1 | 0 | 0 | 0 | empty room |
| Lost & Found | right | Unusual | 0 | 0 | 0 | 0 | empty room (Red Room) |
| Room 8 | left | Rare | 0 | 0 | 0 | 0 | empty room |
| Trophy Room | right | Rare | 5 | 0 | 0 | 0 | empty room |
| Morning Room | left | Rare | 0 | 0 | 0 | 0 | empty room |
| Bookshop | right | Rare | 1 | 0 | 0 | 0 | empty room |
| Treasure Trove | left | Rare* | 1 | 0 | 0 | 0 | empty room |

### T-shaped (20)
| Room | Rarity | Gem cost | Steps | Keys | Gems | Basis |
|---|---|---|---|---|---|---|
| Hallway | Commonplace | 0 | 0 | 0 | 0 | original (no effect) |
| Den | Commonplace | 0 | 0 | 0 | +1 | original |
| Drawing Room | Commonplace | 1 | 0 | 0 | +1 | original (1 gem in the safe) |
| Chapel | Commonplace | 0 | −2 | 0 | 0 | Red Room (the original takes coins) |
| West Wing Hall | Standard | 0 | 0 | 0 | 0 | original (no effect) |
| The Pool | Standard | 1 | 0 | 0 | 0 | empty room |
| Security | Standard | 1 | 0 | 0 | 0 | empty room |
| Dining Room | Standard | 0 | +10 | 0 | 0 | simplified (meal +20, conditional) |
| Courtyard | Standard | 1 | 0 | 0 | +1 | simplified (2 gem flowers) |
| Gymnasium | Standard | 0 | −2 | 0 | 0 | original (the original gives −2 on every entry) |
| Darkroom | Standard | 0 | −1 | 0 | 0 | Red Room (the original blocks vision) |
| East Wing Hall | Unusual | 0 | 0 | 0 | 0 | original (no effect) |
| Conference Room | Unusual | 0 | 0 | 0 | 0 | empty room |
| Boiler Room | Unusual | 1 | 0 | 0 | 0 | empty room |
| Aquarium | Unusual | 1 | 0 | 0 | 0 | empty room |
| The Foundation | Rare | 0 | 0 | 0 | 0 | empty room (a permanent room in the original) |
| Secret Garden | Rare | 0 | 0 | 0 | 0 | empty room |
| Gift Shop | Rare | 0 | 0 | 0 | 0 | empty room |
| Closed Exhibit | Rare | 0 | 0 | 0 | 0 | empty room (Red Room) |
| Throne Room | Rare* | 5 | 0 | 0 | 0 | empty room |

### 4-way (9)
| Room | Rarity | Gem cost | Steps | Keys | Gems | Basis |
|---|---|---|---|---|---|---|
| Passageway | Commonplace | 2 | 0 | 0 | 0 | original (its strength is having many doors) |
| Great Hall | Unusual | 0 | 0 | 0 | 0 | in the original, 7 doors are locked (optional rule) |
| Archives | Unusual | 0 | −1 | 0 | 0 | Red Room |
| Cloister | Unusual | 3 | 0 | 0 | +2 | simplified (2 gem flowers) |
| Vestibule | Unusual | 2 | 0 | 0 | 0 | empty room |
| Mechanarium | Unusual | 0 | 0 | 0 | 0 | empty room |
| Rotunda | Rare | 3 | 0 | 0 | 0 | empty room |
| Weight Room | Rare | 0 | −10 | 0 | 0 | simplified (the original halves steps) |
| Chamber of Mirrors | Rare | 0 | 0 | 0 | 0 | empty room (the wiki pages disagree on its shape; follows the Room shape page) |

\* Treasure Trove, Throne Room: the wiki gives no rarity, so Rare is assumed.

## Rooms with rewards (by reward, 27)
Reward total = steps, keys and gems simply added up. Ties are ordered by lower gem cost. Red Rooms that take steps away are excluded.

| Rank | Room | Shape | Rarity | Gem cost | Reward | Total |
|---|---|---|---|---|---|---|
| 1 | Guest Bedroom | Dead End | Commonplace | 0 | steps +10 | 10 |
| 1 | Dining Room | T-shaped | Standard | 0 | steps +10 | 10 |
| 1 | Master Bedroom | Dead End | Rare | 2 | steps +10 | 10 |
| 4 | Nursery | Dead End | Commonplace | 1 | steps +5 | 5 |
| 5 | Attic | Dead End | Rare | 3 | keys +2, gems +2 | 4 |
| 6 | Pantry | L-shaped (left) | Commonplace | 0 | steps +3 | 3 |
| 6 | Wine Cellar | Dead End | Unusual | 0 | gems +3 | 3 |
| 6 | Garage | Dead End | Unusual | 1 | keys +3 | 3 |
| 6 | Locker Room | Straight | Rare | 1 | keys +3 | 3 |
| 10 | Bedroom | L-shaped (left) | Commonplace | 0 | steps +2 | 2 |
| 10 | Storeroom | Dead End | Commonplace | 0 | keys +1, gems +1 | 2 |
| 10 | Walk-In Closet | Dead End | Standard | 1 | keys +1, gems +1 | 2 |
| 10 | Greenhouse | Dead End | Standard | 1 | gems +2 | 2 |
| 10 | Servant's Quarters | Dead End | Unusual | 1 | keys +2 | 2 |
| 10 | Music Room | L-shaped (left) | Unusual | 2 | keys +2 | 2 |
| 10 | Cloister | 4-way | Unusual | 3 | gems +2 | 2 |
| 17 | Closet | Dead End | Commonplace | 0 | keys +1 | 1 |
| 17 | Billiard Room | L-shaped (left) | Commonplace | 0 | keys +1 | 1 |
| 17 | Nook | L-shaped (right) | Commonplace | 0 | keys +1 | 1 |
| 17 | Parlor | L-shaped (right) | Commonplace | 0 | gems +1 | 1 |
| 17 | Den | T-shaped | Commonplace | 0 | gems +1 | 1 |
| 17 | Boudoir | L-shaped (left) | Standard | 0 | gems +1 | 1 |
| 17 | Study | Dead End | Unusual | 0 | gems +1 | 1 |
| 17 | Drawing Room | T-shaped | Commonplace | 1 | gems +1 | 1 |
| 17 | Patio | L-shaped (left) | Standard | 1 | gems +1 | 1 |
| 17 | Courtyard | T-shaped | Standard | 1 | gems +1 | 1 |
| 17 | Office | L-shaped (right) | Standard | 2 | gems +1 | 1 |

## Goal room (Antechamber)
- Fixed at the top middle (row 9, column C). It has 3 doors: south, west and east.
- It cannot be opened from outside; a path exists only once a new room's door connects to one of the Antechamber's doors.
- **Entering costs 3 keys.** (confirmed)

## Rooms not in the list
| Room | Reason |
|---|---|
| Entrance Hall, Antechamber | fixed separately as the start and goal rooms |
| Room 46 | no shape information (a special goal room in the original too) |
| Outer rooms (Toolshed, Shelter and 6 others) | rooms outside the house grid |
| Upgraded variants (Parlor upgrade, Goldfish Aquarium, etc.) | variants of the same base room |

## Observations
- About **60 of the 99 rooms are empty rooms**. Most of the deck becomes connecting rooms that differ only in "shape + gem cost", so the rooms that give resources (about 30) are relatively rare. This actually makes the choice "go for a reward room, or widen the path" clearer.
- For the AI, the rooms that actually differ are about 30 combinations of (shape, L direction, gem cost, reward). The names are just for flavour.
- Empty rooms with a high gem cost (Trophy Room 5, Throne Room 5, Vault 3, Rotunda 3) are almost never worth picking. Because of the free-first-slot rule, they are sometimes used only when free.
- Among the Red Rooms, Lavatory, Maid's Chamber, Lost & Found, Closed Exhibit and Furnace were kept as empty rooms, and Chapel, Gymnasium, Darkroom, Archives and Weight Room as rooms that cost steps.

## To decide
1. Whether resources are received once on placement, or, as in the original, some rooms (Bedroom, Gymnasium) give them on every entry.
2. ~~How many copies of each room go in the deck~~ → **Confirmed: every room is in the deck exactly once.** A placed room leaves the deck for the rest of the game (99-card deck).
3. Draw chance by rarity (the wiki gives no numbers; e.g. Commonplace 50 / Standard 30 / Unusual 15 / Rare 5, rarer rooms more likely in higher rows).
4. Whether to include the optional rules (Corridor doors never locked, Foyer, Great Hall locks).
5. Whether to lower the cost of the expensive empty rooms (Trophy Room, Throne Room, Vault, Rotunda) and of rooms that are only expensive (Passageway, Foyer).
6. Whether the empty Red Rooms (Lavatory and 4 others) should also cost steps like the other Red Rooms.
