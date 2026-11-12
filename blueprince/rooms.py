"""Room data for minimal Blue Prince (99 rooms, one copy each).

Ported 1:1 from web/drafting-manor.html; see rooms.md for the sources.
"""
from typing import NamedTuple, Optional

SHAPES = "DSLTX"      # Dead end, Straight, L-shape, T-shape, 4-way (X)
RARITIES = "CSUR"     # Commonplace, Standard, Unusual, Rare


class Room(NamedTuple):
    name: str
    shape: str              # one of SHAPES
    turn: Optional[str]     # "L" / "R" for L-shape rooms, else None
    rarity: str             # one of RARITIES
    cost: int               # gem cost
    steps: int              # granted once, when the room is placed
    keys: int
    gems: int
    red: bool


def R(name, shape, rarity, cost, steps=0, keys=0, gems=0, red=0):
    turn = {"l": "L", "r": "R"}.get(shape[1:]) if len(shape) > 1 else None
    return Room(name, shape[0], turn, rarity, cost, steps, keys, gems, bool(red))


ROOMS = (
    # Dead ends (29)
    R("Closet", "D", "C", 0, 0, 1, 0), R("Storeroom", "D", "C", 0, 0, 1, 1), R("Guest Bedroom", "D", "C", 0, 10), R("Nursery", "D", "C", 1, 5),
    R("Walk-In Closet", "D", "S", 1, 0, 1, 1), R("Terrace", "D", "S", 0), R("Greenhouse", "D", "S", 1, 0, 0, 2), R("Lavatory", "D", "S", 0, 0, 0, 0, 1),
    R("Utility Closet", "D", "S", 0), R("Coat Check", "D", "S", 0), R("Solarium", "D", "S", 1), R("Dormitory", "D", "S", 0), R("Planetarium", "D", "S", 0),
    R("Garage", "D", "U", 1, 0, 3), R("Wine Cellar", "D", "U", 0, 0, 0, 3), R("Study", "D", "U", 0, 0, 0, 1), R("Servant's Quarters", "D", "U", 1, 0, 2),
    R("Sauna", "D", "U", 0), R("Mail Room", "D", "U", 0), R("Bunk Room", "D", "U", 0), R("Locksmith", "D", "U", 1), R("Clock Tower", "D", "U", 1),
    R("Master Bedroom", "D", "R", 2, 10), R("Attic", "D", "R", 3, 0, 2, 2), R("Vault", "D", "R", 3), R("Freezer", "D", "R", 0),
    R("Her Ladyship's Chamber", "D", "R", 0), R("Laundry Room", "D", "R", 1), R("Furnace", "D", "R", 0, 0, 0, 0, 1),
    # Straight (14)
    R("Spare Room", "S", "C", 0), R("Corridor", "S", "C", 0), R("Rumpus Room", "S", "S", 1), R("The Kennel", "S", "S", 0), R("Tunnel", "S", "S", 0),
    R("Workshop", "S", "U", 0), R("Foyer", "S", "U", 2), R("Ballroom", "S", "U", 2), R("Secret Passage", "S", "U", 1), R("Veranda", "S", "U", 2),
    R("Locker Room", "S", "R", 1, 0, 3), R("Gallery", "S", "R", 0), R("Drafting Studio", "S", "R", 2), R("Showroom", "S", "R", 2),
    # L-shape (27): l = turns left, r = turns right
    R("Bedroom", "Ll", "C", 0, 2), R("Parlor", "Lr", "C", 0, 0, 0, 1), R("Billiard Room", "Ll", "C", 0, 0, 1), R("Nook", "Lr", "C", 0, 0, 1),
    R("Pantry", "Ll", "C", 0, 3), R("Kitchen", "Lr", "C", 1),
    R("Boudoir", "Ll", "S", 0, 0, 0, 1), R("Office", "Lr", "S", 2, 0, 0, 1), R("Laboratory", "Ll", "S", 1), R("Observatory", "Lr", "S", 1),
    R("Patio", "Ll", "S", 1, 0, 0, 1), R("Commissary", "Lr", "S", 1), R("The Armory", "Ll", "S", 0),
    R("Library", "Lr", "U", 0), R("Music Room", "Ll", "U", 2, 0, 2), R("Pump Room", "Lr", "U", 0), R("Maid's Chamber", "Ll", "U", 0, 0, 0, 0, 1),
    R("Dovecote", "Lr", "U", 0), R("Classroom", "Ll", "U", 1), R("Casino", "Lr", "U", 2), R("Conservatory", "Ll", "U", 1), R("Lost & Found", "Lr", "U", 0, 0, 0, 0, 1),
    R("Room 8", "Ll", "R", 0), R("Trophy Room", "Lr", "R", 5), R("Morning Room", "Ll", "R", 0), R("Bookshop", "Lr", "R", 1), R("Treasure Trove", "Ll", "R", 1),
    # T-shape (20)
    R("Hallway", "T", "C", 0), R("Den", "T", "C", 0, 0, 0, 1), R("Drawing Room", "T", "C", 1, 0, 0, 1), R("Chapel", "T", "C", 0, -2, 0, 0, 1),
    R("West Wing Hall", "T", "S", 0), R("The Pool", "T", "S", 1), R("Security", "T", "S", 1), R("Dining Room", "T", "S", 0, 10),
    R("Courtyard", "T", "S", 1, 0, 0, 1), R("Gymnasium", "T", "S", 0, -2, 0, 0, 1), R("Darkroom", "T", "S", 0, -1, 0, 0, 1),
    R("East Wing Hall", "T", "U", 0), R("Conference Room", "T", "U", 0), R("Boiler Room", "T", "U", 1), R("Aquarium", "T", "U", 1),
    R("The Foundation", "T", "R", 0), R("Secret Garden", "T", "R", 0), R("Gift Shop", "T", "R", 0), R("Closed Exhibit", "T", "R", 0, 0, 0, 0, 1), R("Throne Room", "T", "R", 5),
    # 4-way (9)
    R("Passageway", "X", "C", 2), R("Great Hall", "X", "U", 0), R("Archives", "X", "U", 0, -1, 0, 0, 1), R("Cloister", "X", "U", 3, 0, 0, 2),
    R("Vestibule", "X", "U", 2), R("Mechanarium", "X", "U", 0), R("Rotunda", "X", "R", 3), R("Weight Room", "X", "R", 0, -10, 0, 0, 1), R("Chamber of Mirrors", "X", "R", 0),
)

SHAPE_COUNTS = {s: sum(1 for rm in ROOMS if rm.shape == s) for s in SHAPES}
