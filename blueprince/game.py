"""Rules engine for minimal Blue Prince. Pure Python, no Gymnasium dependency.

The rules are a port of web/drafting-manor.html, with three deliberate differences:

1. Randomness is keyed by position instead of drawn from one stream.
   A door's lock depends only on (seed, edge). A draft ranks the deck by numbers
   that depend only on (seed, cell, room). So two agents playing the same seed
   see the same locks, and the same offers at a cell unless one of them already
   used an offered room or enters the cell from a different side.
2. Entering the Antechamber is automatic (see actions.md section 3).
3. The Antechamber cannot be walked through; it only ends the game.

Coordinates: cell = (r, c), r = 1..9 (1 is the bottom rank), c = 1..5 (A..E).
Directions: N=0, E=1, S=2, W=3. Action numbering follows actions.md.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import NamedTuple, Optional

from .rooms import RARITIES, ROOMS

ROWS, COLS = 9, 5
N, E, S, W = 0, 1, 2, 3
DR = (1, 0, -1, 0)
DC = (0, 1, 0, -1)
DIR_NAMES = "NESW"
START = (1, 3)
GOAL = (9, 3)

N_DOOR_ACTIONS = ROWS * COLS * 4      # 180
N_OFFERS = 3
N_ACTIONS = N_DOOR_ACTIONS + N_OFFERS  # 183
PHASE_DOOR, PHASE_DRAFT = 0, 1

LOSS_CAUSES = ("out_of_steps", "out_of_keys", "goal_keys", "no_doors", "truncated")


@dataclass(frozen=True)
class Config:
    steps: int = 50
    keys: int = 2
    gems: int = 2
    rarity_weights: tuple = (50, 30, 15, 5)          # C, S, U, R
    # chance that a new door is locked, by the higher rank of its two cells (index = rank)
    lock_prob: tuple = (0, 0, 0, 0, 0.25, 0.45, 0.6, 0.7, 0.9, 1.0)
    goal_keys: int = 3                                # keys spent to enter the Antechamber
    win_reward: float = 10.0
    milestone_scale: float = 1 / 256                  # first time at rank k: 2**(k-1) * scale
    max_actions: int = 300                            # truncation safeguard


class Offer(NamedTuple):
    room: int        # index into ROOMS
    doors: tuple     # door directions after orientation
    cost: int        # gems to pay (slot 0 is always free)


class Placed(NamedTuple):
    name: str
    doors: tuple
    room: int        # index into ROOMS, or -1 for the two fixed rooms


class Draft(NamedTuple):
    origin: tuple    # cell whose door was opened
    d: int           # direction of that door
    target: tuple    # cell being drafted
    offers: tuple


# ---------------------------------------------------------------- helpers

def in_grid(r, c):
    return 1 <= r <= ROWS and 1 <= c <= COLS


def cell_id(r, c):
    return (r - 1) * COLS + (c - 1)


def edge_id(r, c, d):
    a, b = cell_id(r, c), cell_id(r + DR[d], c + DC[d])
    return (a, b) if a < b else (b, a)


def encode_door(r, c, d):
    return cell_id(r, c) * 4 + d


def decode_door(action):
    cid, d = divmod(action, 4)
    return cid // COLS + 1, cid % COLS + 1, d


_M = (1 << 64) - 1
_GAMMA = 0x9E3779B97F4A7C15


def _splitmix(z):
    z = (z + _GAMMA) & _M
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _M
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _M
    return z ^ (z >> 31)


class KeyedRng:
    """Small deterministic RNG whose stream depends only on the given integer key."""

    def __init__(self, *key):
        h = 0
        for v in key:
            h = _splitmix(h ^ (v & _M))
        self.state = h

    def random(self):
        self.state = (self.state + _GAMMA) & _M
        return (_splitmix(self.state) >> 11) / (1 << 53)

    def choice(self, seq):
        return seq[int(self.random() * len(seq))]

    def weighted(self, items, weights):
        x = self.random() * sum(weights)
        for item, w in zip(items, weights):
            x -= w
            if x < 0:
                return item
        return items[-1]


def orientations(room, move):
    """Door sets a room can take when entered by moving in direction `move`."""
    entry = (move + 2) % 4
    if room.shape == "D":
        return [(entry,)]
    if room.shape == "S":
        return [(entry, move)]
    if room.shape == "X":
        return [(N, E, S, W)]
    if room.shape == "L":
        return [(entry, (move + 3) % 4 if room.turn == "L" else (move + 1) % 4)]
    # T-shape: the wall can be on any side except the entry side
    return [tuple(d for d in (N, E, S, W) if d != wall) for wall in (N, E, S, W) if wall != entry]


def valid_orientations(room, r, c, move):
    """Orientations in which no door leads outside the house."""
    return [doors for doors in orientations(room, move)
            if all(in_grid(r + DR[d], c + DC[d]) for d in doors)]


# ---------------------------------------------------------------- game

class Game:
    def __init__(self, seed: int, config: Optional[Config] = None):
        self.seed = int(seed)
        self.cfg = cfg = config or Config()
        self.rooms = {
            START: Placed("Entrance Hall", (W, N, E), -1),
            GOAL: Placed("Antechamber", (W, S, E), -1),
        }
        self.locks = {}          # edge -> bool, current state (a used key sets it to False)
        self.lock_rolls = {}     # edge -> bool, as first rolled
        self.dead = set()        # edges whose door turned out to be blocked (no room fits)
        self.deck = set(range(len(ROOMS)))
        self.player = START
        self.steps, self.keys, self.gems = cfg.steps, cfg.keys, cfg.gems
        self.max_rank = 1
        self.phase = PHASE_DOOR
        self.draft: Optional[Draft] = None
        self.over = False
        self.win = False
        self.truncated = False
        self.loss_cause: Optional[str] = None
        self.total_reward = 0.0
        self.actions = []        # action history; (seed, config, actions) replays the game
        self.steps_spent = self.keys_used = self.gems_used = self.drafted = 0
        self._legal = None
        self._roll_locks(START)
        self._settle()

    # ---- queries

    def connected(self, cell, d):
        """True if the door d of the room at `cell` joins a door of the neighbouring room."""
        other = self.rooms.get((cell[0] + DR[d], cell[1] + DC[d]))
        return other is not None and (d + 2) % 4 in other.doors

    def bfs(self, start=None):
        """Shortest walking distances from `start` (default: the player). Returns (dist, prev)."""
        rooms = self.rooms
        start = self.player if start is None else start
        dist, prev = {start: 0}, {}
        queue = deque([start])
        while queue:
            cur = queue.popleft()
            if cur == GOAL:
                continue     # the Antechamber is never walked through
            for d in rooms[cur].doors:
                nb = (cur[0] + DR[d], cur[1] + DC[d])
                other = rooms.get(nb)
                if other is None or nb in dist or (d + 2) % 4 not in other.doors:
                    continue
                dist[nb] = dist[cur] + 1
                prev[nb] = cur
                queue.append(nb)
        return dist, prev

    def frontier(self):
        """Unopened doors as (r, c, d, edge, locked). Antechamber doors are excluded."""
        out = []
        for (r, c), room in self.rooms.items():
            if (r, c) == GOAL:
                continue
            for d in room.doors:
                nr, nc = r + DR[d], c + DC[d]
                if not in_grid(nr, nc) or (nr, nc) in self.rooms:
                    continue
                edge = edge_id(r, c, d)
                if edge not in self.dead:
                    out.append((r, c, d, edge, self.locks[edge]))
        return out

    def legal_actions(self):
        if self._legal is None:
            self._legal = self._compute_legal()
        return self._legal

    def action_mask(self):
        mask = [False] * N_ACTIONS
        for a in self.legal_actions():
            mask[a] = True
        return mask

    def _compute_legal(self, dist=None):
        if self.over:
            return []
        if self.phase == PHASE_DRAFT:
            return [N_DOOR_ACTIONS + i for i, o in enumerate(self.draft.offers) if self.gems >= o.cost]
        if dist is None:
            dist, _ = self.bfs()
        legal = []
        for r, c, d, _, locked in self.frontier():
            walk = dist.get((r, c))
            if walk is None or self.steps < walk + 1 or (locked and self.keys < 1):
                continue
            legal.append(encode_door(r, c, d))
        return legal

    # ---- transitions

    def step(self, action):
        """Apply one action and return the reward it earned."""
        action = int(action)
        if self.over or self.truncated:
            raise RuntimeError("the game is already over")
        if action not in self.legal_actions():
            raise ValueError(f"illegal action {action}")
        before = self.total_reward
        self.actions.append(action)
        if action < N_DOOR_ACTIONS:
            self._open_door(action)
        else:
            self._choose_offer(action - N_DOOR_ACTIONS)
        self._settle()
        if not self.over and len(self.actions) >= self.cfg.max_actions:
            self.truncated = True
            self.loss_cause = "truncated"
        return self.total_reward - before

    def _enter(self, cell):
        self.player = cell
        while self.max_rank < cell[0]:
            self.max_rank += 1
            self.total_reward += 2 ** (self.max_rank - 1) * self.cfg.milestone_scale

    def _walk(self, target, prev):
        path = []
        cur = target
        while cur != self.player:
            path.append(cur)
            cur = prev[cur]
        for cell in reversed(path):
            self._enter(cell)
        self.steps -= len(path)
        self.steps_spent += len(path)

    def _roll_locks(self, cell):
        r, c = cell
        for d in self.rooms[cell].doors:
            nr, nc = r + DR[d], c + DC[d]
            if not in_grid(nr, nc) or (nr, nc) in self.rooms:
                continue
            edge = edge_id(r, c, d)
            if edge not in self.locks:
                roll = KeyedRng(self.seed, 1, edge[0], edge[1]).random() < self.cfg.lock_prob[max(r, nr)]
                self.locks[edge] = self.lock_rolls[edge] = roll

    def _open_door(self, action):
        r, c, d = decode_door(action)
        _, prev = self.bfs()
        self._walk((r, c), prev)
        edge = edge_id(r, c, d)
        if self.locks[edge]:
            self.keys -= 1
            self.keys_used += 1
            self.locks[edge] = False
        target = (r + DR[d], c + DC[d])
        offers = self._draw_offers(target, d)
        if not offers:           # nothing in the deck fits: the door is blocked for good
            self.dead.add(edge)
            return
        self.draft = Draft((r, c), d, target, offers)
        self.phase = PHASE_DRAFT

    def _draw_offers(self, target, move):
        r, c = target
        cands = []
        for idx in sorted(self.deck):
            ors = valid_orientations(ROOMS[idx], r, c, move)
            if ors:
                cands.append((idx, ors))
        if not cands:
            return ()
        # Every room gets a fixed random number per (seed, cell). Offers are the
        # best-ranked rooms, so removing an unrelated room from the deck leaves
        # the offers at this cell unchanged.
        base = KeyedRng(self.seed, 2, cell_id(r, c)).state
        # Slot 0 and the other slots use independent numbers, as two separate draws would.
        def numbers(salt):
            return {i: ((_splitmix(salt ^ (i + 1)) >> 11) + 1) / (1 << 53) for i, _ in cands}
        u_free, u_rest, turn = numbers(base), numbers(_splitmix(base)), _splitmix(base ^ _M)
        weights = dict(zip(RARITIES, self.cfg.rarity_weights))

        def ranked(pool, u):
            # Weighted sampling without replacement (Efraimidis-Spirakis). A room's
            # weight is its rarity weight shared among the pool's rooms of that
            # rarity: "pick a rarity by weight, then a room of that rarity".
            count = {}
            for i, _ in pool:
                count[ROOMS[i].rarity] = count.get(ROOMS[i].rarity, 0) + 1
            def priority(x):
                rarity = ROOMS[x[0]].rarity
                w = weights[rarity] / count[rarity]
                return math.log(u[x[0]]) / w if w > 0 else float("-inf")
            return sorted(pool, key=priority, reverse=True)

        free = [x for x in cands if ROOMS[x[0]].cost == 0]
        picked = ranked(free or cands, u_free)[:1]           # slot 0 is a free room when possible
        rest = ranked([x for x in cands if x not in picked], u_rest)
        picked += rest[:N_OFFERS - 1]
        if len(picked) == N_OFFERS and all(ROOMS[i].shape == "D" for i, _ in picked):
            alt = [x for x in rest[N_OFFERS - 1:] if ROOMS[x[0]].shape != "D"]
            if alt:                                    # never three dead ends
                picked[-1] = alt[0]
        return tuple(Offer(i, ors[_splitmix(turn ^ (i + 1)) % len(ors)], 0 if slot == 0 else ROOMS[i].cost)
                     for slot, (i, ors) in enumerate(picked))

    def _choose_offer(self, slot):
        offer = self.draft.offers[slot]
        room = ROOMS[offer.room]
        target = self.draft.target
        self.gems -= offer.cost
        self.gems_used += offer.cost
        self.deck.discard(offer.room)
        self.rooms[target] = Placed(room.name, offer.doors, offer.room)
        self._roll_locks(target)
        self.draft = None
        self.phase = PHASE_DOOR
        self.drafted += 1
        self.steps -= 1
        self.steps_spent += 1
        self._enter(target)
        self.steps = max(0, self.steps + room.steps)
        self.keys += room.keys
        self.gems += room.gems

    def _settle(self):
        """Automatic Antechamber entry and end-of-game check."""
        self._legal = None
        if self.over or self.phase != PHASE_DOOR:
            return
        dist, prev = self.bfs()
        goal_dist = dist.get(GOAL)
        if goal_dist is not None and goal_dist <= self.steps and self.keys >= self.cfg.goal_keys:
            self._walk(GOAL, prev)
            self.keys -= self.cfg.goal_keys
            self.keys_used += self.cfg.goal_keys
            self.total_reward += self.cfg.win_reward
            self.over = self.win = True
            self._legal = []
            return
        self._legal = self._compute_legal(dist)
        if not self._legal:
            self.over = True
            self.loss_cause = self._loss_cause(dist, goal_dist)

    def _loss_cause(self, dist, goal_dist):
        if self.steps <= 0:
            return "out_of_steps"
        if goal_dist is not None:   # connected to the Antechamber but could not enter
            return "goal_keys" if self.keys < self.cfg.goal_keys else "out_of_steps"
        reachable = locked_only = False
        for r, c, _, _, locked in self.frontier():
            walk = dist.get((r, c))
            if walk is None:
                continue
            reachable = True
            if locked and self.keys < 1 and self.steps >= walk + 1:
                locked_only = True
        if locked_only:
            return "out_of_keys"
        return "out_of_steps" if reachable else "no_doors"

    # ---- display

    def render(self):
        """Text board. Each cell is 3x3 characters; doors are drawn on the cell border."""
        lines = []
        for r in range(ROWS, 0, -1):
            rows = [[], [], []]
            for c in range(1, COLS + 1):
                room = self.rooms.get((r, c))
                if room is None:
                    cell = [" . ", " . ", " . "] if self.draft is None or self.draft.target != (r, c) else [" . ", " ? ", " . "]
                else:
                    def mark(d):
                        if d not in room.doors:
                            return None
                        nb = (r + DR[d], c + DC[d])
                        if not in_grid(*nb) or (r, c) == GOAL and nb not in self.rooms:
                            return "o"
                        if nb in self.rooms:
                            return "o" if self.connected((r, c), d) else "x"
                        edge = edge_id(r, c, d)
                        return "x" if edge in self.dead else "L" if self.locks[edge] else "o"
                    mid = "@" if self.player == (r, c) else "G" if (r, c) == GOAL else "#"
                    cell = [
                        "+" + (mark(N) or "-") + "+",
                        (mark(W) or "|") + mid + (mark(E) or "|"),
                        "+" + (mark(S) or "-") + "+",
                    ]
                for i in range(3):
                    rows[i].append(cell[i])
            for i in range(3):
                lines.append((f"{r} " if i == 1 else "  ") + "".join(rows[i]))
        lines.append("   A  B  C  D  E")
        status = "WIN" if self.win else f"LOSS ({self.loss_cause})" if self.over or self.truncated else ("door" if self.phase == PHASE_DOOR else "draft")
        lines.append(f"steps {self.steps}  keys {self.keys}  gems {self.gems}  max rank {self.max_rank}  [{status}]")
        if self.draft:
            for i, o in enumerate(self.draft.offers):
                rm = ROOMS[o.room]
                doors = "".join(DIR_NAMES[d] for d in o.doors)
                lines.append(f"  [{i}] {rm.name} ({rm.shape}, doors {doors}) cost {o.cost}  steps {rm.steps:+d} keys {rm.keys:+d} gems {rm.gems:+d}")
        return "\n".join(lines)
