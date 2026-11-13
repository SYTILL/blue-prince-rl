"""Gymnasium wrapper around the rules engine, plus the observation encoding.

Observation: flat float32 vector of length OBS_SIZE (see DESIGN.md for the layout).
Action: Discrete(183) as defined in actions.md; use `action_masks()` (sb3-contrib
MaskablePPO convention) to get the legal actions.
"""
from __future__ import annotations

from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .game import (COLS, DC, DR, GOAL, N_ACTIONS, PHASE_DOOR, PHASE_DRAFT, ROWS,
                   Config, Game, cell_id, edge_id, in_grid)
from .rooms import RARITIES, ROOMS, SHAPE_COUNTS, SHAPES

# ---- observation layout
CELL_FEATURES = 17
#  0      room present
#  1-4    room has a door N, E, S, W
#  5-8    that door is an unopened, unlocked door (N, E, S, W)
#  9-12   that door is an unopened, locked door (N, E, S, W)
#  13     player is here
#  14     reachable on foot from the player
#  15     walking distance from the player / 20
#  16     this is the cell being drafted (draft phase only)
GLOBAL_FEATURES = 12
#  0 steps/100  1 keys/10  2 gems/10  3 max rank/9  4 door phase  5 draft phase
#  6 deck left/99  7-11 fraction of each shape (D, S, L, T, X) still in the deck
OFFER_FEATURES = 19
#  0 offer present  1-4 doors N, E, S, W  5 cost/5  6 affordable
#  7-10 rarity one-hot (C, S, U, R)  11 steps/10  12 keys/3  13 gems/3
#  14-18 shape one-hot (D, S, L, T, X)
BOARD_SIZE = ROWS * COLS * CELL_FEATURES
GLOBAL_OFFSET = BOARD_SIZE
OFFER_OFFSET = GLOBAL_OFFSET + GLOBAL_FEATURES
OBS_SIZE = OFFER_OFFSET + 3 * OFFER_FEATURES      # 834 (version 1)

# Version 2 appends computed features to the version 1 vector (Exp 3). They give
# the agent what the rule-based agent computes itself: distance and keys still
# needed for the goal, what is left in the deck, and what each offer does once placed.
PLAN_FEATURES = 12
#  0 Antechamber reachable on foot   1 walking distance to it / 20
#  2 fewest steps to the goal through an open door (walk + 1 + rooms still needed) / 20
#  3 step slack: (steps - that) / 50     4 fewest rooms still needed / 10
#  5 keys short of the goal (3 - keys) / 3, can be negative
#  6 usable doors (reachable, enough steps, unlocked or a key in hand) / 10
#  7 reachable locked doors / 10
#  8-11 rooms left in the deck that give keys / 10, steps / 10, gems / 10, total keys in the deck / 20
OFFER_PLAN_FEATURES = 7
#  0 new open doors (lead to an empty cell) / 3   1 doors that join a neighbour / 3
#  2 doors walled off by a neighbour without a matching door / 3
#  3 joins the Antechamber   4 open doors left on the board after placing it / 10
#  5 fewest rooms to the goal from its new doors / 10 (1 if none)
#  6 average lock chance of its new doors
PLAN_OFFSET = OBS_SIZE
OFFER_PLAN_OFFSET = PLAN_OFFSET + PLAN_FEATURES
OBS_SIZE_V2 = OFFER_PLAN_OFFSET + 3 * OFFER_PLAN_FEATURES   # 867
OBS_SIZES = {1: OBS_SIZE, 2: OBS_SIZE_V2}


def obs_version(size: int) -> int:
    """Observation version from a vector length (used to load older models)."""
    for v, n in OBS_SIZES.items():
        if n == size:
            return v
    raise ValueError(f"unknown observation size {size}")


def rooms_to_goal(cell) -> int:
    """Fewest rooms still to place to join the Antechamber from an empty `cell`,
    counting the room at `cell` (Manhattan distance to the nearest cell next to an
    Antechamber door)."""
    entries = [(GOAL[0] + DR[d], GOAL[1] + DC[d]) for d in (1, 2, 3)]      # E, S, W of the goal
    return 1 + min(abs(cell[0] - r) + abs(cell[1] - c) for r, c in entries if in_grid(r, c))


def encode_observation(game: Game, version: int = 1) -> np.ndarray:
    obs = np.zeros(OBS_SIZES[version], dtype=np.float32)
    dist = game.bfs()[0] if not game.over else {}
    for (r, c), room in game.rooms.items():
        base = cell_id(r, c) * CELL_FEATURES
        obs[base] = 1
        for d in room.doors:
            obs[base + 1 + d] = 1
            nb = (r + DR[d], c + DC[d])
            if (r, c) == GOAL or not in_grid(*nb) or nb in game.rooms:
                continue
            edge = edge_id(r, c, d)
            if edge not in game.dead:
                obs[base + (9 if game.locks[edge] else 5) + d] = 1
        if (r, c) in dist:
            obs[base + 14] = 1
            obs[base + 15] = dist[(r, c)] / 20
    obs[cell_id(*game.player) * CELL_FEATURES + 13] = 1
    if game.draft is not None:
        obs[cell_id(*game.draft.target) * CELL_FEATURES + 16] = 1

    g = GLOBAL_OFFSET
    obs[g + 0] = game.steps / 100
    obs[g + 1] = game.keys / 10
    obs[g + 2] = game.gems / 10
    obs[g + 3] = game.max_rank / ROWS
    obs[g + 4] = game.phase == PHASE_DOOR
    obs[g + 5] = game.phase == PHASE_DRAFT
    obs[g + 6] = len(game.deck) / len(ROOMS)
    for i, shape in enumerate(SHAPES):
        obs[g + 7 + i] = sum(1 for idx in game.deck if ROOMS[idx].shape == shape) / SHAPE_COUNTS[shape]

    if game.draft is not None:
        for slot, offer in enumerate(game.draft.offers):
            o = OFFER_OFFSET + slot * OFFER_FEATURES
            room = ROOMS[offer.room]
            obs[o] = 1
            for d in offer.doors:
                obs[o + 1 + d] = 1
            obs[o + 5] = offer.cost / 5
            obs[o + 6] = game.gems >= offer.cost
            obs[o + 7 + RARITIES.index(room.rarity)] = 1
            obs[o + 11] = room.steps / 10
            obs[o + 12] = room.keys / 3
            obs[o + 13] = room.gems / 3
            obs[o + 14 + SHAPES.index(room.shape)] = 1
    if version >= 2:
        _encode_plan(game, dist, obs)
    return np.clip(obs, -1.0, 1.0)


def _encode_plan(game: Game, dist, obs):
    p = PLAN_OFFSET
    frontier = game.frontier()
    if GOAL in dist:
        obs[p + 0] = 1
        obs[p + 1] = dist[GOAL] / 20
    best_steps = best_rooms = None
    usable = locked = 0
    for r, c, d, _, is_locked in frontier:
        walk = dist.get((r, c))
        if walk is None:
            continue
        locked += is_locked
        if game.steps >= walk + 1 and (not is_locked or game.keys >= 1):
            usable += 1
        rooms = rooms_to_goal((r + DR[d], c + DC[d]))
        need = walk + rooms + 1      # walk there, one step per room placed, one into the goal
        best_steps = need if best_steps is None else min(best_steps, need)
        best_rooms = rooms if best_rooms is None else min(best_rooms, rooms)
    if GOAL in dist:
        best_steps, best_rooms = dist[GOAL], 0
    if best_steps is not None:
        obs[p + 2] = best_steps / 20
        obs[p + 3] = (game.steps - best_steps) / 50
        obs[p + 4] = best_rooms / 10
    else:
        obs[p + 2] = obs[p + 4] = 1
        obs[p + 3] = -1
    obs[p + 5] = (game.cfg.goal_keys - game.keys) / 3
    obs[p + 6] = usable / 10
    obs[p + 7] = locked / 10
    deck = [ROOMS[i] for i in game.deck]
    obs[p + 8] = sum(1 for x in deck if x.keys > 0) / 10
    obs[p + 9] = sum(1 for x in deck if x.steps > 0) / 10
    obs[p + 10] = sum(1 for x in deck if x.gems > 0) / 10
    obs[p + 11] = sum(max(0, x.keys) for x in deck) / 20

    if game.draft is None:
        return
    target = game.draft.target
    into_target = sum(1 for r, c, d, _, _ in frontier if (r + DR[d], c + DC[d]) == target)
    for slot, offer in enumerate(game.draft.offers):
        o = OFFER_PLAN_OFFSET + slot * OFFER_PLAN_FEATURES
        new = joins = walled = 0
        goal_join = False
        nearest = None
        lock_sum = 0.0
        for d in offer.doors:
            nb = (target[0] + DR[d], target[1] + DC[d])
            other = game.rooms.get(nb)
            if other is None:
                new += 1
                rooms = rooms_to_goal(nb)
                nearest = rooms if nearest is None else min(nearest, rooms)
                lock_sum += game.cfg.lock_prob[max(target[0], nb[0])]
            elif (d + 2) % 4 in other.doors:
                joins += 1
                goal_join |= nb == GOAL
            else:
                walled += 1
        joins -= 1                   # the door the player came through always joins
        obs[o + 0] = new / 3
        obs[o + 1] = joins / 3
        obs[o + 2] = walled / 3
        obs[o + 3] = goal_join
        obs[o + 4] = (len(frontier) - into_target + new) / 10
        obs[o + 5] = 0 if goal_join else (nearest / 10 if nearest is not None else 1)
        obs[o + 6] = lock_sum / new if new else 0


class BluePrinceEnv(gym.Env):
    """Minimal Blue Prince.

    reset(seed=s) starts the game with seed s. reset() without a seed draws the
    next game seed from the environment's own generator, so a training run is
    reproducible from its first seed.
    """

    metadata = {"render_modes": ["ansi"]}

    def __init__(self, config: Optional[Config] = None, render_mode: Optional[str] = None, obs_version: int = 1):
        self.config = config or Config()
        self.render_mode = render_mode
        self.obs_version = obs_version
        self.action_space = spaces.Discrete(N_ACTIONS)
        self.observation_space = spaces.Box(-1.0, 1.0, shape=(OBS_SIZES[obs_version],), dtype=np.float32)
        self.game: Optional[Game] = None

    def reset(self, *, seed: Optional[int] = None, options=None):
        super().reset(seed=seed)
        game_seed = seed if seed is not None else int(self.np_random.integers(2 ** 62))
        self.game = Game(game_seed, self.config)
        return encode_observation(self.game, self.obs_version), self._info()

    def step(self, action):
        reward = self.game.step(action)       # raises ValueError on an illegal action
        return (encode_observation(self.game, self.obs_version), float(reward), self.game.over,
                self.game.truncated, self._info())

    def action_masks(self) -> np.ndarray:
        return np.array(self.game.action_mask(), dtype=bool)

    def set_config(self, config: Config):
        """Change the rules for the next reset (used by the training curriculum)."""
        self.config = config

    def render(self):
        return self.game.render()

    def _info(self):
        g = self.game
        return {
            "seed": g.seed, "phase": g.phase, "win": g.win, "loss_cause": g.loss_cause,
            "max_rank": g.max_rank, "steps": g.steps, "keys": g.keys, "gems": g.gems,
            "steps_spent": g.steps_spent, "keys_used": g.keys_used, "gems_used": g.gems_used,
            "rooms_drafted": g.drafted, "n_actions": len(g.actions),
        }
