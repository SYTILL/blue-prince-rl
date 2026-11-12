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
OBS_SIZE = OFFER_OFFSET + 3 * OFFER_FEATURES      # 834


def encode_observation(game: Game) -> np.ndarray:
    obs = np.zeros(OBS_SIZE, dtype=np.float32)
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
    return np.clip(obs, -1.0, 1.0)


class BluePrinceEnv(gym.Env):
    """Minimal Blue Prince.

    reset(seed=s) starts the game with seed s. reset() without a seed draws the
    next game seed from the environment's own generator, so a training run is
    reproducible from its first seed.
    """

    metadata = {"render_modes": ["ansi"]}

    def __init__(self, config: Optional[Config] = None, render_mode: Optional[str] = None):
        self.config = config or Config()
        self.render_mode = render_mode
        self.action_space = spaces.Discrete(N_ACTIONS)
        self.observation_space = spaces.Box(-1.0, 1.0, shape=(OBS_SIZE,), dtype=np.float32)
        self.game: Optional[Game] = None

    def reset(self, *, seed: Optional[int] = None, options=None):
        super().reset(seed=seed)
        game_seed = seed if seed is not None else int(self.np_random.integers(2 ** 62))
        self.game = Game(game_seed, self.config)
        return encode_observation(self.game), self._info()

    def step(self, action):
        reward = self.game.step(action)       # raises ValueError on an illegal action
        return encode_observation(self.game), float(reward), self.game.over, self.game.truncated, self._info()

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
