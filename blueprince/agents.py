"""Baseline agents.

Every agent has the same two methods, so evaluate.py can run any of them:

    agent.reset(seed)        called once before each game
    agent.act(env) -> int    returns a legal action for env's current state

Agents may read `env.game` (the full state) but must not use hidden information:
lock states of doors that do not exist yet, or future draws.
"""
from __future__ import annotations

import os
import random

from .game import (DC, DR, E, GOAL, N, N_DOOR_ACTIONS, PHASE_DOOR, W, decode_door,
                   edge_id, in_grid)
from .rooms import ROOMS


class RandomAgent:
    """Uniform over the legal actions. Lower baseline."""

    name = "random"

    def __init__(self):
        self.rng = random.Random(0)

    def reset(self, seed):
        self.rng = random.Random(seed)

    def act(self, env):
        return self.rng.choice(env.game.legal_actions())


# Cells next to the Antechamber, with the door direction that connects to it.
GOAL_ENTRIES = {(8, 3): N, (9, 2): E, (9, 4): W}


def goal_gap(cell):
    """Fewest rooms still to place after `cell` before one touches the Antechamber."""
    return min(abs(cell[0] - r) + abs(cell[1] - c) for r, c in GOAL_ENTRIES)


class RuleBasedAgent:
    """Hand-written heuristics (proposal section 3).

    Door phase: score every legal door and open the best one.
      - While keys are short and steps are plentiful, distance to the Antechamber
        counts less, so nearby doors are opened first (collect resources early).
      - Otherwise prefer doors whose next cell is closer to the Antechamber.
      - Penalise locked doors, with a separate penalty when the key would come
        out of the keys reserved for the goal.
      - Avoid doors from which the Antechamber is out of reach with the steps left.
    Draft phase: score every affordable blueprint and take the best one.
      - Value keys, steps and gems (net of the gem cost).
      - Value new doors, more so when they lead toward the Antechamber.
      - Never close the last open door, and avoid leaving only one.
      - Next to the Antechamber, take the room whose door connects to it.
    There is no look-ahead. All weights are constructor arguments; with the tuned
    defaults some terms are switched off (dead_end_pen, w_edge = 0).
    """

    name = "rule"

    # Defaults were picked by random search on training seeds 0-299 and checked on
    # seeds 5000-5999 (never on the test seeds).
    def __init__(self, key_target=5, slack=4, w_walk=0.5, w_gap=1.0, w_gap_explore=0.3,
                 lock_pen=5.0, reserve_pen=4.0, infeasible_pen=50.0,
                 w_key=2.0, w_key_spare=2.0, w_step=0.2, w_gem=0.8,
                 w_door=1.0, w_door_forward=2.0, dead_end_pen=0.0, thin_pen=15.0, w_edge=0.0):
        self.key_target = key_target          # keys wanted before committing to the climb
        self.slack = slack                    # spare steps needed to keep exploring
        self.w_walk = w_walk
        self.w_gap = w_gap
        self.w_gap_explore = w_gap_explore
        self.lock_pen = lock_pen
        self.reserve_pen = reserve_pen
        self.infeasible_pen = infeasible_pen
        self.w_key = w_key
        self.w_key_spare = w_key_spare
        self.w_step = w_step
        self.w_gem = w_gem
        self.w_door = w_door
        self.w_door_forward = w_door_forward
        self.dead_end_pen = dead_end_pen
        self.thin_pen = thin_pen              # penalty for leaving only one open door
        self.w_edge = w_edge                  # penalty for drafting into the outer columns

    def reset(self, seed):
        pass

    def act(self, env):
        g = env.game
        return self._door(g) if g.phase == PHASE_DOOR else self._blueprint(g)

    def _door(self, g):
        dist, _ = g.bfs()
        short_of_keys = g.keys < self.key_target
        best, best_score = None, None
        for a in g.legal_actions():
            r, c, d = decode_door(a)
            target = (r + DR[d], c + DC[d])
            walk = dist[(r, c)]
            left = g.steps - walk - 1                 # steps left after entering the new room
            gap = goal_gap(target)
            score = -self.w_walk * walk
            if g.locks[edge_id(r, c, d)]:
                score -= self.lock_pen if g.keys - 1 >= g.cfg.goal_keys else self.reserve_pen
            if left < gap + 1:                        # cannot reach the goal from there any more
                score -= self.infeasible_pen
            exploring = short_of_keys and left - gap - 1 >= self.slack
            score -= (self.w_gap_explore if exploring else self.w_gap) * gap
            if target[1] in (1, 5):                   # fewer room shapes fit on the edge
                score -= self.w_edge
            if best_score is None or score > best_score:
                best, best_score = a, score
        return best

    def _blueprint(self, g):
        draft = g.draft
        tr, tc = draft.target
        entry = (draft.d + 2) % 4
        gap_here = goal_gap(draft.target)
        # open doors elsewhere that survive this placement
        other_doors = sum(1 for r, c, d, _, _ in g.frontier() if (r + DR[d], c + DC[d]) != draft.target)
        w_key = self.w_key if g.keys < self.key_target else self.w_key_spare
        best, best_score = None, None
        for a in g.legal_actions():
            offer = draft.offers[a - N_DOOR_ACTIONS]
            room = ROOMS[offer.room]
            score = w_key * room.keys + self.w_step * room.steps + self.w_gem * (room.gems - offer.cost)
            new_doors = 0
            connects_goal = False
            for d in offer.doors:
                if d == entry:
                    continue
                nb = (tr + DR[d], tc + DC[d])
                if nb == GOAL:
                    connects_goal = True              # Antechamber has doors on all three open sides
                elif in_grid(*nb) and nb not in g.rooms:
                    new_doors += 1
                    score += self.w_door_forward if goal_gap(nb) < gap_here else self.w_door
            if connects_goal:
                score += 100.0
            elif draft.target in GOAL_ENTRIES:
                score -= 20.0                         # would wall off one side of the Antechamber
            if not connects_goal:
                if new_doors == 0:
                    score -= self.dead_end_pen
                if other_doors + new_doors == 0:      # would close the last open door
                    score -= 1000.0
                elif other_doors + new_doors == 1:
                    score -= self.thin_pen
            if best_score is None or score > best_score:
                best, best_score = a, score
        return best


PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAINING_DIR = os.path.join(PROJECT, "results", "training")


def ppo_models():
    """Trained models under results/training/<run>/, newest first: [(run, path)]."""
    found = []
    if os.path.isdir(TRAINING_DIR):
        for run in os.listdir(TRAINING_DIR):
            for fname in ("best_model.zip", "model.zip"):
                path = os.path.join(TRAINING_DIR, run, fname)
                if os.path.isfile(path):
                    found.append((run, path))
                    break
    return sorted(found, key=lambda x: -os.path.getmtime(x[1]))


class PPOAgent:
    """A trained Maskable PPO policy (see train.py).

    PPOAgent()            newest run's best_model.zip (or $BLUEPRINCE_PPO_MODEL)
    PPOAgent("run-name")  that run under results/training/, or a path to a .zip
    PPOAgent(model=m)     an in-memory model, used during training
    """

    name = "ppo"

    def __init__(self, path=None, model=None, deterministic=True):
        if model is None:
            path = path or os.environ.get("BLUEPRINCE_PPO_MODEL")
            if path and not path.endswith(".zip"):
                path = dict(ppo_models()).get(path, path)
            if not path:
                models = ppo_models()
                if not models:
                    raise FileNotFoundError("no trained model in results/training/; run python -m blueprince.train first")
                path = models[0][1]
            from sb3_contrib import MaskablePPO      # imported here so the baselines do not need torch
            model = MaskablePPO.load(path, device="cpu")
        self.model = model
        self.path = path
        self.deterministic = deterministic

    def reset(self, seed):
        pass

    def act(self, env):
        from .env import encode_observation
        action, _ = self.model.predict(encode_observation(env.game), action_masks=env.action_masks(),
                                       deterministic=self.deterministic)
        return int(action)


AGENTS = {"random": RandomAgent, "rule": RuleBasedAgent, "ppo": PPOAgent}
