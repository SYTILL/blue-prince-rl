"""Rebuild a game step by step from (seed, config, actions) for viewing.

replay() returns one JSON-ready frame for the start and one after every action.
"""
from __future__ import annotations

from typing import Optional

from .game import (DC, DIR_NAMES, DR, GOAL, N_DOOR_ACTIONS, PHASE_DRAFT, START, Config,
                   Game, decode_door, edge_id, in_grid)
from .rooms import ROOMS

DIR_WORDS = ("north", "east", "south", "west")
LOSS_TEXT = {
    "out_of_steps": "ran out of steps",
    "out_of_keys": "every reachable door is locked and there are no keys",
    "goal_keys": "the Antechamber is connected but there are fewer than 3 keys",
    "no_doors": "no door left to open",
    "truncated": "stopped at the action limit",
}


def cell_name(cell):
    return "ABCDE"[cell[1] - 1] + str(cell[0])


def door_state(g: Game, cell, d):
    r, c = cell
    nb = (r + DR[d], c + DC[d])
    if not in_grid(*nb):
        return "outside"
    other = g.rooms.get(nb)
    if other is not None:
        return "joined" if (d + 2) % 4 in other.doors else "blocked"
    if cell == GOAL:
        return "sealed"
    edge = edge_id(r, c, d)
    if edge in g.dead:
        return "dead"
    return "locked" if g.locks[edge] else "open"


def room_info(idx):
    rm = ROOMS[idx]
    return {"name": rm.name, "shape": rm.shape, "rarity": rm.rarity, "cost": rm.cost,
            "steps": rm.steps, "keys": rm.keys, "gems": rm.gems, "red": rm.red}


def snapshot(g: Game, action=None, event="", path=None):
    rooms = []
    for cell, placed in g.rooms.items():
        kind = "start" if cell == START else "goal" if cell == GOAL else "room"
        info = room_info(placed.room) if placed.room >= 0 else {"name": placed.name}
        rooms.append({"r": cell[0], "c": cell[1], "kind": kind, **info,
                      "doors": {DIR_NAMES[d]: door_state(g, cell, d) for d in placed.doors}})
    draft = None
    if g.phase == PHASE_DRAFT and g.draft is not None:
        draft = {
            "target": list(g.draft.target), "origin": list(g.draft.origin), "dir": DIR_NAMES[g.draft.d],
            "offers": [{**room_info(o.room), "cost": o.cost, "doors": [DIR_NAMES[d] for d in o.doors],
                        "affordable": g.gems >= o.cost} for o in g.draft.offers],
        }
    return {
        "action": action, "event": event, "path": [list(p) for p in (path or [])],
        "rooms": rooms, "player": list(g.player), "draft": draft,
        "steps": g.steps, "keys": g.keys, "gems": g.gems, "max_rank": g.max_rank,
        "reward": round(g.total_reward, 4), "phase": "draft" if g.phase == PHASE_DRAFT else "door",
        "over": g.over or g.truncated, "win": g.win, "loss_cause": g.loss_cause,
    }


def _path(prev, start, target):
    out = [target]
    while out[-1] != start:
        out.append(prev[out[-1]])
    return out[::-1]


def replay(seed: int, actions, config: Optional[Config] = None):
    g = Game(seed, config)
    cfg = g.cfg
    frames = [snapshot(g, None, f"Start in the Entrance Hall with {cfg.steps} steps, {cfg.keys} keys, {cfg.gems} gems.",
                       [list(START)])]
    for a in actions:
        a = int(a)
        before = (g.steps, g.keys, g.gems, g.max_rank)
        start = g.player
        if a < N_DOOR_ACTIONS:
            r, c, d = decode_door(a)
            _, prev = g.bfs()
            path = _path(prev, start, (r, c))
            locked = g.locks[edge_id(r, c, d)]
            g.step(a)
            walked = len(path) - 1
            text = f"Opened the {DIR_WORDS[d]} door of {cell_name((r, c))}"
            if walked:
                text += f" after walking {walked} room{'s' if walked > 1 else ''}"
            if locked:
                text += ", using a key"
            text += "."
            if g.phase != PHASE_DRAFT:
                text += " No room in the deck fits there; the door is blocked."
        else:
            draft = g.draft
            offer = draft.offers[a - N_DOOR_ACTIONS]
            rm = ROOMS[offer.room]
            target = draft.target
            path = [start, list(target)]
            g.step(a)
            text = f"Drafted {rm.name} at {cell_name(target)}"
            if offer.cost:
                text += f" for {offer.cost} gem{'s' if offer.cost > 1 else ''}"
            gains = [f"{v:+d} {k}" for k, v in (("steps", rm.steps), ("keys", rm.keys), ("gems", rm.gems)) if v]
            text += (": " + ", ".join(gains) if gains else "") + "."
            if g.win:      # automatic Antechamber entry happened after the placement
                _, prev = g.bfs(target)
                path = [list(start)] + _path(prev, target, GOAL)
        path = [list(p) for p in path]
        if g.max_rank > before[3]:
            text += f" Reached rank {g.max_rank}."
        if g.win:
            text += f" Entered the Antechamber ({-cfg.goal_keys} keys). Win."
        elif g.over or g.truncated:
            text += f" Game over: {LOSS_TEXT.get(g.loss_cause, g.loss_cause)}."
        frames.append(snapshot(g, a, text, path))
    return frames
