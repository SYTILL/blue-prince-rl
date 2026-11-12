"""Run agents on a fixed set of seeds and report the proposal's metrics.

    python -m blueprince.evaluate                       # random + rule, 1000 test seeds
    python -m blueprince.evaluate --agents rule --n 200 --keys 4
    python -m blueprince.evaluate --out results --plot  # also write CSV / JSON / PNG
    python -m blueprince.evaluate --agents random rule ppo --ppo-model ppo-base

Test seeds are TEST_SEED_START .. TEST_SEED_START + n - 1. Never pass these to
reset() during training or tuning. Unseeded resets draw from 2**62 values, so
training will not hit them by chance.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter
from dataclasses import replace

from .agents import AGENTS
from .env import BluePrinceEnv
from .game import LOSS_CAUSES, Config

TEST_SEED_START = 1_000_000


def run_episode(env, agent, seed):
    """Play one game and return its result row."""
    env.reset(seed=seed)
    agent.reset(seed)
    total = 0.0
    while True:
        _, reward, terminated, truncated, info = env.step(agent.act(env))
        total += reward
        if terminated or truncated:
            break
    return {
        "seed": seed, "win": int(info["win"]), "loss_cause": info["loss_cause"] or "",
        "max_rank": info["max_rank"], "return": round(total, 4),
        "steps_spent": info["steps_spent"], "steps_left": info["steps"],
        "keys_left": info["keys"], "gems_left": info["gems"],
        "keys_used": info["keys_used"], "gems_used": info["gems_used"],
        "rooms_drafted": info["rooms_drafted"], "n_actions": info["n_actions"],
        "actions": list(env.game.actions),     # with the seed and config, replays the game exactly
    }


def evaluate(agent, seeds, config=None):
    env = BluePrinceEnv(config)
    return [run_episode(env, agent, s) for s in seeds]


INT_FIELDS = ("seed", "win", "max_rank", "steps_spent", "steps_left", "keys_left", "gems_left",
              "keys_used", "gems_used", "rooms_drafted", "n_actions")


def save_run(out, rows_by_agent, overrides, seed_start, extra=None):
    """Write episodes_<agent>.csv (actions as a space-separated column) and summary.json."""
    os.makedirs(out, exist_ok=True)
    summaries = {}
    for name, rows in rows_by_agent.items():
        if not rows:
            continue
        summaries[name] = summarize(rows)
        with open(os.path.join(out, f"episodes_{name}.csv"), "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            for row in rows:
                writer.writerow({**row, "actions": " ".join(map(str, row["actions"]))})
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump({"config": overrides, "seed_start": seed_start, **(extra or {}), "agents": summaries}, f, indent=2)
    return summaries


def load_rows(path):
    """Read an episodes_<agent>.csv back into result rows."""
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            for k in INT_FIELDS:
                row[k] = int(row[k])
            row["return"] = float(row["return"])
            row["actions"] = [int(a) for a in row.get("actions", "").split()]
            rows.append(row)
    return rows


def summarize(rows):
    n = len(rows)

    def mean(key):
        return sum(r[key] for r in rows) / n

    causes = Counter(r["loss_cause"] for r in rows if not r["win"])
    return {
        "games": n,
        "win_rate": mean("win"),
        "avg_max_rank": mean("max_rank"),
        "avg_return": mean("return"),
        "avg_steps_spent": mean("steps_spent"),
        "avg_keys_left": mean("keys_left"),
        "avg_gems_left": mean("gems_left"),
        "avg_rooms_drafted": mean("rooms_drafted"),
        "loss_causes": {c: causes.get(c, 0) / n for c in LOSS_CAUSES},
        "max_rank_hist": {k: sum(1 for r in rows if r["max_rank"] == k) for k in range(1, 10)},
    }


def print_table(summaries):
    cols = [("win %", lambda s: f"{100 * s['win_rate']:.1f}"),
            ("max rank", lambda s: f"{s['avg_max_rank']:.2f}"),
            ("return", lambda s: f"{s['avg_return']:.2f}"),
            ("steps", lambda s: f"{s['avg_steps_spent']:.1f}"),
            ("keys left", lambda s: f"{s['avg_keys_left']:.2f}"),
            ("gems left", lambda s: f"{s['avg_gems_left']:.2f}"),
            ("rooms", lambda s: f"{s['avg_rooms_drafted']:.1f}")]
    cols += [(c + " %", lambda s, c=c: f"{100 * s['loss_causes'][c]:.1f}") for c in LOSS_CAUSES]
    print("| agent | " + " | ".join(name for name, _ in cols) + " |")
    print("|---|" + "---:|" * len(cols))
    for name, s in summaries.items():
        print(f"| {name} | " + " | ".join(f(s) for _, f in cols) + " |")


def plot(summaries, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(summaries)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.bar(names, [100 * summaries[n]["win_rate"] for n in names], color="#4477aa")
    ax1.set_ylabel("win rate (%)")
    ax1.set_title("Win rate")
    width = 0.8 / len(names)
    for i, n in enumerate(names):
        hist = summaries[n]["max_rank_hist"]
        games = summaries[n]["games"]
        ax2.bar([k + i * width - 0.4 + width / 2 for k in range(1, 10)],
                [100 * hist[k] / games for k in range(1, 10)], width=width, label=n)
    ax2.set_xticks(range(1, 10))
    ax2.set_xlabel("highest rank reached")
    ax2.set_ylabel("games (%)")
    ax2.set_title("Highest rank reached")
    ax2.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--agents", nargs="+", default=["random", "rule"], choices=sorted(AGENTS))
    p.add_argument("--n", type=int, default=1000, help="number of test seeds")
    p.add_argument("--seed-start", type=int, default=TEST_SEED_START)
    p.add_argument("--steps", type=int, help="starting steps (default 50)")
    p.add_argument("--keys", type=int, help="starting keys (default 2)")
    p.add_argument("--gems", type=int, help="starting gems (default 2)")
    p.add_argument("--out", help="directory for episodes_<agent>.csv and summary.json")
    p.add_argument("--plot", action="store_true", help="also write summary.png (needs --out)")
    p.add_argument("--ppo-model", help="training run name or .zip path for the ppo agent (default: newest run)")
    args = p.parse_args(argv)

    overrides = {k: getattr(args, k) for k in ("steps", "keys", "gems") if getattr(args, k) is not None}
    config = replace(Config(), **overrides)
    seeds = range(args.seed_start, args.seed_start + args.n)

    agents = {name: AGENTS[name](args.ppo_model) if name == "ppo" else AGENTS[name]() for name in args.agents}
    rows_by_agent = {name: evaluate(agent, seeds, config) for name, agent in agents.items()}
    summaries = {name: summarize(rows) for name, rows in rows_by_agent.items()}
    print(f"{args.n} games, seeds {seeds[0]}..{seeds[-1]}, start: {config.steps} steps, {config.keys} keys, {config.gems} gems")
    extra = {}
    if "ppo" in agents:
        extra["ppo_model"] = os.path.basename(os.path.dirname(agents["ppo"].path))
        print(f"ppo model: {agents['ppo'].path}")
    print()
    print_table(summaries)
    if args.out:
        save_run(args.out, rows_by_agent, overrides, args.seed_start, extra)
        if args.plot:
            plot(summaries, os.path.join(args.out, "summary.png"))


if __name__ == "__main__":
    main()
