"""Train a Maskable PPO agent.

    python -m blueprince.train --timesteps 2e6 --name first
    python -m blueprince.train --timesteps 3e6 --cur-keys 6 --cur-gems 6 --cur-frac 0.5

Writes results/training/<name>/:
    config.json      arguments, validation baselines, status
    progress.jsonl   one line per evaluation (training and validation metrics)
    model.zip        latest model;  best_model.zip  best validation win rate
    tb/              TensorBoard logs (tensorboard --logdir results/training)

Validation uses seeds VAL_SEED_START.. at the target difficulty. They are separate
from the test seeds in evaluate.py, which training never uses. Training games use
seeds drawn from 2**62 values, so they do not hit either set in practice.

Curriculum (optional): start with --cur-keys / --cur-gems and move linearly to the
target --keys / --gems over the first --cur-frac of training.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from collections import deque
from dataclasses import replace

from .agents import TRAINING_DIR, PPOAgent, RandomAgent, RuleBasedAgent
from .env import BluePrinceEnv
from .evaluate import evaluate, summarize
from .game import Config

VAL_SEED_START = 500_000


def metrics(summary):
    keep = ("games", "win_rate", "avg_max_rank", "avg_return", "avg_steps_spent", "avg_keys_left",
            "avg_gems_left", "loss_causes", "max_rank_hist")
    return {k: summary[k] for k in keep}


def make_callback(run_dir, args, target, val_seeds):
    from stable_baselines3.common.callbacks import BaseCallback

    class TrainLogger(BaseCallback):
        """Tracks finished training games, applies the curriculum, evaluates and saves."""

        def __init__(self):
            super().__init__()
            self.recent = deque(maxlen=1000)
            self.episodes = 0
            self.last_eval = -10 ** 12
            self.best = -1.0
            self.stage = None
            self.t0 = time.time()

        def current_config(self):
            if args.cur_keys is None and args.cur_gems is None:
                return target
            frac = min(1.0, self.num_timesteps / max(1.0, args.cur_frac * args.timesteps))
            lerp = lambda a, b: int(round(a + (b - a) * frac))
            return replace(target,
                           keys=lerp(args.cur_keys if args.cur_keys is not None else target.keys, target.keys),
                           gems=lerp(args.cur_gems if args.cur_gems is not None else target.gems, target.gems))

        def apply_curriculum(self):
            cfg = self.current_config()
            if cfg != self.stage:
                self.training_env.env_method("set_config", cfg)
                self.stage = cfg

        def _on_training_start(self):
            self.apply_curriculum()

        def _on_rollout_start(self):
            self.apply_curriculum()

        def _on_step(self):
            for done, info in zip(self.locals["dones"], self.locals["infos"]):
                if done:
                    self.episodes += 1
                    self.recent.append((info["win"], info["max_rank"], info.get("episode", {}).get("r", 0.0)))
            if self.num_timesteps - self.last_eval >= args.eval_every:
                self.evaluate()
            return True

        def _on_training_end(self):
            if self.num_timesteps != self.last_eval:
                self.evaluate()

        def evaluate(self):
            self.last_eval = self.num_timesteps
            t = time.time()
            val = metrics(summarize(evaluate(PPOAgent(model=self.model), val_seeds, target)))
            train = None
            if self.recent:
                n = len(self.recent)
                train = {"games": n, "win_rate": sum(x[0] for x in self.recent) / n,
                         "avg_max_rank": sum(x[1] for x in self.recent) / n,
                         "avg_return": sum(x[2] for x in self.recent) / n}
            stage = self.stage or target
            record = {"timesteps": self.num_timesteps, "time": round(time.time() - self.t0, 1),
                      "eval_seconds": round(time.time() - t, 1), "episodes": self.episodes,
                      "stage": {"steps": stage.steps, "keys": stage.keys, "gems": stage.gems},
                      "train": train, "val": val}
            with open(os.path.join(run_dir, "progress.jsonl"), "a") as f:
                f.write(json.dumps(record) + "\n")
            self.logger.record("val/win_rate", val["win_rate"])
            self.logger.record("val/avg_max_rank", val["avg_max_rank"])
            if train:
                self.logger.record("train/win_rate_recent", train["win_rate"])
                self.logger.record("train/max_rank_recent", train["avg_max_rank"])
            self.model.save(os.path.join(run_dir, "model.zip"))
            if val["win_rate"] > self.best:
                self.best = val["win_rate"]
                self.model.save(os.path.join(run_dir, "best_model.zip"))
            train_txt = f"train win {100 * train['win_rate']:.1f}%  " if train else ""
            print(f"[{self.num_timesteps:>9,}] {train_txt}val win {100 * val['win_rate']:.1f}%  "
                  f"val rank {val['avg_max_rank']:.2f}  (best {100 * self.best:.1f}%, {record['time']:.0f}s)", flush=True)

    return TrainLogger()


def update_config_file(run_dir, **changes):
    path = os.path.join(run_dir, "config.json")
    with open(path) as f:
        data = json.load(f)
    data.update(changes)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", default=None, help="run name (default: time stamp)")
    p.add_argument("--timesteps", type=float, default=2e6)
    p.add_argument("--n-envs", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--threads", type=int, default=4, help="torch threads for the update step")
    p.add_argument("--steps", type=int, help="target starting steps (default 50)")
    p.add_argument("--keys", type=int, help="target starting keys (default 2)")
    p.add_argument("--gems", type=int, help="target starting gems (default 2)")
    p.add_argument("--cur-keys", type=int, help="curriculum: starting keys at the beginning of training")
    p.add_argument("--cur-gems", type=int, help="curriculum: starting gems at the beginning of training")
    p.add_argument("--cur-frac", type=float, default=0.5, help="curriculum: fraction of training to reach the target")
    p.add_argument("--eval-every", type=int, default=100_000)
    p.add_argument("--eval-games", type=int, default=300)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--n-steps", type=int, default=256, help="rollout length per env")
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--n-epochs", type=int, default=10)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--policy", choices=("mlp", "cnn"), default="mlp",
                   help="mlp: dense layers on the flat observation; cnn: BoardCNN (networks.py) on the 9x5 board")
    p.add_argument("--net", default=None, help="hidden layer sizes of the policy and value heads "
                                               "(default 256,256 for mlp, 256 for cnn)")
    p.add_argument("--cnn-channels", type=int, default=64)
    p.add_argument("--cnn-layers", type=int, default=3)
    p.add_argument("--obs", type=int, choices=(1, 2), default=1,
                   help="observation version: 1 = 834 numbers (Exp 1-2); 2 = adds goal, deck and offer-placement features")
    args = p.parse_args(argv)
    args.timesteps = int(args.timesteps)
    if args.net is None:
        args.net = "256" if args.policy == "cnn" else "256,256"

    import torch
    from sb3_contrib import MaskablePPO
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

    torch.set_num_threads(args.threads)
    overrides = {k: getattr(args, k) for k in ("steps", "keys", "gems") if getattr(args, k) is not None}
    target = replace(Config(), **overrides)
    name = args.name or time.strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join(TRAINING_DIR, name)
    if os.path.exists(os.path.join(run_dir, "progress.jsonl")):
        raise SystemExit(f"{run_dir} already has a run; pick another --name")
    os.makedirs(run_dir, exist_ok=True)

    val_seeds = range(VAL_SEED_START, VAL_SEED_START + args.eval_games)
    print(f"Run {name}: baselines on {args.eval_games} validation seeds...", flush=True)
    baselines = {a.name: metrics(summarize(evaluate(a, val_seeds, target))) for a in (RandomAgent(), RuleBasedAgent())}
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump({"name": name, "args": vars(args), "target": overrides, "val_seed_start": VAL_SEED_START,
                   "baselines": baselines, "status": "running", "started": time.strftime("%Y-%m-%d %H:%M:%S")}, f, indent=2)
    print("  " + "  ".join(f"{k} {100 * v['win_rate']:.1f}%" for k, v in baselines.items()), flush=True)

    vec_cls = SubprocVecEnv if args.n_envs > 1 else DummyVecEnv
    env = make_vec_env(BluePrinceEnv, n_envs=args.n_envs, seed=args.seed, env_kwargs={"config": target, "obs_version": args.obs}, vec_env_cls=vec_cls)
    net = [int(x) for x in args.net.split(",") if x]
    policy_kwargs = {"net_arch": {"pi": net, "vf": net}}
    if args.policy == "cnn":
        from .networks import BoardCNN
        policy_kwargs.update(features_extractor_class=BoardCNN,
                             features_extractor_kwargs={"channels": args.cnn_channels, "layers": args.cnn_layers})
    model = MaskablePPO(
        "MlpPolicy", env, learning_rate=args.lr, n_steps=args.n_steps, batch_size=args.batch_size,
        n_epochs=args.n_epochs, gamma=args.gamma, gae_lambda=args.gae_lambda, ent_coef=args.ent_coef,
        clip_range=args.clip, policy_kwargs=policy_kwargs,
        tensorboard_log=os.path.join(run_dir, "tb"), seed=args.seed, device="cpu", verbose=0)
    callback = make_callback(run_dir, args, target, val_seeds)
    status = "done"
    try:
        model.learn(total_timesteps=args.timesteps, callback=callback, tb_log_name="ppo")
    except KeyboardInterrupt:
        status = "stopped"
        model.save(os.path.join(run_dir, "model.zip"))
        print("Stopped; latest model saved.")
    finally:
        env.close()
        update_config_file(run_dir, status=status, finished=time.strftime("%Y-%m-%d %H:%M:%S"),
                           best_val_win_rate=callback.best, timesteps_done=int(model.num_timesteps))
    print(f"Saved to {run_dir}")


if __name__ == "__main__":
    main()
