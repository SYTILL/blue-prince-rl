"""Local evaluation dashboard.

    python -m blueprince.dashboard            # opens http://127.0.0.1:8175
    python -m blueprince.dashboard --port 9000 --no-browser

Runs agents live and streams each finished game to the page, loads saved runs
from results/, and replays any game step by step. Uses only the standard library.
Every live run is saved to results/runs/<time>/ in the same format as evaluate.py.
"""
from __future__ import annotations

import argparse
import json
import os
import time
import webbrowser
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from .agents import AGENTS, TRAINING_DIR, PPOAgent, ppo_models
from .env import BluePrinceEnv
from .evaluate import TEST_SEED_START, load_rows, run_episode, save_run
from .game import LOSS_CAUSES, Config
from .replay import replay

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)
RESULTS = os.path.join(PROJECT, "results")
PAGE = os.path.join(HERE, "dashboard.html")
MAX_GAMES = 100_000


def config_from(q):
    overrides = {}
    for k in ("steps", "keys", "gems"):
        if q.get(k, [""])[0] != "":
            overrides[k] = int(q[k][0])
    return overrides, replace(Config(), **overrides)


def list_runs():
    """Saved runs: results/ itself (evaluate.py --out results) and results/runs/*."""
    found = []
    candidates = [RESULTS]
    runs_dir = os.path.join(RESULTS, "runs")
    if os.path.isdir(runs_dir):
        candidates += [os.path.join(runs_dir, d) for d in os.listdir(runs_dir)]
    for path in candidates:
        summary_path = os.path.join(path, "summary.json")
        if not os.path.isfile(summary_path):
            continue
        with open(summary_path) as f:
            summary = json.load(f)
        agents = summary.get("agents", {})
        found.append({
            "id": os.path.relpath(path, RESULTS).replace("\\", "/"),
            "agents": list(agents),
            "games": max((a["games"] for a in agents.values()), default=0),
            "config": summary.get("config", {}),
            "partial": summary.get("partial", False),
            "modified": os.path.getmtime(summary_path),
        })
    return sorted(found, key=lambda r: -r["modified"])


def load_run(run_id):
    path = os.path.realpath(os.path.join(RESULTS, run_id))
    if not path.startswith(os.path.realpath(RESULTS)):
        raise ValueError("run must be inside results/")
    with open(os.path.join(path, "summary.json")) as f:
        summary = json.load(f)
    rows = {}
    for name in summary.get("agents", {}):
        csv_path = os.path.join(path, f"episodes_{name}.csv")
        if os.path.isfile(csv_path):
            rows[name] = load_rows(csv_path)
    return {"id": run_id, "config": summary.get("config", {}), "seed_start": summary.get("seed_start"),
            "ppo_model": summary.get("ppo_model"), "agents": rows}


def training_runs():
    out = []
    if not os.path.isdir(TRAINING_DIR):
        return out
    for name in os.listdir(TRAINING_DIR):
        path = os.path.join(TRAINING_DIR, name, "config.json")
        if not os.path.isfile(path):
            continue
        with open(path) as f:
            cfg = json.load(f)
        progress = os.path.join(TRAINING_DIR, name, "progress.jsonl")
        out.append({"name": name, "status": cfg.get("status"), "started": cfg.get("started"),
                    "timesteps": cfg.get("args", {}).get("timesteps"),
                    "best": cfg.get("best_val_win_rate"),
                    "modified": os.path.getmtime(progress if os.path.isfile(progress) else path)})
    return sorted(out, key=lambda r: -r["modified"])


def training_run(name):
    run_dir = os.path.realpath(os.path.join(TRAINING_DIR, name))
    if not run_dir.startswith(os.path.realpath(TRAINING_DIR)):
        raise ValueError("run must be inside results/training/")
    with open(os.path.join(run_dir, "config.json")) as f:
        cfg = json.load(f)
    progress = []
    path = os.path.join(run_dir, "progress.jsonl")
    if os.path.isfile(path):
        with open(path) as f:
            progress = [json.loads(line) for line in f if line.strip()]
    return {"name": name, "config": cfg, "progress": progress}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):      # keep the terminal quiet
        pass

    def _json(self, data, status=200):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        url = urlparse(self.path)
        q = parse_qs(url.query)
        try:
            if url.path == "/":
                with open(PAGE, "rb") as f:
                    body = f.read()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif url.path == "/api/meta":
                cfg = Config()
                models = [run for run, _ in ppo_models()]
                agents = [a for a in AGENTS if a != "ppo" or models]
                self._json({"agents": agents, "all_agents": list(AGENTS), "ppo_models": models,
                            "test_seed_start": TEST_SEED_START,
                            "defaults": {"steps": cfg.steps, "keys": cfg.keys, "gems": cfg.gems},
                            "lock_prob": list(cfg.lock_prob), "loss_causes": list(LOSS_CAUSES)})
            elif url.path == "/api/runs":
                self._json(list_runs())
            elif url.path == "/api/load":
                self._json(load_run(q["run"][0]))
            elif url.path == "/api/training":
                self._json(training_runs())
            elif url.path == "/api/training_run":
                self._json(training_run(q["name"][0]))
            elif url.path == "/api/replay":
                _, config = config_from(q)
                actions = [int(a) for a in q.get("actions", [""])[0].split(",") if a != ""]
                self._json(replay(int(q["seed"][0]), actions, config))
            elif url.path == "/api/stream":
                self._stream(q)
            else:
                self._json({"error": "not found"}, 404)
        except (KeyError, ValueError, RuntimeError, OSError) as e:
            self._json({"error": str(e)}, 400)

    def _stream(self, q):
        names = [a for a in q.get("agents", [""])[0].split(",") if a in AGENTS]
        n = min(int(q.get("n", ["200"])[0]), MAX_GAMES)
        seed_start = int(q.get("seed_start", [str(TEST_SEED_START)])[0])
        overrides, config = config_from(q)
        if not names:
            raise ValueError("choose at least one agent")
        ppo_model = q.get("ppo_model", [""])[0] or None
        agents = {}
        for name in names:                           # load models before the stream starts, so errors reach the page
            if name == "ppo":
                agents[name] = PPOAgent(ppo_model)
                ppo_model = ppo_model or os.path.basename(os.path.dirname(agents[name].path))
            else:
                agents[name] = AGENTS[name]()
        if q.get("dry"):                             # only check that the run can start
            self._json({"ok": True})
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()

        def send(event, data):
            self.wfile.write(f"event: {event}\ndata: {json.dumps(data)}\n\n".encode())
            self.wfile.flush()

        run_id = "runs/" + time.strftime("%Y%m%d-%H%M%S")
        rows = {name: [] for name in names}
        envs = {name: BluePrinceEnv(config) for name in names}
        extra = {"ppo_model": ppo_model} if "ppo" in names else {}
        done = False
        try:
            send("start", {"run": run_id, "agents": names, "n": n, "seed_start": seed_start, "config": overrides, **extra})
            for seed in range(seed_start, seed_start + n):
                for name in names:                     # interleave so all agents advance together
                    row = run_episode(envs[name], agents[name], seed)
                    rows[name].append(row)
                    send("game", {"agent": name, **row})
            done = True
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass                                       # the page stopped the run
        finally:
            if any(rows.values()):
                save_run(os.path.join(RESULTS, run_id), rows, overrides, seed_start,
                         {"partial": not done, "requested_games": n, **extra})
        if done:
            try:
                send("done", {"run": run_id})
            except OSError:
                pass


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port", type=int, default=8175)
    p.add_argument("--no-browser", action="store_true")
    args = p.parse_args(argv)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    server.daemon_threads = True
    url = f"http://127.0.0.1:{args.port}"
    print(f"Dashboard at {url}  (Ctrl+C to stop)")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
