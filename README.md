# Minimal Blue Prince: RL project (CS 175, Team 10)

Team: Songhao Gao, Junsang Yoon, Eason Chen.

This repository holds:
- a Gymnasium environment for a minimal version of *Blue Prince*,
- two baseline agents (random and rule-based),
- a Maskable PPO training pipeline with an MLP or a CNN policy,
- an evaluation script on fixed test seeds,
- a local dashboard that shows evaluation and training and replays any game.

## Status (2026-11-12)

Done:
- **Environment**: rules engine, Gymnasium wrapper and action masking, with 219 tests.
- **Baselines and evaluation**: random and rule-based agents, evaluated on 1000 test seeds.
- **PPO**: training pipeline, validation and checkpoints. Exp 1 (MLP) and Exp 2 (CNN) are finished.
- **Observation v2** (`--obs 2`, 867 numbers): adds goal distance, step slack, keys short, deck contents, and what each offered blueprint does once placed. Version 1 models still load.
- **Dashboard and documentation**: design document, project log, experiment log.

Results on the 1000 test seeds (1,000,000–1,000,999). Every game starts with 50 steps, 2 keys and 2 gems.

| Agent | Win rate | Avg highest rank |
|---|---:|---:|
| Random | 0.0% | 3.69 |
| Rule-based (tuned heuristic) | **24.4%** | 8.00 |
| PPO, MLP (`ppo-base`) | 7.5% | 7.69 |
| PPO, MLP with curriculum (`ppo-curriculum`) | 7.1% | 7.54 |
| PPO, CNN, seed 0 (`ppo-cnn`) | 8.9% | 7.61 |
| PPO, CNN, seed 1 (`ppo-cnn-s1`) | 9.0% | 7.75 |

What the results show:
- PPO clearly beats random and learns to climb the manor, but it wins less than half as often as the rule-based agent.
- The CNN adds about 1.5 points over the MLP. A paired test on the same games says this is not significant (z 1.2–1.6), and the CNN trains 3.3× slower, so we went back to the MLP.
- PPO's typical failures:
  - It walks too much and runs out of steps.
  - It drafts fewer rooms than the rule-based agent.
  - It arrives at the Antechamber with fewer than the 3 keys needed to enter.

**Exp 3 (stopped half-way, 2026-11-12).** Five MLP runs were stopped at 1.2–1.5M of 3M steps. None is evaluated on the test seeds yet.
- **3a, PPO settings with observation v1**: `ppo-base-s1`, `ppo-ent03`, `ppo-ent003`, `ppo-lr1e4`.
- **3b, observation v2**: `ppo-obs2-s1`. Its seed-0 twin `ppo-obs2` never started.

At 1.2M steps, measured as training win rate over the last 1000 games:
- Observation v2: 8.5%.
- Every observation v1 run: 2.4–4.4%.
- The PPO setting changes made no difference.

So far missing information, not the PPO settings, looks like the bottleneck. This is one seed and half-trained, so it is not yet a result.

All the details are in three files:
- [EXPERIMENTS.md](EXPERIMENTS.md): every training run with its settings and results.
- [PROJECT_LOG.md](PROJECT_LOG.md): timeline, decisions D1–D22 with their reasons, analysis notes and material for the report.
- [DESIGN.md](DESIGN.md): how the system works.

## Where to start

1. **Set up and check.** Follow *Setup* below, then run `python -m pytest`. All 219 tests should pass.
2. **Read [DESIGN.md](DESIGN.md).** It covers:
   - the rules,
   - the action space (183 actions with a mask),
   - the observation (version 1: 834 numbers; version 2: 867),
   - the reward and the seeds,
   - how to add an agent.
3. **Look at the results.** Run `python -m blueprince.dashboard`.
   - The Training tab shows the learning curves, including the stopped Exp 3 runs.
   - On the Evaluation tab, load a run from `results/runs/` and click a game to replay it.
4. **Continue Exp 3b. This is the most useful next step.** Train observation v2 to 3M steps on two seeds. Each run takes about 1–1.5 hours alone. Two at once is fine; six at once ran out of memory on a 32 GB Windows machine.
   ```
   python -m blueprince.train --name ppo-obs2 --obs 2 --timesteps 3e6 --n-envs 8 --seed 0
   python -m blueprince.train --name ppo-obs2-s1b --obs 2 --timesteps 3e6 --n-envs 8 --seed 1
   ```
   Runs cannot resume from a checkpoint, so use a new name for the seed-1 rerun, or delete `results/training/ppo-obs2-s1` first. Then evaluate on the test seeds:
   ```
   python -m blueprince.evaluate --agents ppo --ppo-model ppo-obs2 --out results/runs/test-ppo-obs2
   ```
   Compare against MLP v1 (7.1–7.5%), CNN (8.9–9.0%) and rule-based (24.4%).
5. **Other tasks**, ordered by expected value. [EXPERIMENTS.md](EXPERIMENTS.md) and [PROJECT_LOG.md](PROJECT_LOG.md) section 5 have the reasoning.
   1. **Qualitative analysis.** Replay about 20 PPO losses in the dashboard and describe the typical mistakes. The proposal promises this, and the status report needs it.
   2. **Finish Exp 3a only if needed.** At 1.2M steps the PPO setting changes were not better than the defaults. One full MLP v1 run with seed 1 (`ppo-base-s1`) would still give the two-seed comparison.
   3. **Reward shaping**, on top of observation v2: a small cost per step walked, and a bonus for reaching rank 8–9 with 3 or more keys. Keep both well below the +10 win reward.
   4. **Longer training** (10M steps or more) of the best setup. On CPU, an MLP run does about 600–800 steps/s.
6. **Log what you do.** Add each run to [EXPERIMENTS.md](EXPERIMENTS.md) and each decision to [PROJECT_LOG.md](PROJECT_LOG.md), with the date and the reason. The report will be written from these files.

Open question for the team: should room colour mean anything? In the minimal rules it has no effect, so it is not observed (D19).

## Setup

Python 3.12. CPU only is fine.

```
python -m venv .venv
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```
python -m pytest                                   # 219 tests, about a minute
python -m blueprince.evaluate                      # random + rule on the 1000 test seeds
python -m blueprince.evaluate --out results --plot # also write CSV, JSON and PNG
python -m blueprince.evaluate --agents rule --n 200 --keys 5 --gems 5
python -m blueprince.dashboard                     # live dashboard at http://127.0.0.1:8175
python -m blueprince.train --name my-run --timesteps 3e6               # MLP policy
python -m blueprince.train --name my-cnn --timesteps 3e6 --policy cnn  # CNN policy
python -m blueprince.train --name my-v2 --timesteps 3e6 --obs 2        # observation version 2
python -m blueprince.evaluate --agents random rule ppo --ppo-model ppo-cnn
```

Test seeds (1,000,000 and up) are for final comparisons only. Training validates on seeds 500,000 and up.

## Files

| Path | What it is |
|---|---|
| `blueprince/rooms.py` | the 99 rooms |
| `blueprince/game.py` | rules engine (pure Python) |
| `blueprince/env.py` | Gymnasium wrapper and observation encoding |
| `blueprince/agents.py` | random, rule-based and PPO agents |
| `blueprince/train.py` | Maskable PPO training with validation and checkpoints (`--policy mlp` or `cnn`) |
| `blueprince/networks.py` | BoardCNN feature extractor for the CNN policy |
| `blueprince/evaluate.py` | evaluation on fixed seeds, metrics, plot, saving runs |
| `blueprince/replay.py` | rebuilds a game frame by frame from seed + actions |
| `blueprince/dashboard.py`, `dashboard.html` | local dashboard server (standard library only) and page |
| `tests/test_env.py` | sanity cases from the proposal and rule checks |
| `results/` | baseline results (`summary.json`, `summary.png`, per-game CSVs) |
| `results/runs/` | saved evaluation runs, including the test runs of each PPO model |
| `results/training/<run>/` | each training run: `config.json`, `progress.jsonl`, `best_model.zip`, `model.zip`, TensorBoard `tb/` |
| `DESIGN.md` | system design: rules, spaces, reward, seeds, code |
| `EXPERIMENTS.md` | every training experiment, results and next ideas |
| `PROJECT_LOG.md` | timeline, every decision with its reason, results, analysis notes, report material |
| `web/drafting-manor.html` | playable web version (the rule reference) |
| `actions.md`, `rooms.md` | original design notes (Korean) |
