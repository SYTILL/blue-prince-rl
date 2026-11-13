# Design: Minimal Blue Prince environment and baselines

CS 175 Team 10. Status as of 2026-11-11: the environment, the random and
rule-based agents, the evaluation script, the dashboard and the Maskable PPO
training pipeline are implemented and tested. First PPO training runs are in
`results/training/` (section 7).

This document is the single reference for the code in `blueprince/`. It is
written so that a teammate, or an AI assistant given this file, can work on the
project without reading the Korean notes (`actions.md`, `rooms.md`).

## 1. The game

The player builds a manor on a grid of 9 ranks by 5 columns and must reach the
Antechamber.

- **Cells** are `(r, c)`: `r` = rank 1..9 (1 is the bottom), `c` = column 1..5 (A..E).
- **Directions** are integers: N=0, E=1, S=2, W=3.
- **Fixed rooms**: Entrance Hall at C1 = `(1, 3)` with doors W, N, E (the start);
  Antechamber at C9 = `(9, 3)` with doors W, S, E (the goal).
- **Start resources**: 50 steps, 2 keys, 2 gems (`Config`).
- **Deck**: 99 rooms, one copy each. A room has a shape, a gem cost, a rarity and
  the steps / keys / gems it grants once, when placed. No other room effects.

A turn has two decisions:

1. **Open a door.** Pick an unopened door of any room you can walk to. The player
   walks there by the shortest path (1 step per room entered). A locked door
   costs 1 key.
2. **Draft.** Three blueprints are drawn for the cell behind the door. Pick one
   you can afford. It is placed, the player enters it (1 step) and receives its
   resources.

### Rules in detail

| Rule | Detail |
|---|---|
| Shapes | Dead end (1 door), Straight (2, opposite), L (2, corner; each L room turns a fixed way, left or right, seen from the entry door), T (3), 4-way |
| Placement | A room is only offered where none of its doors would lead outside the grid. T rooms get a random valid orientation; other shapes have one |
| Drawing | Slot 0 is a free room when the deck has one that fits, and it always costs 0. Rarity weights are Commonplace 50, Standard 30, Unusual 15, Rare 5: a rarity is chosen by weight among those available, then a room of that rarity. No duplicates. Never three dead ends if another shape fits |
| Locks | When a door to an empty cell appears, it is locked with a probability set by the higher rank of its two cells: ranks 1–3 0%, 4 25%, 5 45%, 6 60%, 7 70%, 8 90%, 9 100%. Locks are visible as soon as the door exists |
| Connecting | If a new room's door meets a door of an existing room, the two join and are open, locked or not. If it meets a wall, that door is blocked |
| Blocked door | If no room in the deck fits behind a door, the door is marked dead and the player stays in the door phase (the key, if any, is still spent). This needs the deck to run out of dead ends, so it is rare |
| Antechamber | Cannot be opened from outside. A room placed next to it with a door facing it creates the path. Entering costs 3 keys |
| Steps | `steps = max(0, steps + room.steps)` after placing; some rooms take steps away |

### Differences from the web version

`web/drafting-manor.html` is the rule reference. The Python engine differs in four ways:

1. **Randomness is keyed by position** (section 5). A seed does not give the same
   game as the same seed in the web version.
2. **Walking is not an action.** The web player can walk freely; here the player
   only walks as part of opening a door.
3. **Entering the Antechamber is automatic**: after any action, if a path to it
   exists, `distance <= steps` and `keys >= 3`, the player walks in and wins.
4. **The Antechamber cannot be walked through** on the way to somewhere else.

## 2. Action space

`Discrete(183)`, with an action mask. Most actions are illegal at any moment.

| Index | Meaning | Phase |
|---|---|---|
| 0–179 | open door `d` of the room at `(r, c)`: `index = ((r - 1) * 5 + (c - 1)) * 4 + d` | door |
| 180–182 | take blueprint 0, 1 or 2 | draft |

Example: the Entrance Hall's doors are north 8, east 9, west 11.

A door action is legal when all of these hold: the game is in the door phase;
the room at `(r, c)` has door `d`; the cell behind it is inside the grid and
empty; the room is not the Antechamber; the door is not marked dead; the player
can walk to `(r, c)`; `steps >= distance + 1`; and the door is unlocked or
`keys >= 1`.

A blueprint action is legal when the game is in the draft phase, that slot holds
an offer, and `gems >= cost`. Slot 0 is always free, so a draft always has a
legal action.

Helpers: `encode_door(r, c, d)`, `decode_door(action)` in `game.py`.

## 3. Observation

A flat `float32` vector of length **834**, values in `[-1, 1]`. Built by
`encode_observation(game)` in `env.py`. Constants there give the offsets.

**Board: 45 cells × 17 features** (cell index = `(r - 1) * 5 + (c - 1)`, offset = index × 17)

| Feature | Meaning |
|---|---|
| 0 | a room is here |
| 1–4 | the room has a door N, E, S, W |
| 5–8 | that door is unopened and unlocked (N, E, S, W) |
| 9–12 | that door is unopened and locked (N, E, S, W) |
| 13 | the player is here |
| 14 | reachable on foot from the player |
| 15 | walking distance from the player / 20 |
| 16 | this is the cell being drafted (draft phase only) |

**Global: 12 features** (offset 765)

| Feature | Meaning |
|---|---|
| 0–3 | steps / 100, keys / 10, gems / 10, highest rank reached / 9 |
| 4–5 | door phase, draft phase (one-hot) |
| 6 | rooms left in the deck / 99 |
| 7–11 | fraction of each shape (D, S, L, T, X) still in the deck |

**Offers: 3 slots × 19 features** (offset 777; all zero in the door phase)

| Feature | Meaning |
|---|---|
| 0 | the slot holds an offer |
| 1–4 | doors N, E, S, W after orientation |
| 5–6 | gem cost / 5, affordable |
| 7–10 | rarity one-hot (C, S, U, R) |
| 11–13 | steps / 10, keys / 3, gems / 3 granted |
| 14–18 | shape one-hot (D, S, L, T, X) |

This is version 1, used by Exp 1–2. Known gaps: it does not say which individual
rooms are still in the deck, and the board is flat rather than a 9×5 image. To
change it, wrap the environment (`gymnasium.ObservationWrapper`) or edit
`encode_observation`; the rules engine does not depend on it.

### Version 2 (Exp 3): 867 numbers

`BluePrinceEnv(obs_version=2)` or `train.py --obs 2`. The first 834 numbers are
exactly version 1; 33 computed features follow. They hand the agent what the
rule-based agent computes for itself. `PPOAgent` reads the version from the
model's input size, so version 1 models keep working.

"Rooms still needed" from an empty cell = 1 + Manhattan distance to the nearest
cell beside an Antechamber door ((9,2), (8,3), (9,4)); a lower bound that ignores
walls and shapes.

**Plan: 12 features** (offset 834)

| Feature | Meaning |
|---|---|
| 0–1 | Antechamber reachable on foot; walking distance to it / 20 |
| 2 | fewest steps to win through an open door: walk + rooms still needed + 1 (or the walk if connected) / 20 |
| 3 | step slack: (steps − feature 2) / 50 |
| 4 | fewest rooms still needed / 10 |
| 5 | keys short of the 3 needed: (3 − keys) / 3 |
| 6–7 | usable doors / 10, reachable locked doors / 10 |
| 8–11 | rooms in the deck giving keys / 10, steps / 10, gems / 10; total keys in the deck / 20 |

**Offer placement: 3 slots × 7 features** (offset 846; zero in the door phase)

| Feature | Meaning |
|---|---|
| 0 | new open doors (to an empty cell) / 3 |
| 1 | doors that join a neighbour, not counting the entry door / 3 |
| 2 | doors walled off by a neighbour with no matching door / 3 |
| 3 | joins the Antechamber |
| 4 | unopened doors on the board after placing it / 10 (0 = the board is closed) |
| 5 | rooms still needed from its best new door / 10 (0 if it joins the goal, 1 if no new door) |
| 6 | average lock chance of its new doors |

Encoding version 2 slows the environment by about 20% (2250 → 1770 steps/s
with random actions).

## 4. Reward and episode end

| Event | Reward |
|---|---|
| Reaching rank `k` for the first time (k = 2..9) | `2**(k-1) / 256` (sum over all ranks = 510/256 ≈ 1.99) |
| Entering the Antechamber | +10 |
| Anything else, including losing | 0 |

The rank rewards total less than a fifth of the win reward. Both values are in
`Config` (`milestone_scale`, `win_reward`).

- **Win** (`terminated`): the automatic Antechamber entry.
- **Loss** (`terminated`): no legal action in the door phase. `info["loss_cause"]` is one of:
  - `out_of_steps`: no steps left, or every door is too far for the steps left
  - `out_of_keys`: a door is in reach but every such door is locked and there are no keys
  - `goal_keys`: a path to the Antechamber exists but fewer than 3 keys, and nothing else to open
  - `no_doors`: no unopened door can be reached at all
- **Truncation** (`truncated`): 300 actions. This is a safeguard; no game in the tests reaches it.

## 5. Randomness and seeds

The proposal says every agent faces the same games on a given seed. A single
random stream cannot give that, because the first different action shifts every
later draw. So randomness is keyed by position:

- **Locks**: a door's lock is a function of `(seed, edge)` only. Two agents on the
  same seed see exactly the same lock on every door they both uncover (tested).
- **Drafts**: every room gets fixed random numbers per `(seed, cell)`, and the
  offers are the best-ranked rooms in the deck (weighted sampling without
  replacement, with weights matching the rarity rule). Two agents get the same
  offers at a cell unless one already used an offered room, or they enter the cell
  from different sides, which changes which rooms fit. In a test where the decks
  differ by one unrelated room, over 90% of drafts are identical.
- Slot 0 and slots 1–2 use independent numbers. Sharing them made gem-cost rooms
  appear about 25% more often than in the web version's draw.

So games on one seed are the same world, not the same sequence of events. The
report should describe it this way.

**Seed handling**: `env.reset(seed=s)` plays game `s`. `env.reset()` with no seed
draws the next game seed from the environment's generator, so a training run is
reproducible from its first seed. The **test seeds are 1,000,000 to 1,000,999**
(`TEST_SEED_START` in `evaluate.py`). Do not train or tune on them. The
rule-based agent was tuned on seeds 0–299 and checked on 5000–5999.

## 6. Code

```
blueprince/
  rooms.py      ROOMS: tuple of Room(name, shape, turn, rarity, cost, steps, keys, gems, red)
  game.py       Game, Config, KeyedRng, orientation helpers
  env.py        BluePrinceEnv (gymnasium.Env), encode_observation
  agents.py     RandomAgent, RuleBasedAgent, PPOAgent, AGENTS registry, ppo_models()
  train.py      Maskable PPO training (command line)
  networks.py   BoardCNN: convolutional encoder of the 9x5 board for --policy cnn
  evaluate.py   run_episode, evaluate, summarize, save_run, load_rows, command line
  replay.py     replay(seed, actions, config) -> list of JSON frames for viewing
  dashboard.py  local HTTP server for the dashboard (standard library only)
  dashboard.html  the dashboard page (no external libraries)
tests/test_env.py
```

### `Game` (game.py)

The rules engine. It has no Gymnasium or NumPy dependency.

```python
g = Game(seed, Config())
g.legal_actions()      # list of legal action indices
g.action_mask()        # list of 183 bools
reward = g.step(a)     # raises ValueError on an illegal action
g.render()             # text board
```

State an agent may read: `rooms` (`{cell: Placed(name, doors, room_index)}`),
`locks` (`{edge: bool}`), `player`, `steps`, `keys`, `gems`, `deck` (set of
indices into `ROOMS`), `max_rank`, `phase` (`PHASE_DOOR` / `PHASE_DRAFT`),
`draft` (`Draft(origin, d, target, offers)` with `Offer(room, doors, cost)`),
`over`, `win`, `truncated`, `loss_cause`, `actions` (history).
Queries: `bfs()` → `(dist, prev)` from the player, `frontier()` → unopened doors
as `(r, c, d, edge, locked)`, `connected(cell, d)`.

`(seed, config, actions)` replays a game exactly (tested).

### `BluePrinceEnv` (env.py)

```python
env = BluePrinceEnv(config=None)
obs, info = env.reset(seed=123)
mask = env.action_masks()                 # np.ndarray of 183 bools
obs, reward, terminated, truncated, info = env.step(action)
print(env.render())
env.game                                  # the underlying Game
```

`info` keys: `seed`, `phase`, `win`, `loss_cause`, `max_rank`, `steps`, `keys`,
`gems`, `steps_spent`, `keys_used`, `gems_used`, `rooms_drafted`, `n_actions`.

`step` raises `ValueError` on an illegal action instead of ignoring it. For that
reason `gymnasium.utils.env_checker.check_env`, which samples unmasked actions,
fails on this environment by design.

Speed: about 4,000 steps per second on one core, including observation and mask.

### Agents (agents.py)

An agent is any object with:

```python
agent.name                # label used in tables
agent.reset(seed)         # called before each game
agent.act(env) -> int     # a legal action for env's current state
```

- **`RandomAgent`**: uniform over legal actions.
- **`RuleBasedAgent`**: scores every legal action and takes the best. No look-ahead.
  - Door phase: prefers doors whose next cell is closer to the Antechamber, minus
    walking distance, minus a penalty for locked doors, minus a large penalty if
    the Antechamber is out of reach with the steps left. While it has fewer than
    `key_target` keys and spare steps, distance to the goal counts less, so it
    opens nearby doors first.
  - Draft phase: values keys, steps and gems granted (net of cost) and new doors,
    more for doors leading toward the goal; takes a room that connects to the
    Antechamber; never closes the last open door.
  - It does not read hidden information (locks of doors that do not exist yet, or
    future draws). Any new agent must follow the same rule.

- **`PPOAgent`**: a trained Maskable PPO policy, acting greedily
  (`deterministic=True`) on `encode_observation(env.game)` with the action mask.
  `PPOAgent()` loads the newest run's `best_model.zip`; `PPOAgent("ppo-base")`
  loads that run under `results/training/`; a `.zip` path or
  `$BLUEPRINCE_PPO_MODEL` also works. torch is only imported when a PPO agent is
  created, so the baselines run without it.

### Training (train.py)

```
python -m blueprince.train --name ppo-base --timesteps 3e6
python -m blueprince.train --name ppo-curriculum --timesteps 3e6 --cur-keys 6 --cur-gems 6 --cur-frac 0.5
tensorboard --logdir results/training          # optional; the dashboard shows the same curves
```

- `MaskablePPO("MlpPolicy")` from sb3-contrib on `--n-envs` parallel environments
  (`SubprocVecEnv`). The mask comes from `env.action_masks()`.
- Defaults: lr 3e-4, γ 0.99, GAE λ 0.95, rollout 256 per env, batch 512,
  10 epochs, entropy 0.01, clip 0.2, two hidden layers of 256 for both policy
  and value networks. Every one is a command-line flag (`--help`).
- **Policy network** (`--policy`): `mlp` (default) runs dense layers on the flat
  834-vector. `cnn` uses `BoardCNN` from `networks.py`: the board block of the
  same vector becomes a 17-channel 9×5 image plus 2 coordinate channels, then
  `--cnn-layers` 3×3 convolutions with `--cnn-channels` channels (default 3 × 64),
  joined with the global and blueprint features, then a 256-unit head. Both read
  the same observation, so results are directly comparable. CNN training is about
  3× slower on CPU.
- **Experiment log**: `EXPERIMENTS.md` records every run, its settings and its
  test result.
- **Validation**: every `--eval-every` steps (default 100k), the current policy
  plays `--eval-games` (default 300) games on validation seeds 500,000+ at the
  target difficulty. The random and rule-based agents are scored on the same
  seeds once at the start, as baselines. Validation seeds are separate from the
  test seeds.
- **Curriculum** (optional): `--cur-keys` / `--cur-gems` set the starting keys and
  gems at the beginning; they move linearly to the target (`--keys`, `--gems`,
  default 2 / 2) over the first `--cur-frac` of training, through
  `BluePrinceEnv.set_config`. Validation always uses the target.
- **Output** in `results/training/<name>/`: `config.json` (arguments, baselines,
  status, best validation win rate), `progress.jsonl` (one line per evaluation:
  timesteps, elapsed time, curriculum stage, recent training games, validation
  metrics), `model.zip` (latest), `best_model.zip` (best validation win rate),
  `tb/` (TensorBoard).
- Speed on this machine: about 1,100 steps per second with 8 environments
  including updates; two runs side by side get about 720 each.

## 7. Evaluation

`python -m blueprince.evaluate --out results --plot` runs each agent on the test
seeds and writes `episodes_<agent>.csv` (one row per game), `summary.json` and
`summary.png`. `--steps`, `--keys`, `--gems` change the starting resources.

Each CSV row holds the game's metrics and an `actions` column (space-separated
action indices). With the seed and the config in `summary.json`, that replays the
game exactly, so every saved game can be inspected later. `load_rows` reads a CSV
back.

### Dashboard

`python -m blueprince.dashboard` serves a local page at http://127.0.0.1:8175.

- **Run**: pick agents, number of games, first seed and starting resources. Games
  stream to the page as they finish (`/api/stream`, server-sent events), with agents
  interleaved seed by seed. Each run is saved to `results/runs/<time>/` in the same
  format as above; a stopped run is saved with `"partial": true`.
- **Load**: any saved run (`results/` from the command line, or `results/runs/*`).
- **Views**: win rate per agent with a 95% interval, running win-rate curve,
  highest-rank histogram, outcome breakdown by loss cause, and a filterable game list.
- **Replay**: click a game to rebuild it with `replay()` (`/api/replay`). It shows
  the board after each action (rooms, door states, the path walked, the cell being
  drafted), the three blueprints offered and the one chosen, resources over time
  and the move log. When several agents played the same seed, tabs switch between
  them. Keys: ← → step, space play/pause.
- **PPO**: `ppo` appears as an agent once a trained model exists, with a selector
  for which training run's model to use. The run's summary records the model.
- **Training tab** (`#train`): pick a training run to see best and latest
  validation win rate against the baselines, progress, validation and recent
  training win rate and highest rank over timesteps, all runs' validation curves
  on one chart, how validation games end, and the settings. It refreshes every
  5 s while a run is training.

A new agent appears in the dashboard once it is added to `AGENTS`.
`python -m blueprince.evaluate --agents random rule ppo --ppo-model <run>` gives
the same comparison on the command line.

Baseline results on the 1000 test seeds (50 steps, 2 keys, 2 gems):

| Agent | Win rate | Avg highest rank | Out of steps | Out of keys | Goal keys | No doors |
|---|---:|---:|---:|---:|---:|---:|
| Random | 0.0% | 3.69 | 8.2% | 4.1% | 0.0% | 87.7% |
| Rule-based | 24.4% | 8.00 | 5.5% | 25.4% | 23.8% | 20.9% |

With 5 keys and 5 gems at the start, the rule-based agent wins 58.6% and the
random agent still 0.0%.

First PPO runs (3M steps each, default settings, `best_model.zip` by validation
win rate), same 1000 test seeds:

| Agent | Win rate | Avg highest rank | Out of steps | Out of keys | Goal keys | No doors |
|---|---:|---:|---:|---:|---:|---:|
| PPO `ppo-curriculum` (keys/gems 6 → 2 over the first half) | 7.1% | 7.54 | 19.1% | 26.9% | 20.1% | 26.8% |
| PPO `ppo-base` (2 / 2 throughout) | 7.5% | 7.69 | 18.9% | 29.4% | 18.3% | 25.9% |

PPO learns to climb (rank 3.7 → 7.5–7.7) but wins far less than the rule-based
agent. The best validation scores (10.3% and 6.7% on 300 games) overstate it:
picking the best of 30 noisy evaluations favours a lucky one. Compared with the
rule-based agent it runs out of steps much more often (19% vs 5.5%) and drafts
fewer rooms (18.6 vs 24.3). The curriculum made no clear difference on the test
seeds.

What the rule-based agent's losses show: about half are key shortages (blocked
by locks, or at the Antechamber with fewer than 3 keys), and a fifth are boards
it closed off itself. Those are the places where an agent that plans ahead
should gain.

### Sanity cases from the proposal

| Case | Where it is checked |
|---|---|
| No agent takes an illegal action | `step` raises on one; 200 random games and all evaluations run clean |
| Every game ends | 200 random games end by termination, none by the 300-action limit |
| Same seed and actions give the same result | `test_same_seed_and_actions_replay_exactly` |
| Extra resources raise win rates | `test_agents_rank_as_expected_and_resources_help`; 24.4% → 58.6% above |

## 8. Open items

- **PPO tuning**: reward shaping, γ (0.99 or 1.0), observation changes, network
  size, longer training.
- **Rule choices kept from the web version** and still open in `rooms.md`:
  resources are granted once when a room is placed; no special rules for Corridor,
  Foyer or Great Hall; no penalty for a blocked door.
- **Proposal wording**: section 4 says each seed fixes the draws. See section 5 for
  what is actually fixed.
