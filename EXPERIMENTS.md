# Experiment log

Every PPO experiment, newest last. Models and curves are in
`results/training/<run>/`; open the dashboard's Training tab to compare them.
Test numbers are on the 1000 test seeds (1,000,000+), start 50 steps / 2 keys /
2 gems, using each run's `best_model.zip`. Baselines on the same seeds:
random 0.0%, rule-based 24.4%.

Validation win rates are on 300 games, so they move by about ±3 points from luck
alone, and the best of many evaluations is biased upward. Compare runs on the
test seeds.

## 2026-11-11 · Exp 1: first MLP runs

Question: does Maskable PPO learn at all with the default observation and
reward, and does a resource curriculum help?

| Run | Setup | Best val (at) | Test win | Test avg rank |
|---|---|---:|---:|---:|
| `ppo-base` | MLP 256×256, 3M steps, 2 keys / 2 gems throughout | 6.7% (2.6M) | 7.5% | 7.69 |
| `ppo-curriculum` | same, keys and gems 6 → 2 over the first 1.5M steps | 10.3% (2.8M) | 7.1% | 7.54 |

Shared settings: 8 envs, lr 3e-4, γ 0.99, GAE 0.95, rollout 256/env, batch 512,
10 epochs, entropy 0.01, clip 0.2, seed 0. About 63 minutes each, run side by side.

Findings
- PPO learns to climb (avg highest rank 3.7 → 7.5) but wins about a third as
  often as the rule-based agent.
- Loss causes vs rule-based: out of steps 19% vs 5.5%; it also drafts fewer
  rooms (18.6 vs 24.3). It wastes steps walking.
- The curriculum made no difference on the test seeds (within noise). Its
  training win rate drops each time the starting resources step down, which is
  expected and not a sign of forgetting.

## 2026-11-11 · Exp 2: CNN board encoder

Question: does a convolutional encoder of the 9×5 board learn better than the
flat MLP with the same observation, reward and settings?

| Run | Setup | Best val (at) | Test win | Test avg rank |
|---|---|---:|---:|---:|
| `ppo-cnn` | BoardCNN 3×64 conv + 256 head, 3M steps, seed 0 | 12.3% (2.9M) | 8.9% | 7.61 |
| `ppo-cnn-s1` | same, seed 1 | 10.7% (1.8M) | 9.0% | 7.75 |

Test loss causes (out of steps / out of keys / goal keys / no doors):
`ppo-cnn` 10.4 / 33.5 / 23.5 / 23.7%; `ppo-cnn-s1` 19.5 / 30.5 / 20.1 / 20.9%.
Rooms drafted 19.5 and 20.3 (MLP 18.6, rule-based 24.3). Saved test runs:
`results/runs/test-ppo-cnn`, `results/runs/test-ppo-cnn-s1`. Each run took about
5.5 hours in the end (both finished about 23:05).

Findings
- The CNN wins about 9% on both seeds vs 7.1–7.5% for the MLP: a gain of about
  1.5 points. With 1000 games the standard error of one win rate is about
  0.9 points, so the gain is small and not yet firm; the MLP has only seed 0.
- Still far below the rule-based agent (24.4%).
- Running out of steps differs a lot between the two CNN seeds (10% vs 20%), so
  that measure depends on the seed as much as on the network.
- Validation best again overstates the test result (12.3% → 8.9%).

Significance check (2026-11-12, paired on the same 1000 test seeds, McNemar
z = (wins only B − wins only A) / √(wins only A + wins only B)):

| A vs B | Wins only A | Wins only B | z |
|---|---:|---:|---:|
| `ppo-base` vs `ppo-cnn` | 63 | 77 | 1.18 |
| `ppo-base` vs `ppo-cnn-s1` | 62 | 77 | 1.27 |
| `ppo-curriculum` vs `ppo-cnn` | 61 | 79 | 1.52 |
| `ppo-curriculum` vs `ppo-cnn-s1` | 63 | 82 | 1.58 |
| `ppo-cnn` vs `ppo-cnn-s1` | 71 | 72 | 0.08 |

No single pair reaches z = 1.96 (5% level). Pooling both runs per network
(179 vs 146 wins in 2000 games) gives about z = 1.9: borderline. Only two
training runs per network, so run-to-run variation is not measured well.
Conclusion: the CNN is at best a small gain, not shown to be real, and costs
3.3× the training time.

The two CNN seeds win almost the same number of games (89 vs 90) but share only
18 of them. Which games a policy wins is mostly luck of the draws, not a stable
skill; the policies have not learned a reliable way to win.

Decision (default): keep the MLP for Exp 3, because it is 3.3× faster and the
CNN gain is not established. The bottleneck looks like missing information
(D10 gaps), not the network.

Seed meaning: `--seed` sets the network's initial weights, the training games
each environment draws, and PPO's action sampling. Validation and test seeds are
the same for every run. Exp 1 used seed 0 only, so an MLP run with seed 1 is
needed for a like-for-like two-seed comparison (about 1 hour).

BoardCNN (`blueprince/networks.py`) reshapes the board block of the same
observation into a 17-channel 9×5 image, adds two coordinate channels, applies
three 3×3 convolutions (64 channels), and joins the result with the global and
blueprint features. Training is about 3.3× slower than the MLP on CPU
(about 235 vs 790 steps/s per run with two runs side by side), so each run takes
about 3–3.5 hours. Started 17:33; settings otherwise identical to Exp 1
(8 envs, 6 torch threads per run).

## 2026-11-12 · Exp 3: hyperparameters and observation v2 (MLP, running)

The user chose to continue with the MLP (3.3× faster than the CNN). Two
questions, run side by side:

**3a. Do the PPO settings matter?** Version 1 observation, one change at a time
from the Exp 1 settings (all seed 0 unless named), plus the missing second MLP seed.

| Run | Change | Best val (at) | Test win | Test avg rank |
|---|---|---:|---:|---:|
| `ppo-base-s1` | none, seed 1 | | | |
| `ppo-ent03` | entropy 0.01 → 0.03 (more exploration) | | | |
| `ppo-ent003` | entropy 0.01 → 0.003 (less exploration) | | | |
| `ppo-lr1e4` | learning rate 3e-4 → 1e-4 | | | |

**3b. Does the missing information matter?** Observation version 2 (867 numbers,
`DESIGN.md` section 3): goal distance and step slack, keys short, usable doors,
key/step/gem rooms left in the deck, and per offer what placing it does (new,
joined and walled doors, joins the goal, doors left on the board, rooms to the
goal, lock chance). Exp 1 settings otherwise.

| Run | Setup | Best val (at) | Test win | Test avg rank |
|---|---|---:|---:|---:|
| `ppo-obs2` | obs v2, seed 0 | | | |
| `ppo-obs2-s1` | obs v2, seed 1 | | | |

Settings: 8 envs, 3 torch threads per run, 3M steps. Five runs started together
at about 22:00 on 2026-11-12. `ppo-obs2` failed at start with a MemoryError (six
runs at once exhaust the Windows commit memory: each run is 9 Python processes
that load torch); it is queued to start when the first 3a run finishes.

Expectation: if the PPO settings were the bottleneck, one of the 3a runs should
move clearly above 7–9%; if missing information is, 3b should.

**Stopped by the user (2026-11-12, about 23:00)** before any run reached 3M
steps; `ppo-obs2` (seed 0) was never started. Models up to the stop point are in
each run's folder (`model.zip` from the last evaluation, `best_model.zip`), and
`config.json` says `stopped`. Not evaluated on the test seeds.

Interim comparison at the same point (up to 1.2M steps). Training win rate is
over the last 1000 training games, so it is less noisy than validation (300 games):

| Run | Stopped at | Best val up to 1.2M | Train win at 1.2M |
|---|---:|---:|---:|
| `ppo-base` (Exp 1) | 3M | 3.7% | 3.2% |
| `ppo-cnn` / `ppo-cnn-s1` (Exp 2) | 3M | 5.0% / 7.3% | 3.0% / 4.2% |
| `ppo-base-s1` | 1.2M | 5.7% | 2.7% |
| `ppo-ent03` | 1.5M | 4.7% | 2.4% |
| `ppo-ent003` | 1.4M | 4.7% | 2.4% |
| `ppo-lr1e4` | 1.4M | 5.3% | 4.4% |
| `ppo-obs2-s1` | 1.2M | **9.0%** | **8.5%** |

Early read (one seed, half-trained, not a test result): observation v2 learns
faster than every v1 run, about twice the training win rate at 1.2M steps. The
PPO setting changes stay within the v1 band. This points to missing information
as the bottleneck (D21), but needs the full runs, a second seed and the test
seeds to confirm. To resume: rerun with the same commands (a new run starts from
scratch; there is no resume from a checkpoint).

## Next candidates

1. **Observation additions**: count of key-giving (and step-giving) rooms left in
   the deck; fewest rooms still needed to reach the Antechamber; keys still needed
   for the goal (3 + locked doors on the best path); and per-offer consequences of
   placing it (new open doors, joins, walled doors, connects to the Antechamber,
   open doors left). The draft offers currently give only door directions.
2. **Reward shaping**: nothing rewards keeping 3 keys for the end or penalises
   walking. Options: small penalty per step walked, bonus for reaching rank 8–9
   with 3+ keys. Keep the total well below the win reward.
3. **Policy head that follows the board**: door actions are per cell and side, so
   a convolutional head (4 logits per cell) instead of a dense head.
4. **Longer training** (10M+ steps) and more seeds per setup.
