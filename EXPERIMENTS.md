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
