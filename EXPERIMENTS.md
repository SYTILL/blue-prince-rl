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

## 2026-11-12 · Exp 3: hyperparameters and observation v2 (MLP)

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
| `ppo-obs2` | obs v2, seed 0 | 14.3% (2.8M) | **11.4%** | 7.27 |
| `ppo-obs2-s1` | obs v2, seed 1 | stopped at 1.2M (9.0% at 1.0M) | | |
| `ppo-obs2-s1b` | obs v2, seed 1, rerun from scratch | 14.7% (2.7M) | **13.0%** | 7.78 |

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

**Rerun of 3b only (user, 2026-11-13 18:00).** `ppo-obs2` (seed 0) and
`ppo-obs2-s1b` (seed 1, new name because runs cannot resume) started side by
side, 3M steps, 8 envs, 6 torch threads each. The 3a runs are not rerun: at
1.2M steps they were no better than the defaults.

**Result of 3b (2026-11-13 19:20).** Both runs finished in about 67 minutes.

Test loss causes (out of steps / out of keys / goal keys / no doors):
`ppo-obs2` 12.8 / 23.0 / 22.2 / 30.6%; `ppo-obs2-s1b` 18.1 / 26.7 / 22.4 / 19.8%.
Rooms drafted 18.4 and 19.1. Saved test runs: `results/runs/test-ppo-obs2`,
`results/runs/test-ppo-obs2-s1b`.

Paired McNemar test on the 1000 test seeds (wins only by the older model /
wins only by v2, z):

| Older model | vs `ppo-obs2` (114 wins) | vs `ppo-obs2-s1b` (130 wins) |
|---|---:|---:|
| `ppo-base` MLP v1 (75) | 60 / 99, z 3.09 | 58 / 113, z 4.21 |
| `ppo-curriculum` MLP v1 (71) | 57 / 100, z 3.43 | 54 / 113, z 4.57 |
| `ppo-cnn` CNN v1 (89) | 67 / 92, z 1.98 | 66 / 107, z 3.12 |
| `ppo-cnn-s1` CNN v1 (90) | 75 / 99, z 1.82 | 69 / 109, z 3.00 |

The two v2 seeds do not differ significantly from each other (87 / 103, z 1.16)
and share 27 wins.

Findings
- Observation v2 raises the MLP from 7.1–7.5% to 11.4–13.0% on the test seeds,
  about +5 points, significant on every pairing with the MLP v1 runs (z 3.1–4.6).
  This is the first change that clearly helps.
- It is also better than the CNN with the old observation (z 1.8–3.1), while
  training 3.3× faster. Information mattered more than the network.
- With the setting changes of 3a showing no gain at 1.2M steps, this supports
  D21: the bottleneck was missing information, not the PPO settings.
- Still about half the rule-based agent's 24.4%. Key-related losses (out of keys
  plus goal keys) remain the largest group at 45–49%, and v2 still drafts fewer
  rooms (18–19) than the rule-based agent (24).

## 2026-11-13 · Exp 4: longer training of observation v2 (stopped at 6.6M)

Question: were the observation v2 runs still improving at 3M steps? Both reached
their best validation near the end (2.7–2.8M).

`ppo-obs2` and `ppo-obs2-s1b` resumed from their 3M `model.zip` with the new
`--resume` option (D23) and train to 10M steps in total. Settings are unchanged;
the environments are seeded with seed + 10000 so the training games are new.
Started 20:04, side by side, 6 torch threads each. Expect about 2.5–3 hours.

The 3M best models evaluated in Exp 3b are kept as `best_model_3M.zip` (copied
before the resume). Test them with
`--ppo-model results/training/<run>/best_model_3M.zip`.

Stopped by the user at 21:29, at 6.6M steps each, because validation had
flattened. `model.zip` holds the 6.6M weights, so `--resume <run> --timesteps 10e6`
can still finish them.

Mean validation win rate per million steps (each evaluation is 300 games, so a
single value moves about ±2 points by chance):

| Steps | `ppo-obs2` | `ppo-obs2-s1b` |
|---|---:|---:|
| 1–2M | 7.8% | 8.6% |
| 2–3M | 10.6% | 11.1% |
| 3–4M | 12.2% | 11.7% |
| 4–5M | 11.5% | 12.5% |
| 5–6.6M | 13.1% | 12.0% |

| Run | Best val (at) | Test win | Test avg rank |
|---|---:|---:|---:|
| `ppo-obs2` → 6.6M | 16.7% (5.0M, again at 6.6M) | not run | |
| `ppo-obs2-s1b` → 6.6M | 15.7% (5.7M) | not run | |

Findings:
- The gain stops at about 3M steps. From 3M to 6.6M the mean stays near 12%,
  for both seeds. More than double the training gave almost nothing.
- The higher best values (16.7%, 15.7%) are most likely the max of many noisy
  300-game evaluations, not real progress. The test seeds would tell; the new
  `best_model.zip` files have not been tested yet.
- So with this observation and reward, the MLP is near its limit. Together with
  Exp 2 and 3a this points again at the information and the reward, not at
  training length, the network or the PPO settings.

## 2026-11-16 · Exp 5: potential-based reward shaping (stopped at 1.7–1.8M)

Question: does denser feedback about keys, steps and dead ends help the
observation v2 MLP get past ~12%? Design and reasons: PROJECT_LOG D24.

Shaping adds γΦ(s′) − Φ(s) to every reward (γ = 0.99, same as PPO), with
Φ = w_keys·min(keys, 5) + w_steps·steps/50 + w_doors·min(reachable unopened doors, 4),
and Φ = 0 when the game ends. Everything else is as in Exp 3b (`--obs 2 --net 256,256
--n-envs 8`, 3M steps), so the baselines are `ppo-obs2` / `ppo-obs2-s1b` at 3M
(test 11.4% / 13.0%, `best_model_3M.zip`). Four runs side by side, 4 torch threads each.

| Run | Seed | w_keys | w_steps | w_doors | Best val (at) | Test win | Test avg rank |
|---|---:|---:|---:|---:|---:|---:|---:|
| `ppo-shape-ks` | 0 | 0.2 | 0.5 | 0 | 10.3% (1.8M) | not run | |
| `ppo-shape-ks-s1` | 1 | 0.2 | 0.5 | 0 | 11.3% (1.8M) | not run | |
| `ppo-shape-ksd` | 0 | 0.2 | 0.5 | 0.1 | 9.7% (1.4M) | not run | |
| `ppo-shape-ksd-s1` | 1 | 0.2 | 0.5 | 0.1 | 10.7% (1.3M) | not run | |

Stopped by the user at 17:34 (1.7–1.8M steps of 3M); started 16:30. `model.zip`
holds the last weights, so `--resume <run> --timesteps 3e6` finishes them.

Mean validation win rate, same step ranges as the unshaped baselines:

| Steps | ks s0 | ks s1 | ksd s0 | ksd s1 | `ppo-obs2` | `ppo-obs2-s1b` |
|---|---:|---:|---:|---:|---:|---:|
| 0.5–1M | 4.5% | 5.7% | 6.0% | 3.9% | 3.0% | 7.1% |
| 1–1.8M (baselines 1–2M) | 7.1% | 8.3% | 7.6% | 8.3% | 7.8% | 8.6% |

Interim finding: up to 1.8M, shaping does not learn faster than no shaping; all
six runs are within the ±2-point noise of a 300-game evaluation. It could still
matter later, but nothing suggests a large effect. Loss causes were not analysed.

Note (2026-11-16): rule-based loses as many games on keys as PPO; the gap to
rule-based is steps and early dead ends (PROJECT_LOG section 4). Judge any rerun
on `out_of_steps` and early `no_doors` as well as win rate.

## Next candidates

1. ~~**Observation additions**~~: done as observation v2 (Exp 3b). Still open:
   keys needed for the locks on the best route.
2. **Reward shaping** (Exp 5, stopped at 1.8M, no early gain): nothing rewards keeping 3 keys for the end or penalises
   walking. Options: small penalty per step walked, bonus for reaching rank 8–9
   with 3+ keys. Keep the total well below the win reward. Targets the key losses
   (45–49% of games for observation v2).
3. **Policy head that follows the board**: door actions are per cell and side, so
   a convolutional head (4 logits per cell) instead of a dense head.
4. ~~**Longer training**~~: Exp 4, flat after 3M. More seeds per setup is still useful.
5. **Imitation of the rule-based agent, then PPO**: rule-based loses as many
   games on keys (49%) as PPO; the gap is steps and early dead ends (PROJECT_LOG
   section 4, 2026-11-16). Behaviour cloning from rule-based games would start PPO
   from its step and room choices.
