# Project log: progress, decisions, directions

Running record for writing the report. Newest entries go at the end of each
section. Related files:

- `DESIGN.md`: how the system works now (rules, spaces, code).
- `EXPERIMENTS.md`: each training run, its settings and results.
- This file: what was done when, what was decided and why, and where the report
  material is.

---

## 1. Timeline

| Date | Proposal milestone | Done |
|---|---|---|
| Weeks 1–4 | Proposal | Proposal submitted. Rules drafted in Korean notes (`actions.md`, `rooms.md`) and a playable web version (`web/drafting-manor.html`) |
| 2026-11-11 (week 7) | Weeks 5–6: environment | Python rules engine ported from the web version; Gymnasium environment with action masking; 211 tests. The team was about 1–2 weeks behind the proposal schedule before this day |
| 2026-11-11 | Week 7: alpha | Random and rule-based agents; evaluation on 1000 test seeds; sanity cases checked. Rule-based 24.4%, random 0.0% |
| 2026-11-11 | (extra) | Local dashboard: live evaluation, saved runs, step-by-step game replay. Episodes now store their action list so every game can be replayed |
| 2026-11-11 | Weeks 8–9: beta | Maskable PPO training pipeline, PPO agent, dashboard Training tab. Experiment 1 (MLP, with and without curriculum): about 7% test win rate |
| 2026-11-11 | Weeks 8–9: beta | Experiment 2 started: CNN board encoder, two seeds |
| 2026-11-11 23:00 | Weeks 8–9: beta | Exp 2 at 2.8M of 3M steps: best validation 12.0% / 10.7%, recent values in the same band as the MLP. Test evaluation pending. Room colour discussed and not added (see D19) |
| 2026-11-11 23:10 | Weeks 8–9: beta | Exp 2 done: CNN 8.9% / 9.0% on the test seeds vs MLP 7.1–7.5%. Code and results pushed to GitHub for the team (D20) |
| 2026-11-12 | Weeks 8–9: beta | Exp 2 significance check (not significant). Observation v2 implemented (219 tests). Exp 3 started: PPO setting changes (3a) and observation v2 (3b), MLP | 
| 2026-11-12 23:00 | Weeks 8–9: beta | Exp 3 stopped by the user at 1.2–1.5M of 3M steps. Interim: obs v2 trains about twice as well as any v1 run at 1.2M (train win 8.5% vs 2.4–4.4%); PPO setting changes no different from v1 |
| 2026-11-13 18:00 | Weeks 8–9: beta | Only observation v2 rerun (user): `ppo-obs2` and `ppo-obs2-s1b`, two seeds, 3M steps |
| 2026-11-13 19:20 | Weeks 8–9: beta | Exp 3b done: observation v2 wins 11.4% / 13.0% on the test seeds vs 7.1–7.5% for MLP v1 (McNemar z 3.1–4.6) and 8.9–9.0% for the CNN. First clear gain |
| 2026-11-13 20:04 | Weeks 8–9: beta | `--resume` added to training. Exp 4: both observation v2 runs continue from 3M to 10M steps |
| 2026-11-13 21:29 | Weeks 8–9: beta | Exp 4 stopped by the user at 6.6M steps: validation flat near 12% after 3M for both seeds. Longer training does not help; next is reward shaping for keys |

---

## 2. Decisions

Each entry: what was decided, the options, why. "User" means the team decided;
"default" means it was set to a reasonable value and can still be changed.

### Game rules

**D1. Rules follow the web version** (default)
The Python engine is a port of `web/drafting-manor.html`: lock chance by rank
(0, 0, 0, 25, 45, 60, 70, 90, 100%), rarity weights 50/30/15/5, slot 0 always
free, never three dead ends, 3 keys to enter the Antechamber, rank milestone
reward `2^(k-1)/256` and win reward +10. The open items in `rooms.md` were set
to the web version's choices: resources are granted once when a room is placed;
no special rules for Corridor, Foyer or Great Hall; no penalty for a blocked door.
Why: the web version was already played and checked, and copying it avoids
reopening settled rules.

**D2. Walking is not an action; entering the Antechamber is automatic** (proposal)
As written in the proposal and `actions.md`. Opening a door walks there by the
shortest path. Entering the Antechamber is always the best move once possible,
so it is not a choice.

**D3. The Antechamber cannot be walked through** (default; differs from the web version)
In the web version a player with fewer than 3 keys could walk through the
Antechamber to reach rooms beyond it. Here the goal is a terminal room only.
Why: walking through the goal makes no sense in the game's terms and would give
the agent a shortcut the proposal does not describe.

**D4. Blocked door keeps the key spent** (default, same as the web version)
If no room in the deck fits behind an opened door, the door is marked dead; a key
used on it is not refunded. Rare in practice.

**D5. Illegal actions raise an error instead of being ignored** (default)
Why: with masking an illegal action can only come from a bug, and an error finds
it at once. Side effect: Gymnasium's generic `check_env` (which samples unmasked
actions) does not apply.

**D6. Truncation at 300 actions** (from `actions.md`)
A safety limit. No game in any test or evaluation has reached it.

### Randomness and seeds

**D7. Randomness keyed by position** (user)
Options: (a) one random stream as in the web version, so seeds match the web
game; (b) randomness keyed by position; (c) both behind a switch.
Chosen: (b). A door's lock depends only on (seed, door); a draft ranks the deck by
numbers that depend only on (seed, cell, room).
Why: the proposal claims every agent faces the same games on a seed. With one
stream the first different action changes every later draw, so the claim fails.
Cost: a seed does not give the same game as in the web version.
What it achieves (tested): locks are identical for every agent; offers at a cell
are identical unless an agent already used an offered room or enters from another
side (over 90% identical when decks differ by one unrelated room).
For the report: describe seeds as "the same world", not "the same sequence of
events"; proposal section 4 needs this correction.

**D8. Implementation of the keyed draw, and a bug found on the way**
The draw is weighted sampling without replacement (Efraimidis–Spirakis keys),
with each room's weight equal to its rarity weight shared among the rooms of that
rarity, which matches "pick a rarity, then a room".
First version: slot 0 and slots 1–2 shared the same random numbers. Comparing
with the web-style draw over 600 games showed gem-cost rooms appeared about 25%
more often (1.18 vs 0.93 paid offers per draft) and the rule-based win rate fell
from 24.8% to 18.7%. Fix: independent numbers for slot 0. After the fix the
distributions match the web draw (per-slot shape and rarity frequencies within
about 1 point) and the rule-based agent wins 27.0% on the same games.
Lesson for the report: check a re-implemented random process against the
original by distribution, not only by unit tests.

**D9. Seed sets** (default)
- Test: 1,000,000–1,000,999. Only for final comparisons.
- PPO validation: 500,000–500,299. Used during training to pick the best model.
- Rule-based tuning: 0–299 (search) and 5000–5999 (check).
- PPO training: seeds drawn from 2^62 values by each environment.
Why: nothing that chooses or tunes an agent ever sees the test seeds.

### Agents

**D10. Observation: flat vector of 834 numbers** (user approved the default)
45 cells × 17 features (room, doors, unopened/locked doors, player, reachable,
distance, draft target), 12 global features (steps, keys, gems, highest rank,
phase, deck size, deck share by shape), 3 offers × 19 features (doors, cost,
affordable, rarity, rewards, shape). Details in `DESIGN.md` section 3.
Known gaps (candidates for Exp 3): no count of key-giving rooms left in the deck;
no direct distance-to-goal or keys-still-needed; joined vs walled doors are not
explicit; the board is flat for the MLP.
Draft-phase gap (noted 2026-11-11): each offer's doors are given as absolute
directions, but not what they lead to once placed: new open doors, doors joining
a neighbour, doors walled off, a connection to the Antechamber, or whether any
open door is left. The agent has to combine offer doors, target cell and board
itself. In the CNN (D15) the offers bypass the convolution and meet the board
only in the head, so the CNN does not help here either. Fix candidates:
(a) per-offer computed features (new doors, joins, walled, connects to goal,
open doors left after placement); (b) draw each offer onto extra board channels
at the target cell. Relevant because "no doors left" causes 26% of PPO losses.

**D11. Rule-based agent: one-step scoring, tuned by random search** (default)
Scores every legal action (door phase: progress toward the Antechamber, walking
distance, lock cost, whether the goal is still reachable with the steps left;
draft phase: resources, new doors, connecting to the goal, never closing the last
door). No look-ahead.
17 weights were tuned by random search: 120 settings on seeds 0–299, top 4
re-checked on 5000–5999. Hand-set defaults won 12.7% / 14.4%; the chosen
setting 27.7% / 25.2% (before the D8 fix). Final test: 24.4%.
For the report: it is a tuned heuristic, which makes it a strong baseline; say so.

**D12. PPO setup** (default)
sb3-contrib MaskablePPO, 8 parallel environments, lr 3e-4, γ 0.99 (an episode is
about 50 actions, two per turn), GAE 0.95, rollout 256 per env, batch 512,
10 epochs, entropy 0.01, clip 0.2. Standard values; not tuned yet.

**D13. Model selection by validation win rate** (default)
Every 100k steps the policy plays 300 validation games; the best one is kept as
`best_model.zip`.
Caveat seen in Exp 1: the best of about 30 noisy evaluations overstates the model
(validation best 10.3% → test 7.1%). Report test numbers only.

**D14. Curriculum on starting keys and gems** (default; tested in Exp 1)
Start 6 keys / 6 gems, move linearly to 2 / 2 over the first half of training;
validation always at 2 / 2. Result: no clear gain on the test seeds.

**D15. CNN uses the same observation** (user chose CNN; design default)
The board block is reshaped into a 17-channel 9×5 image with 2 coordinate
channels, so the MLP and CNN differ only in the network. Two seeds to separate
the effect of the network from run-to-run luck.

**D18. Keep the CNN runs at the same settings despite the cost** (user, 2026-11-11)
Measured speed: about 235 steps/s per CNN run with two side by side, vs about
790 for the MLP pair (3.3× slower; CPU only, no NVIDIA GPU). 3M steps take
about 3–3.5 hours per run. Options: stop at 1.5M and compare with the MLP at the
same point; lighten the update (4 epochs or 32 channels) and restart; or let it
run. Chosen: let it run, so the only difference from Exp 1 is the network.

**D19. Room colour is not in the observation** (discussed with the user, 2026-11-11; open)
The user suggested giving the agent each room's colour. In the minimal rules
colour has no effect: rooms differ only in shape, gem cost, rarity and the steps,
keys and gems they give, which are all observed. Red rooms' step penalty already
shows as negative steps. Adding colour would only add inputs with no effect on
the outcome. It becomes useful if (a) a colour rule is added to the game (then
it must be observed, and earlier results are no longer comparable), or (b) for
drawing the board in the dashboard and report figures (no effect on agents;
needs the colours of all 99 rooms, which `rooms.py` lacks). Waiting for the team
to say which, if either.

**D21. Continue with the MLP; test PPO settings and observation v2 together** (user, 2026-11-12)
After Exp 2 the CNN gain was small and not significant, and it trains 3.3×
slower. The user asked whether tuning PPO settings could help a lot. Expected
answer: probably not much, because training had plateaued and the rule-based
agent's edge comes from values it computes (goal distance, step feasibility,
whether a room closes the board) that PPO does not observe. Both were tested
side by side with the MLP (Exp 3a: entropy 0.03 / 0.003, learning rate 1e-4,
second seed; Exp 3b: observation v2).

**D22. Observation v2 appended to v1, versioned** (default)
Options: replace the observation, or append features and keep a version switch.
Chosen: append (`obs_version=2`, 867 numbers). Version 1 models still load and
evaluate (the agent reads the version from the model's input size), and the
CNN's board block is unchanged. The added features are computed from the state
(no hidden information): lower bounds on rooms and steps to the goal, keys short,
deck contents by reward, and per-offer placement effects. Room colour is still
left out (D19).

**D23. Longer training by resuming, not restarting** (user asked for longer training and for a way to continue runs, 2026-11-13)
Options:
- (a) Train new 10M runs from scratch: about 3.5 hours, and they repeat the first 3M steps.
- (b) Add `--resume` and continue the 3M runs to 10M.

Chosen: (b). Weights and optimizer state are loaded from `model.zip`, and the step count and logs continue. The environments get new seeds, so training games do not repeat.

Caveats for the report:
- The rollout buffer starts empty at the resume point.
- The recent training win rate restarts its 1000-game window.
- Neither matters for PPO, which collects a fresh rollout every update.

### Tooling

**D16. Python virtual environment in `project/.venv`** (user)
Dependencies in `requirements.txt`.

**D17. Every evaluated game stores its action list** (default)
With the seed and config, that replays the game exactly. It makes the
qualitative analysis in the proposal possible for any game after the fact.

**D20. Public GitHub repository with all models and logs** (user, 2026-11-11)
The team continues from the repository. Public was the user's choice; all trained
models (`model.zip`, `best_model.zip`) and TensorBoard logs are included so
teammates can evaluate without retraining. Left out: the instructor's status
report handout (`status.pdf`) and the copied wiki and Reddit texts
(`drafting.md`, `locked doors.md`), which are not ours to publish; they stay in
the local folder.

---

## 3. Results so far (test seeds 1,000,000–1,000,999; 50 steps, 2 keys, 2 gems)

| Agent | Win rate | Avg highest rank | Out of steps | Out of keys | Goal keys | No doors |
|---|---:|---:|---:|---:|---:|---:|
| Random | 0.0% | 3.69 | 8.2% | 4.1% | 0.0% | 87.7% |
| Rule-based | 24.4% | 8.00 | 5.5% | 25.4% | 23.8% | 20.9% |
| PPO MLP `ppo-base` | 7.5% | 7.69 | 18.9% | 29.4% | 18.3% | 25.9% |
| PPO MLP `ppo-curriculum` | 7.1% | 7.54 | 19.1% | 26.9% | 20.1% | 26.8% |
| PPO CNN `ppo-cnn` (seed 0) | 8.9% | 7.61 | 10.4% | 33.5% | 23.5% | 23.7% |
| PPO CNN `ppo-cnn-s1` (seed 1) | 9.0% | 7.75 | 19.5% | 30.5% | 20.1% | 20.9% |
| PPO MLP obs v2 `ppo-obs2` (seed 0) | 11.4% | 7.27 | 12.8% | 23.0% | 22.2% | 30.6% |
| PPO MLP obs v2 `ppo-obs2-s1b` (seed 1) | 13.0% | 7.78 | 18.1% | 26.7% | 22.4% | 19.8% |

Rule-based with 5 keys and 5 gems at the start: 58.6% (random still 0.0%).

Proposal hypotheses
- Random wins almost never: confirmed (0 of 1000).
- Rule-based beats random: confirmed.
- PPO beats random: confirmed (7–9% and highest rank 7.5–7.8 vs 3.7).
- PPO matches or beats rule-based: not yet (best 13% vs 24%).

Sanity cases: all four pass (`DESIGN.md` section 7).

---

## 4. Observations for the analysis section

- About half of the rule-based agent's losses are key shortages (blocked by
  locks, or at the Antechamber with fewer than 3 keys); a fifth are boards it
  closed off itself.
- PPO learns to climb quickly (highest rank 3.7 → 7.3 in the first 100k steps)
  but rarely finishes. Compared with the rule-based agent it runs out of steps
  three times as often (19% vs 5.5%) and drafts fewer rooms (18.6 vs 24.3): it
  walks more and builds less.
- With a curriculum the training win rate drops at each step down in starting
  resources (16% → 11% → 5% → 4%) while the validation win rate keeps rising;
  the drops come from the harder setting, not from forgetting.
- Validation on 300 games moves by about ±3 points from luck alone.

To do: replay a sample of PPO losses in the dashboard and describe the typical
failure (routes, key use, detours), as the proposal's qualitative analysis says.

---

## 5. Directions

Ordered by expected value; see `EXPERIMENTS.md` for detail.

1. CNN board encoder (Exp 2, done: about +1.5 points over the MLP; paired
   McNemar z 1.2–1.6 per pair, not significant; MLP kept for Exp 3 because it is
   3.3× faster).
2. Observation additions (Exp 3b, done: observation v2, 11.4% / 13.0% on the
   test seeds, +5 points over MLP v1, significant). PPO setting changes (Exp 3a)
   showed no gain and were stopped. Next: more key information (keys needed for
   locks on the best route).
3. Reward shaping for key saving and walking cost, kept well below the win reward.
4. A policy head that outputs door logits per cell (convolutional).
5. Longer training and more seeds per setup. First: an MLP run with seed 1, so
   the MLP and the CNN both have two seeds.

---

## 6. Report material

| What | Where |
|---|---|
| Baseline and PPO comparison on test seeds | `results/summary.json`, `results/episodes_<agent>.csv` |
| Win rate and highest-rank figure | `results/summary.png` (`python -m blueprince.evaluate --out results --plot`) |
| Training curves | dashboard Training tab; `results/training/<run>/progress.jsonl`; TensorBoard in `tb/` |
| Game replays for qualitative analysis | dashboard Evaluation tab, click a game |
| Method details (rules, spaces, reward, seeds) | `DESIGN.md` |
| Every training run with settings | `EXPERIMENTS.md` |

Points to correct or add relative to the proposal
- Section 4 says each seed fixes the blueprint draws and lock states. True for
  locks; for draws, see D7.
- The Antechamber cannot be walked through (D3).
- The rule-based agent's weights were tuned (D11).
- Observation and network details (D10, D15) were not in the proposal.
