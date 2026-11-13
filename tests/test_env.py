"""Sanity cases from the proposal, plus rule checks. Run: python -m pytest"""
import random
from dataclasses import replace

import numpy as np
import pytest

from blueprince import OBS_SIZE, BluePrinceEnv, Config, Game
from blueprince.agents import RandomAgent, RuleBasedAgent
from blueprince.evaluate import evaluate, summarize
from blueprince.game import (E, GOAL, N, N_ACTIONS, N_DOOR_ACTIONS, PHASE_DRAFT, S, W,
                             decode_door, encode_door, valid_orientations)
from blueprince.rooms import ROOMS, SHAPE_COUNTS


def play_random(seed, policy_seed, config=None):
    game, rng = Game(seed, config), random.Random(policy_seed)
    while not (game.over or game.truncated):
        game.step(rng.choice(game.legal_actions()))
    return game


def test_room_data():
    assert len(ROOMS) == 99
    assert len({r.name for r in ROOMS}) == 99
    assert SHAPE_COUNTS == {"D": 29, "S": 14, "L": 27, "T": 20, "X": 9}
    assert sum(1 for r in ROOMS if r.turn == "L") == 14
    assert sum(1 for r in ROOMS if r.turn == "R") == 13


def test_action_numbering_matches_actions_md():
    # Entrance Hall is C1 = (1, 3): north 8, east 9, west 11
    assert [encode_door(1, 3, d) for d in (N, E, W)] == [8, 9, 11]
    assert decode_door(8) == (1, 3, N)
    assert sorted(Game(0).legal_actions()) == [8, 9, 11]


def test_placement_rules():
    by_name = {r.name: r for r in ROOMS}
    fits = lambda name, r, c, move: bool(valid_orientations(by_name[name], r, c, move))
    # corner A1 entered moving west: only dead ends and L rooms that turn inward
    assert fits("Closet", 1, 1, W)
    assert not fits("Spare Room", 1, 1, W) and not fits("Hallway", 1, 1, W) and not fits("Passageway", 1, 1, W)
    assert fits("Parlor", 1, 1, W) and not fits("Bedroom", 1, 1, W)      # right turn -> N, left turn -> S
    # edge A5: straight only along the edge, never a 4-way; T has one orientation
    assert fits("Spare Room", 5, 1, N) and not fits("Spare Room", 5, 1, W)
    assert not fits("Passageway", 5, 1, N)
    assert len(valid_orientations(by_name["Hallway"], 5, 1, W)) == 1
    assert len(valid_orientations(by_name["Hallway"], 5, 3, N)) == 3


@pytest.mark.parametrize("seed", range(200))
def test_random_games_are_legal_and_end(seed):
    env = BluePrinceEnv()
    obs, info = env.reset(seed=seed)
    rng = random.Random(seed)
    for _ in range(Config().max_actions):
        mask = env.action_masks()
        assert mask.shape == (N_ACTIONS,) and mask.any()
        assert sorted(np.flatnonzero(mask)) == sorted(env.game.legal_actions())
        g = env.game
        if g.phase == PHASE_DRAFT:
            offers = g.draft.offers
            assert not mask[:N_DOOR_ACTIONS].any()
            assert offers[0].cost == 0
            assert len({o.room for o in offers}) == len(offers)
            assert not (len(offers) == 3 and all(ROOMS[o.room].shape == "D" for o in offers))
        else:
            assert not mask[N_DOOR_ACTIONS:].any()
        obs, reward, terminated, truncated, info = env.step(rng.choice(g.legal_actions()))
        assert obs.shape == (OBS_SIZE,) and env.observation_space.contains(obs)
        assert g.steps >= 0 and g.keys >= 0 and g.gems >= 0
        if terminated or truncated:
            break
    assert terminated and not truncated          # the 99-room deck ends every game long before 300 actions
    assert info["win"] or info["loss_cause"]


def test_illegal_action_is_rejected():
    env = BluePrinceEnv()
    env.reset(seed=0)
    with pytest.raises(ValueError):
        env.step(0)
    with pytest.raises(ValueError):
        env.step(180)       # no draft in progress


def test_same_seed_and_actions_replay_exactly():
    a = play_random(7, policy_seed=1)
    b = Game(7)
    for action in a.actions:
        b.step(action)
    assert (b.rooms, b.locks, b.steps, b.keys, b.gems, b.win, b.loss_cause, b.total_reward) == \
           (a.rooms, a.locks, a.steps, a.keys, a.gems, a.win, a.loss_cause, a.total_reward)


def test_locks_do_not_depend_on_the_agent():
    for seed in range(30):
        a, b = play_random(seed, 1), play_random(seed, 2)
        shared = a.lock_rolls.keys() & b.lock_rolls.keys()
        assert shared and all(a.lock_rolls[e] == b.lock_rolls[e] for e in shared)
    assert play_random(1, 1).lock_rolls != play_random(2, 1).lock_rolls


def test_offers_at_a_cell_survive_unrelated_deck_changes():
    same = total = 0
    for seed in range(200):
        a, b = Game(seed), Game(seed)
        a.step(8)                                    # open the north door straight away
        b.step(9); b.step(180)                       # draft east first, then the same north door
        if 8 not in b.legal_actions():
            continue
        b.step(8)
        if b.rooms[(1, 4)].room in {o.room for o in a.draft.offers}:
            continue                                 # b already used a room that a was offered
        total += 1
        same += a.draft.offers == b.draft.offers
    assert total > 100 and same / total > 0.9


def test_draw_follows_rarity_weights():
    # slot 0 of the first draft: rarity frequencies should be close to 50/30/15/5
    n = 4000
    counts = {k: 0 for k in "CSUR"}
    for seed in range(n):
        g = Game(seed)
        g.step(8)
        counts[ROOMS[g.draft.offers[0].room].rarity] += 1
    for k, w in zip("CSUR", (0.50, 0.30, 0.15, 0.05)):
        assert abs(counts[k] / n - w) < 0.03


def test_win_reward_and_goal_keys():
    wins = [g for g in (evaluate_game(s) for s in range(300)) if g.win]
    assert wins
    for g in wins:
        assert g.player == GOAL and g.max_rank == 9
        assert g.total_reward == pytest.approx(10 + 510 / 256)


def evaluate_game(seed):
    env, agent = BluePrinceEnv(), RuleBasedAgent()
    env.reset(seed=seed)
    done = False
    while not done:
        _, _, term, trunc, _ = env.step(agent.act(env))
        done = term or trunc
    return env.game


def test_gymnasium_api():
    env = BluePrinceEnv()
    assert env.action_space.n == N_ACTIONS and env.observation_space.shape == (OBS_SIZE,)
    env.reset(seed=3)
    first = env.reset()[1]["seed"]
    env.reset(seed=3)
    assert env.reset()[1]["seed"] == first       # unseeded resets are reproducible from the first seed


def test_agents_rank_as_expected_and_resources_help():
    seeds = range(2000, 2300)
    rand = summarize(evaluate(RandomAgent(), seeds))["win_rate"]
    rule = summarize(evaluate(RuleBasedAgent(), seeds))["win_rate"]
    rich = summarize(evaluate(RuleBasedAgent(), seeds, replace(Config(), keys=5, gems=5)))["win_rate"]
    assert rand < 0.02 < rule < rich


def test_replay_rebuilds_the_game():
    from blueprince.replay import replay
    g = evaluate_game(1)
    frames = replay(1, g.actions)
    assert len(frames) == len(g.actions) + 1
    last = frames[-1]
    assert (last["steps"], last["keys"], last["gems"], last["win"]) == (g.steps, g.keys, g.gems, g.win)
    assert len(last["rooms"]) == len(g.rooms)
    assert all(f["event"] for f in frames)
    assert all((f["draft"] is not None) == (f["phase"] == "draft") for f in frames)


def test_saved_run_round_trips(tmp_path):
    from blueprince.evaluate import load_rows, save_run
    rows = evaluate(RuleBasedAgent(), range(5))
    save_run(str(tmp_path), {"rule": rows}, {}, 0)
    assert load_rows(str(tmp_path / "episodes_rule.csv")) == rows


def test_ppo_agent_plays_legal_games():
    pytest.importorskip("sb3_contrib")
    from sb3_contrib import MaskablePPO
    from blueprince.agents import PPOAgent
    model = MaskablePPO("MlpPolicy", BluePrinceEnv(), n_steps=64, batch_size=64, seed=0, device="cpu")
    rows = evaluate(PPOAgent(model=model), range(10))   # untrained policy; step() raises on any illegal action
    assert len(rows) == 10 and all(r["win"] or r["loss_cause"] for r in rows)


def test_board_cnn_sees_cells_in_place():
    torch = pytest.importorskip("torch")
    from blueprince.networks import BoardCNN
    env = BluePrinceEnv()
    obs, _ = env.reset(seed=0)                      # player in the Entrance Hall, C1 = (1, 3)
    image = BoardCNN.board_image(torch.as_tensor(obs[None]))
    assert image.shape == (1, 17, 9, 5)
    assert image[0, 13].nonzero().tolist() == [[0, 2]]     # player channel at rank 1, column C
    assert image[0, 0, 8, 2] == 1                           # Antechamber at rank 9, column C
    net = BoardCNN(env.observation_space)
    assert net(torch.as_tensor(obs[None])).shape == (1, net.features_dim)


# ---- observation version 2 (Exp 3)

def test_obs_v2_extends_v1():
    from blueprince.env import OBS_SIZE_V2, encode_observation
    v1, v2 = BluePrinceEnv(), BluePrinceEnv(obs_version=2)
    assert v2.observation_space.shape == (OBS_SIZE_V2,)
    for seed in range(20):
        o1, o2 = v1.reset(seed=seed)[0], v2.reset(seed=seed)[0]
        rng = random.Random(seed)
        while True:
            assert np.array_equal(o2[:OBS_SIZE], o1) and v2.observation_space.contains(o2)
            if v1.game.over or v1.game.truncated:
                break
            a = rng.choice(v1.game.legal_actions())
            o1, o2 = v1.step(a)[0], v2.step(a)[0]


def test_obs_v2_plan_features():
    from blueprince.env import PLAN_OFFSET, encode_observation, rooms_to_goal
    assert rooms_to_goal((8, 3)) == 1 and rooms_to_goal((2, 3)) == 7 and rooms_to_goal((9, 1)) == 2
    game = Game(0)
    p = encode_observation(game, 2)[PLAN_OFFSET:]
    # start: Entrance Hall (1, 3) with doors W, N, E; best is N to (2, 3): 0 walk + 7 rooms + 1 step in
    assert p[0] == 0 and p[2] == pytest.approx(8 / 20) and p[3] == pytest.approx((50 - 8) / 50)
    assert p[4] == pytest.approx(7 / 10) and p[5] == pytest.approx(1 / 3) and p[6] == pytest.approx(3 / 10)
    assert p[8] == pytest.approx(sum(1 for r in ROOMS if r.keys > 0) / 10)


def test_obs_v2_offer_features_match_placement():
    from blueprince.env import OFFER_PLAN_FEATURES, OFFER_PLAN_OFFSET, encode_observation
    for seed in range(30):
        game, rng = Game(seed), random.Random(seed)
        while not (game.over or game.truncated):
            if game.phase == PHASE_DRAFT:
                obs = encode_observation(game, 2)
                for slot, offer in enumerate(game.draft.offers):
                    if game.gems < offer.cost:
                        continue
                    f = obs[OFFER_PLAN_OFFSET + slot * OFFER_PLAN_FEATURES:][:OFFER_PLAN_FEATURES]
                    after = Game(seed)
                    for a in game.actions:
                        after.step(a)
                    after.step(N_DOOR_ACTIONS + slot)
                    target = game.draft.target
                    assert f[0] + f[1] + f[2] == pytest.approx((len(offer.doors) - 1) / 3)
                    open_after = sum(1 for r, c, d, _, _ in after.frontier())
                    assert f[4] == pytest.approx(min(1.0, open_after / 10))
                    new = sum(1 for r, c, d, _, _ in after.frontier() if (r, c) == target)
                    assert f[0] == pytest.approx(new / 3)
            game.step(rng.choice(game.legal_actions()))


def test_ppo_agent_reads_obs_version_from_model():
    pytest.importorskip("sb3_contrib")
    from sb3_contrib import MaskablePPO
    from blueprince.agents import PPOAgent
    for version in (1, 2):
        model = MaskablePPO("MlpPolicy", BluePrinceEnv(obs_version=version), n_steps=64, batch_size=64,
                            seed=0, device="cpu")
        agent = PPOAgent(model=model)
        assert agent.obs_version == version
        assert len(evaluate(agent, range(3))) == 3
