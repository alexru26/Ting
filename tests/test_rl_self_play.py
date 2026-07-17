import json
import os
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'src'))

from action_codec import ActionCodec
from rl_self_play import build_opponent_league_pool, run_self_play_worker, shape_rewards
from rl_self_play import evaluate_against_baseline, promotion_gate, run_ppo_fine_tuning
from model import CnnPolicyValueModel


class _FakeGame:
    call_count = 0

    def __init__(self, quan=0, seed=None):
        self.quan = quan
        self.seed = seed

    def run(self):
        index = _FakeGame.call_count
        _FakeGame.call_count += 1
        if index % 3 == 0:
            return {'winner': 0, 'fan': 8, 'scores': [24, -8, -8, -8]}
        if index % 3 == 1:
            return {'winner': 2, 'fan': 10, 'scores': [-8, -8, 24, -8]}
        return {'winner': None, 'fan': 0, 'scores': [0, 0, 0, 0]}


class _FakeState:
    def __init__(self, my_id=0):
        self.my_id = my_id
        self.quan = 0
        self.flowers = 0
        self.packs = []
        self.last_request_type = 2
        self.last_actor = 0
        self.last_request_action = 'DRAW'
        self.last_tile = 'W1'
        self.hand = ['W1', 'W2', 'W3']
        self.discards = []
        self.seen_tiles = {}
        self.opponent_discards = {1: [], 2: [], 3: []}
        self.opponent_packs = {1: [], 2: [], 3: []}

    def enumerate_legal_actions(self):
        return ['PASS', 'PLAY W1', 'PLAY W2']

    def is_legal_action(self, action):
        return action in self.enumerate_legal_actions()


class _FakePpoGame:
    call_count = 0

    def __init__(self, quan=0, seed=None, policy_factory=None):
        self.quan = quan
        self.seed = seed
        self.policy_factory = policy_factory

    def run(self):
        state = _FakeState(my_id=0)
        policy = self.policy_factory(state)
        _ = policy.choose_action()
        _FakePpoGame.call_count += 1
        return {'winner': 0, 'fan': 8, 'scores': [24, -8, -8, -8]}


class _FakeLeagueGame:
    last_policy_types = None

    def __init__(self, quan=0, seed=None, policy_factory=None):
        self.quan = quan
        self.seed = seed
        self.policy_factory = policy_factory

    def run(self):
        policies = []
        for seat in range(4):
            policy = self.policy_factory(_FakeState(my_id=seat))
            policies.append(type(policy).__name__)
            if seat == 0:
                _ = policy.choose_action()
        _FakeLeagueGame.last_policy_types = policies
        return {'winner': 0, 'fan': 8, 'scores': [24, -8, -8, -8]}


class TestRlSelfPlay(unittest.TestCase):
    def test_shape_rewards_combines_components(self):
        result = {'winner': 1, 'fan': 12, 'scores': [-8, 24, -8, -8]}
        rewards = shape_rewards(result, score_delta_weight=1.0, fan_weight=0.5, placement_weight=2.0)

        self.assertEqual(len(rewards), 4)
        self.assertGreater(rewards[1], rewards[0])
        self.assertGreater(rewards[1], rewards[2])

    def test_run_self_play_worker_aggregates_results(self):
        _FakeGame.call_count = 0
        summary = run_self_play_worker(
            games=6,
            seed=7,
            quan=0,
            game_factory=_FakeGame,
            score_delta_weight=1.0,
            fan_weight=0.0,
            placement_weight=0.0,
        )

        self.assertEqual(summary['games'], 6)
        self.assertEqual(summary['wins'][0], 2)
        self.assertEqual(summary['wins'][2], 2)
        self.assertEqual(summary['draws'], 2)
        self.assertEqual(len(summary['reward_totals']), 4)

    def test_build_opponent_league_pool_includes_external_registry(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry_path = os.path.join(tmp, 'opponents_registry.json')
            payload = {
                'count': 3,
                'opponents': [
                    {'id': 'ext-a', 'path': 'data/models/a.pkl', 'policy_mode': 'imitation'},
                    {'id': 'ext-b', 'path': 'data/models/b.pkl', 'policy_mode': 'imitation'},
                    {'id': 'ext-c', 'path': 'data/models/c.pkl', 'policy_mode': 'imitation'},
                ],
            }
            with open(registry_path, 'w', encoding='utf-8') as handle:
                json.dump(payload, handle)

            pool = build_opponent_league_pool(
                candidate_path='src/cnn_1000.h5',
                historical_checkpoints=['src/h1.h5', 'src/h2.h5'],
                external_registry_path=registry_path,
                max_external=2,
            )

            kinds = [row['kind'] for row in pool]
            self.assertIn('baseline', kinds)
            self.assertIn('candidate', kinds)
            self.assertEqual(kinds.count('historical'), 2)
            self.assertEqual(kinds.count('external'), 2)

    def test_ppo_fine_tuning_runs_and_promotes_on_good_evaluation(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_path = os.path.join(tmp, 'candidate.h5')
            codec = ActionCodec()
            model = CnnPolicyValueModel(
                action_space_size=codec.size,
                hidden_size=8,
                learning_rate=0.01,
            )
            model.save(model_path)

            _FakePpoGame.call_count = 0
            summary = run_ppo_fine_tuning(
                model_path=model_path,
                games=4,
                seed=11,
                eval_games=3,
                candidate_seat=0,
                game_factory=_FakePpoGame,
                promote_min_win_rate=0.5,
                promote_min_avg_score_delta=0.0,
                device='auto',
            )

            self.assertEqual(summary['episodes'], 4)
            self.assertGreater(summary['policy_updates'], 0)
            self.assertTrue(summary['promoted'])
            self.assertEqual(summary['evaluation']['candidate_win_rate'], 1.0)
            self.assertEqual(summary['requested_device'], 'auto')
            self.assertIn(summary['resolved_device'], ['cpu', 'cuda'])
            self.assertTrue(os.path.exists(model_path))

    def test_ppo_fine_tuning_uses_registry_opponents(self):
        with tempfile.TemporaryDirectory() as tmp:
            candidate_path = os.path.join(tmp, 'candidate.h5')
            opponent_path = os.path.join(tmp, 'opponent.h5')
            registry_path = os.path.join(tmp, 'opponents_registry.json')

            codec = ActionCodec()
            candidate = CnnPolicyValueModel(
                action_space_size=codec.size,
                hidden_size=8,
                learning_rate=0.01,
            )
            candidate.save(candidate_path)

            opponent = CnnPolicyValueModel(
                action_space_size=codec.size,
                hidden_size=8,
                learning_rate=0.01,
            )
            opponent.save(opponent_path)

            with open(registry_path, 'w', encoding='utf-8') as handle:
                json.dump(
                    {
                        'count': 1,
                        'opponents': [
                            {
                                'id': 'ext-a',
                                'path': opponent_path,
                                'policy_mode': 'imitation',
                            }
                        ],
                    },
                    handle,
                )

            _FakeLeagueGame.last_policy_types = None
            summary = run_ppo_fine_tuning(
                model_path=candidate_path,
                games=1,
                seed=11,
                eval_games=0,
                candidate_seat=0,
                game_factory=_FakeLeagueGame,
                promote_min_win_rate=0.5,
                promote_min_avg_score_delta=0.0,
                device='auto',
                opponent_registry_path=registry_path,
            )

            self.assertEqual(summary['episodes'], 1)
            self.assertIsNotNone(_FakeLeagueGame.last_policy_types)
            self.assertIn('PpoPolicy', _FakeLeagueGame.last_policy_types)
            self.assertTrue(any(policy_type not in ('GoalBasedPolicy', 'PpoPolicy') for policy_type in _FakeLeagueGame.last_policy_types))

    def test_promotion_gate_checks_metrics(self):
        accepted = promotion_gate({'games': 10, 'candidate_win_rate': 0.6, 'avg_score_delta': 1.0}, min_win_rate=0.55, min_avg_score_delta=0.0)
        rejected = promotion_gate({'games': 10, 'candidate_win_rate': 0.4, 'avg_score_delta': 1.0}, min_win_rate=0.55, min_avg_score_delta=0.0)
        self.assertTrue(accepted)
        self.assertFalse(rejected)


if __name__ == '__main__':
    unittest.main()