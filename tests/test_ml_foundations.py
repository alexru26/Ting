import json
import os
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'src'))

from action_codec import ActionCodec
from bot import MahjongBot
from local_game import run_games
from policy import GoalBasedPolicy, NeuralPolicy, create_policy
from state import GameState


class TestLegalActionAgreement(unittest.TestCase):
    def _make_draw_state(self):
        gs = GameState()
        gs.my_id = 0
        gs.hand = ['W1', 'W1', 'W1', 'W1', 'B2', 'B3', 'T4']
        gs.packs = [('PENG', 'B2', 1)]
        gs.last_request_type = 2
        gs.last_request_action = 'DRAW'
        gs.last_tile = 'W1'
        gs.last_actor = 0
        return gs

    def _make_play_state(self):
        gs = GameState()
        gs.my_id = 0
        gs.hand = ['W1', 'W2', 'W4', 'B1', 'B1', 'B5', 'J1']
        gs.last_request_type = 3
        gs.last_request_action = 'PLAY'
        gs.last_tile = 'W3'
        gs.last_actor = 3
        return gs

    def _assert_agreement(self, gs):
        codec = ActionCodec()
        bot = MahjongBot()
        legal = set(gs.enumerate_legal_actions())

        for action in codec.all_actions():
            self.assertEqual(action in legal, bot._is_legal_action(gs, action), action)

        mask = gs.legal_action_mask(codec)
        self.assertEqual(len(mask), codec.size)

        for idx, action in enumerate(codec.all_actions()):
            self.assertEqual(mask[idx], 1 if action in legal else 0)

    def test_draw_state_agreement(self):
        self._assert_agreement(self._make_draw_state())

    def test_play_state_agreement(self):
        self._assert_agreement(self._make_play_state())


class TestSimulatorDatasetExport(unittest.TestCase):
    def test_run_games_export_dataset(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'dataset.jsonl')
            run_games(n=1, seed=7, export_dataset_path=path)

            self.assertTrue(os.path.exists(path))
            with open(path, 'r', encoding='utf-8') as handle:
                first = handle.readline().strip()

            self.assertTrue(first)
            payload = json.loads(first)
            self.assertIn('action', payload)
            self.assertIn('features', payload)


class _DummyFallbackPolicy:
    def choose_action(self):
        return 'PASS'


class TestPolicyModeSelection(unittest.TestCase):
    def _make_state(self):
        gs = GameState()
        gs.my_id = 0
        gs.hand = [
            'W1',
            'W1',
            'W2',
            'W3',
            'B1',
            'B2',
            'B3',
            'T1',
            'T2',
            'T3',
            'F1',
            'J1',
            'J2',
        ]
        gs.last_request_type = 2
        gs.last_request_action = 'DRAW'
        gs.last_tile = 'W1'
        gs.last_actor = 0
        return gs

    def test_create_policy_rule_returns_goal_based(self):
        policy = create_policy(self._make_state(), mode='rule')
        self.assertIsInstance(policy, GoalBasedPolicy)

    def test_create_policy_unknown_mode_defaults_to_neural(self):
        policy = create_policy(self._make_state(), mode='anything-else')
        self.assertIsInstance(policy, NeuralPolicy)

    def test_create_policy_without_mode_defaults_to_neural(self):
        policy = create_policy(self._make_state())
        self.assertIsInstance(policy, NeuralPolicy)

    def test_create_policy_neural_mode_returns_neural_policy(self):
        policy = create_policy(self._make_state(), mode='neural')
        self.assertIsInstance(policy, NeuralPolicy)

    def test_create_policy_mode_is_case_insensitive(self):
        policy = create_policy(self._make_state(), mode='NeUrAl')
        self.assertIsInstance(policy, NeuralPolicy)

    def test_neural_policy_uses_fallback_policy_action(self):
        state = self._make_state()
        policy = NeuralPolicy(
            state,
            model_path='missing-model.bin',
            fallback_policy=_DummyFallbackPolicy(),
        )
        self.assertEqual(policy.choose_action(), 'PASS')


if __name__ == '__main__':
    unittest.main()
