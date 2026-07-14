import json
import os
import sys
import tempfile
import unittest
from unittest import mock
ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'src'))
from action_codec import ActionCodec
from bot import MahjongBot
from local_game import run_games
from policy import GoalBasedPolicy, NeuralPolicy
from search_planner import BoundedRolloutPlanner
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


class _AdaptiveStubModel:
    def __init__(self, chosen_action='HU', entropy=0.0, belief_entropy=0.0):
        self.chosen_action = chosen_action
        self.entropy = entropy
        self.belief_entropy = belief_entropy
        self.last_belief_weight = None

    def policy_info_from_features(self, features, legal_actions, belief_weight=0.0, efficiency_weight=0.0, temperature=None):
        self.last_belief_weight = belief_weight
        return {
            'actions': list(legal_actions),
            'scores': [2.0, 1.0][:len(legal_actions)],
            'probabilities': [0.8, 0.2][:len(legal_actions)],
            'log_probabilities': [0.0, -1.0][:len(legal_actions)],
            'belief_probs': [1.0 / 34.0] * 34,
            'belief_entropy': self.belief_entropy,
            'entropy': self.entropy,
            'value': 0.0,
        }

    def choose_action_from_features(self, features, legal_actions, belief_weight=0.0, efficiency_weight=0.0, temperature=None):
        self.last_belief_weight = belief_weight
        if belief_weight > 0.1 and 'PLAY W1' in legal_actions:
            return 'PLAY W1'
        return self.chosen_action if self.chosen_action in legal_actions else legal_actions[0]


class _SearchStubModel:
    def __init__(self):
        self.last_belief_weight = None

    def policy_info_from_features(self, features, legal_actions, belief_weight=0.0, efficiency_weight=0.0, temperature=None):
        self.last_belief_weight = belief_weight
        scores = []
        for action in legal_actions:
            if action == 'PLAY W1':
                scores.append(0.9)
            elif action == 'PLAY W2':
                scores.append(0.8)
            else:
                scores.append(0.1)
        total = sum(scores) if sum(scores) > 0 else 1.0
        return {
            'actions': list(legal_actions),
            'scores': scores,
            'probabilities': [score / total for score in scores],
            'log_probabilities': [0.0 for _ in legal_actions],
            'belief_probs': [0.0] * 34,
            'belief_entropy': 0.0,
            'entropy': 0.0,
            'value': 0.0,
        }

    def estimate_value_from_features(self, features):
        hand_counts = features.get('hand_counts', []) if isinstance(features, dict) else []
        self_discard_counts = features.get('self_discard_counts', []) if isinstance(features, dict) else []
        if len(self_discard_counts) > 0 and self_discard_counts[0] > 0:
            return 0.2
        if len(self_discard_counts) > 1 and self_discard_counts[1] > 0:
            return 0.25
        return 0.1

    def choose_action_from_features(self, features, legal_actions, belief_weight=0.0, efficiency_weight=0.0, temperature=None):
        return 'PLAY W2' if 'PLAY W2' in legal_actions else legal_actions[0]


class TestNeuralPolicyAdaptation(unittest.TestCase):

    def _make_uncertain_state(self):
        gs = GameState()
        gs.my_id = 0
        gs.hand = ['W1', 'W2', 'W3', 'B1', 'B2', 'B3', 'T1']
        gs.last_request_type = 3
        gs.last_request_action = 'PLAY'
        gs.last_tile = 'W4'
        gs.last_actor = 3
        return gs

    def _make_draw_state(self):
        gs = GameState()
        gs.my_id = 0
        gs.hand = ['W1', 'W2', 'W3']
        gs.last_request_type = 2
        gs.last_request_action = 'DRAW'
        gs.last_tile = 'W4'
        gs.last_actor = 0
        return gs

    def test_uncertainty_triggers_fallback(self):
        state = self._make_uncertain_state()
        policy = NeuralPolicy(
            state=state,
            adaptation={'uncertainty_threshold': 0.1, 'fallback_on_uncertain': True, 'risk_mode': 'balanced'},
        )
        policy.model = _AdaptiveStubModel(chosen_action='HU', entropy=5.0, belief_entropy=5.0)

        action = policy.choose_action()
        self.assertEqual(action, 'PASS')

    def test_belief_ablation_changes_selected_action(self):
        state = self._make_draw_state()
        policy = NeuralPolicy(
            state=state,
            adaptation={
                'uncertainty_threshold': 10.0,
                'fallback_on_uncertain': False,
                'belief_weight': 0.5,
                'disable_belief': False,
                'risk_mode': 'balanced',
            },
        )
        policy.model = _AdaptiveStubModel(chosen_action='PLAY W2', entropy=0.0, belief_entropy=0.0)

        action_with_belief = policy.choose_action()
        self.assertEqual(action_with_belief, 'PLAY W1')
        self.assertGreater(policy.model.last_belief_weight, 0.0)

        policy.adaptation['disable_belief'] = True
        action_without_belief = policy.choose_action()
        self.assertEqual(action_without_belief, 'PLAY W2')
        self.assertEqual(policy.model.last_belief_weight, 0.0)


class TestSearchTimeAugmentation(unittest.TestCase):

    def _make_state(self):
        gs = GameState()
        gs.my_id = 0
        gs.hand = ['W1', 'W2', 'W3', 'B1', 'B2', 'B3', 'T1']
        gs.last_request_type = 2
        gs.last_request_action = 'DRAW'
        gs.last_tile = 'W4'
        gs.last_actor = 0
        return gs

    def test_bounded_rollout_planner_prefers_high_value_action_under_hidden_risk(self):
        model = _SearchStubModel()
        planner = BoundedRolloutPlanner(
            model=model,
            top_k=2,
            rollout_samples=16,
            budget_ms=50,
            disabled=False,
            belief_weight=5.0,
            seed=11,
        )
        features = {
            'hand_counts': [1] + [0] * 33,
            'seen_counts': [0] * 34,
            'self_discard_counts': [0] * 34,
            'pack_counts': [0] * 34,
            'opponent_discard_counts': [[0] * 34, [0] * 34, [0] * 34],
            'meta': [0, 0, 0, 0, 2, 0, 1, 0],
            'seat': 0,
        }
        legal_actions = ['PLAY W1', 'PLAY W2']

        action = planner.plan(features, legal_actions, belief_weight=1.0)
        self.assertIn(action, legal_actions)
        self.assertEqual(action, 'PLAY W1')

    def test_bounded_rollout_planner_can_be_disabled_or_budgeted_out(self):
        model = _SearchStubModel()
        features = {
            'hand_counts': [1] + [0] * 33,
            'seen_counts': [0] * 34,
            'self_discard_counts': [0] * 34,
            'pack_counts': [0] * 34,
            'opponent_discard_counts': [[0] * 34, [0] * 34, [0] * 34],
            'meta': [0, 0, 0, 0, 2, 0, 1, 0],
            'seat': 0,
        }
        legal_actions = ['PLAY W1', 'PLAY W2']

        disabled_planner = BoundedRolloutPlanner(model=model, budget_ms=50, disabled=True)
        self.assertIsNone(disabled_planner.plan(features, legal_actions))

        zero_budget_planner = BoundedRolloutPlanner(model=model, budget_ms=0, disabled=False)
        self.assertIsNone(zero_budget_planner.plan(features, legal_actions))

    def test_neural_policy_uses_search_planner_when_enabled(self):
        state = self._make_state()
        policy = NeuralPolicy(
            state=state,
            adaptation={
                'search_enabled': True,
                'search_disabled': False,
                'search_budget_ms': 50,
                'search_top_k': 2,
                'search_rollout_samples': 16,
                'search_belief_weight': 5.0,
                'uncertainty_threshold': 10.0,
                'fallback_on_uncertain': False,
                'disable_belief': False,
            },
        )
        policy.model = _SearchStubModel()

        action = policy.choose_action()
        self.assertEqual(action, 'PLAY W1')

    def test_neural_policy_search_can_be_disabled(self):
        state = self._make_state()
        policy = NeuralPolicy(
            state=state,
            adaptation={
                'search_enabled': True,
                'search_disabled': True,
                'search_budget_ms': 50,
                'search_top_k': 2,
                'search_rollout_samples': 16,
                'search_belief_weight': 5.0,
                'uncertainty_threshold': 10.0,
                'fallback_on_uncertain': False,
                'disable_belief': False,
            },
        )
        policy.model = _SearchStubModel()

        action = policy.choose_action()
        self.assertEqual(action, 'PLAY W2')


class TestGoalPolicySafety(unittest.TestCase):

    def _make_state(self):
        gs = GameState()
        gs.my_id = 0
        gs.hand = ['W1', 'W1', 'W1', 'B1', 'B2']
        gs.packs = []
        gs.opponent_packs = {1: [], 2: [], 3: []}
        gs.opponent_discards = {1: [], 2: [], 3: []}
        gs.seen_tiles = {}
        return gs

    def test_choose_discard_empty_hand_returns_empty_string(self):
        policy = GoalBasedPolicy(self._make_state())
        self.assertEqual(policy._choose_discard([], [], None), '')

    def test_should_gang_open_removes_only_the_gang_tiles(self):
        policy = GoalBasedPolicy(self._make_state())
        hand = ['W1', 'W1', 'W1', 'B1', 'B2']
        with mock.patch('policy.min_shanten', side_effect=[2, 2]) as min_shanten_mock:
            policy._should_gang_open('W1', hand, [])
        reduced_hand = min_shanten_mock.call_args_list[1][0][0]
        self.assertEqual(reduced_hand, ['B1', 'B2'])

    def test_peng_and_discard_returns_none_when_tile_count_is_too_low(self):
        policy = GoalBasedPolicy(self._make_state())
        self.assertIsNone(policy._peng_and_discard('W2', ['W1', 'B1', 'B2'], []))

if __name__ == '__main__':
    unittest.main()
