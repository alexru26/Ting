import json
import os
import sys
import tempfile
import unittest
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'src'))

from action_codec import ActionCodec
from bot import MahjongBot
from model_governance import build_model_registry_entry, duplicate_wall_evaluation, elo_update, sprt_promotion_gate, update_elo_ladder, write_model_registry
from local_game import Game
from policy import GoalBasedPolicy, NeuralPolicy
from state import GameState
from model import CnnPolicyValueModel


class _DuplicateWallGame:
    def __init__(self, quan=0, seed=None, policy_factory=None):
        self.quan = quan
        self.seed = seed
        self.policy_factory = policy_factory

    def run(self):
        state = GameState()
        state.my_id = 0
        state.hand = ['W1', 'W2', 'W3', 'B1', 'B2', 'B3', 'T1']
        state.last_request_type = 2
        state.last_request_action = 'DRAW'
        state.last_tile = 'W4'
        state.last_actor = 0
        state.opponent_discards = {1: [], 2: [], 3: []}
        state.opponent_packs = {1: [], 2: [], 3: []}
        state.seen_tiles = defaultdict(int)
        policy = self.policy_factory(state)
        action = policy.choose_action()
        scores = [10, -3, -3, -4] if action == 'PLAY W1' else [2, -1, -1, 0]
        return {'winner': 0 if action == 'PLAY W1' else 2, 'fan': 8, 'scores': scores}


class TestModelGovernance(unittest.TestCase):
    def test_model_registry_entry_contains_checksum_and_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_path = os.path.join(tmp, 'model.h5')
            codec = ActionCodec()
            model = CnnPolicyValueModel(action_space_size=codec.size, hidden_size=8, learning_rate=0.01)
            model.save(model_path)

            entry = build_model_registry_entry(
                model_path=model_path,
                version='v1.2.3',
                training_corpus=['data/external_trajectories.jsonl'],
                metrics={'candidate_win_rate': 0.63},
                extra_metadata={'phase': 'phase_6'},
            )

            self.assertEqual(entry['version'], 'v1.2.3')
            self.assertTrue(entry['checksum_sha256'])
            self.assertEqual(entry['training_corpus'], ['data/external_trajectories.jsonl'])
            self.assertEqual(entry['metrics']['candidate_win_rate'], 0.63)
            self.assertEqual(entry['extra_metadata']['phase'], 'phase_6')

            registry_path = os.path.join(tmp, 'registry.json')
            payload = write_model_registry(registry_path, [entry])
            with open(registry_path, 'r', encoding='utf-8') as handle:
                on_disk = json.load(handle)
            self.assertEqual(payload['count'], 1)
            self.assertEqual(on_disk['count'], 1)
            self.assertEqual(on_disk['entries'][0]['version'], 'v1.2.3')

    def test_duplicate_wall_evaluation_pairs_same_seeds(self):
        class _CandidatePolicy:
            def choose_action(self):
                return 'PLAY W1'

        class _BaselinePolicy:
            def choose_action(self):
                return 'PLAY W2'

        def candidate_policy_factory(_state):
            return _CandidatePolicy()

        def baseline_policy_factory(_state):
            return _BaselinePolicy()

        summary = duplicate_wall_evaluation(
            seeds=[11, 12, 13],
            game_factory=_DuplicateWallGame,
            candidate_policy_factory=candidate_policy_factory,
            baseline_policy_factory=baseline_policy_factory,
            quan=0,
        )

        self.assertEqual(summary['games'], 3)
        self.assertGreater(summary['avg_score_delta'], 0.0)
        self.assertGreater(summary['avg_win_delta'], 0.0)
        self.assertEqual(len(summary['paired_results']), 3)

    def test_elo_update_and_sprt_gate(self):
        candidate_rating, baseline_rating = elo_update(1500, 1500, score_a=1.0, k_factor=32)
        self.assertGreater(candidate_rating, 1500)
        self.assertLess(baseline_rating, 1500)

        ladder = update_elo_ladder(
            ratings={'baseline': 1500, 'candidate': 1500},
            match_results=[{'player_a': 'candidate', 'player_b': 'baseline', 'score_a': 1.0}],
            k_factor=32,
        )
        self.assertEqual(ladder[0]['player'], 'candidate')
        self.assertGreater(ladder[0]['rating'], ladder[1]['rating'])

        accepted = sprt_promotion_gate(
            {'games': 20, 'candidate_wins': 18, 'baseline_wins': 1, 'draws': 1, 'avg_score_delta': 3.0},
            min_win_rate=0.55,
            alpha=0.05,
            beta=0.2,
            min_avg_score_delta=0.0,
        )
        rejected = sprt_promotion_gate(
            {'games': 20, 'candidate_wins': 7, 'baseline_wins': 12, 'draws': 1, 'avg_score_delta': -1.0},
            min_win_rate=0.55,
            alpha=0.05,
            beta=0.2,
            min_avg_score_delta=0.0,
        )
        self.assertEqual(accepted['decision'], 'accept')
        self.assertEqual(rejected['decision'], 'reject')

    def test_regression_suite_legal_and_fallback_consistency(self):
        gs = GameState()
        gs.my_id = 0
        gs.hand = ['W1', 'W2', 'W3', 'B1', 'B2', 'B3', 'T1']
        gs.last_request_type = 2
        gs.last_request_action = 'DRAW'
        gs.last_tile = 'W4'
        gs.last_actor = 0
        gs.opponent_discards = {1: [], 2: [], 3: []}
        gs.opponent_packs = {1: [], 2: [], 3: []}
        gs.seen_tiles = {}

        bot = MahjongBot()
        fallback = GoalBasedPolicy(gs).choose_action()
        self.assertTrue(bot._is_legal_action(gs, fallback))
        self.assertEqual(fallback, GoalBasedPolicy(gs).choose_action())

        policy = NeuralPolicy(gs, model_path='c:/does/not/exist.h5')
        self.assertEqual(policy.choose_action(), fallback)

        actions = gs.enumerate_legal_actions()
        self.assertTrue(actions)
        for action in actions:
            self.assertTrue(bot._is_legal_action(gs, action))

    def test_deterministic_seed_replays_match(self):
        first = Game(quan=0, seed=77).run()
        second = Game(quan=0, seed=77).run()

        self.assertEqual(first['winner'], second['winner'])
        self.assertEqual(first['fan'], second['fan'])
        self.assertEqual(first['self_drawn'], second['self_drawn'])
        self.assertEqual(first['scores'], second['scores'])
        self.assertEqual(first['turn_logs'], second['turn_logs'])


if __name__ == '__main__':
    unittest.main()