import os
import sys
import tempfile
import unittest
ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'src'))

import random

from model import CnnPolicyValueModel
from rl_self_play import (
    OpponentLeague,
    backfill_decayed_returns,
    evaluate_against_baseline,
    evaluate_against_finalists,
    promotion_gate,
    run_ppo_fine_tuning,
    shape_rewards,
)


class TestRewardShaping(unittest.TestCase):

    def test_score_delta_only(self):
        result = {'winner': 0, 'fan': 10, 'scores': [30, -10, -10, -10]}
        rewards = shape_rewards(result)
        self.assertEqual(rewards, [30.0, -10.0, -10.0, -10.0])

    def test_fan_and_placement_weights(self):
        result = {'winner': 1, 'fan': 10, 'scores': [-10, 30, -10, -10]}
        rewards = shape_rewards(result, fan_weight=1.0, placement_weight=1.0)
        self.assertAlmostEqual(rewards[1], 30.0 + 10.0 + 1.0)


class TestPromotionGate(unittest.TestCase):

    def test_gate_accepts_strong_summary(self):
        evaluation = {
            'games': 16,
            'candidate_wins': 0,
            'baseline_wins': 0,
            'draws': 0,
            'candidate_win_rate': 0.75,
            'avg_score_delta': 4.0,
        }
        self.assertTrue(promotion_gate(evaluation))

    def test_gate_rejects_weak_summary(self):
        evaluation = {
            'games': 16,
            'candidate_wins': 0,
            'baseline_wins': 0,
            'draws': 0,
            'candidate_win_rate': 0.1,
            'avg_score_delta': -5.0,
        }
        self.assertFalse(promotion_gate(evaluation))


class TestFanBackwardCredit(unittest.TestCase):

    def test_decayed_returns_and_advantages(self):
        buffer = [{'value': 0.0} for _ in range(3)]
        backfill_decayed_returns(buffer, episode_return=1.0, credit_gamma=0.5)
        self.assertAlmostEqual(buffer[0]['return_target'], 0.25)
        self.assertAlmostEqual(buffer[1]['return_target'], 0.5)
        self.assertAlmostEqual(buffer[2]['return_target'], 1.0)
        self.assertEqual(buffer[0]['win_target'], 1.0)
        self.assertAlmostEqual(buffer[2]['advantage'], 1.0)

    def test_losses_decay_toward_early_turns(self):
        buffer = [{'value': 0.0} for _ in range(2)]
        backfill_decayed_returns(buffer, episode_return=-0.8, credit_gamma=0.9)
        self.assertAlmostEqual(buffer[0]['return_target'], -0.72)
        self.assertEqual(buffer[0]['win_target'], 0.0)


class TestOpponentLeague(unittest.TestCase):

    def test_probabilities_zeroed_without_pools(self):
        league = OpponentLeague(random.Random(1), finalist_models=[], historical_models=[], candidate_model=None)
        kinds = {league.sample_seat()[0] for _ in range(20)}
        self.assertEqual(kinds, {'rule'})

    def test_sampling_covers_configured_pools(self):
        league = OpponentLeague(
            random.Random(2),
            finalist_models=['finalist'],
            historical_models=['hist'],
            candidate_model='self',
            finalist_prob=0.4,
            historical_prob=0.3,
            self_play_prob=0.2,
        )
        kinds = [league.sample_seat()[0] for _ in range(300)]
        self.assertIn('finalist', kinds)
        self.assertIn('historical', kinds)
        self.assertIn('self', kinds)
        self.assertIn('rule', kinds)


class TestPpoPipeline(unittest.TestCase):

    def test_ppo_train_smoke(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_path = os.path.join(tmp, 'model.h5')
            out_path = os.path.join(tmp, 'model_ppo.h5')
            CnnPolicyValueModel(channels=8, blocks=1, hidden_size=32).save(model_path)
            summary = run_ppo_fine_tuning(
                model_path=model_path,
                games=1,
                eval_games=1,
                seed=13,
                update_every=1,
                device='cpu',
                out_path=out_path,
            )
            self.assertEqual(summary['episodes'], 1)
            self.assertGreater(summary['updates'], 0)
            self.assertTrue(os.path.exists(out_path))
            self.assertIn('evaluation', summary)
            self.assertIn(summary['promoted'], (True, False))
            loaded = CnnPolicyValueModel.load(out_path)
            self.assertEqual(loaded.channels, 8)

    def test_evaluate_against_baseline_smoke(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_path = os.path.join(tmp, 'model.h5')
            CnnPolicyValueModel(channels=8, blocks=1, hidden_size=32).save(model_path)
            summary = evaluate_against_baseline(model_path=model_path, games=1, seed=21)
            self.assertEqual(summary['games'], 1)
            self.assertEqual(len(summary['mean_scores']), 4)

    def test_ppo_train_learning_rate_and_kl_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_path = os.path.join(tmp, 'model.h5')
            out_path = os.path.join(tmp, 'model_ppo.h5')
            CnnPolicyValueModel(channels=8, blocks=1, hidden_size=32).save(model_path)
            summary = run_ppo_fine_tuning(
                model_path=model_path,
                games=1,
                eval_games=1,
                seed=13,
                update_every=1,
                device='cpu',
                out_path=out_path,
                learning_rate=0.00005,
                target_kl=0.02,
                snapshot_every=1,
            )
            self.assertEqual(summary['learning_rate'], 0.00005)
            self.assertEqual(summary['target_kl'], 0.02)
            self.assertIn('kl_stops', summary)
            self.assertTrue(os.path.exists(out_path + '.snapshot.h5'))

    def test_evaluate_against_finalists_smoke(self):
        finalist_dir = os.path.join(ROOT, 'data', 'models')
        if not os.path.isdir(finalist_dir):
            self.skipTest('finalist checkpoints not available')
        with tempfile.TemporaryDirectory() as tmp:
            model_path = os.path.join(tmp, 'model.h5')
            CnnPolicyValueModel(channels=8, blocks=1, hidden_size=32).save(model_path)
            summary = evaluate_against_finalists(
                model_path=model_path,
                finalist_dir=finalist_dir,
                games=1,
                seed=5,
                finalist_limit=3,
            )
            self.assertEqual(summary['games'], 1)
            self.assertEqual(len(summary['finalists']), 3)
            total = summary['candidate_wins'] + summary['opponent_wins'] + summary['draws']
            self.assertEqual(total, 1)


if __name__ == '__main__':
    unittest.main()
