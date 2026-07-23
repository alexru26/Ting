import os
import sys
import tempfile
import unittest
ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'src'))

from model import CnnPolicyValueModel
from rl_self_play import (
    evaluate_against_baseline,
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


if __name__ == '__main__':
    unittest.main()
