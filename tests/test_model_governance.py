import os
import sys
import unittest
ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'src'))

from model_governance import duplicate_wall_evaluation, sprt_promotion_gate


class _StubGame:
    def __init__(self, result):
        self._result = result

    def run(self):
        return self._result


class TestDuplicateWallEvaluation(unittest.TestCase):

    def test_paired_summary_uses_candidate_seat(self):
        results = {
            'candidate': {'winner': 0, 'scores': [24, -8, -8, -8]},
            'baseline': {'winner': 2, 'scores': [-8, -8, 24, -8]},
        }

        def game_factory(quan, seed, policy_factory):
            return _StubGame(results[policy_factory])

        summary = duplicate_wall_evaluation(
            seeds=[1, 2],
            game_factory=game_factory,
            candidate_policy_factory='candidate',
            baseline_policy_factory='baseline',
            candidate_seat=0,
        )
        self.assertEqual(summary['games'], 2)
        self.assertEqual(summary['candidate_wins'], 2)
        self.assertEqual(summary['baseline_wins'], 0)
        self.assertEqual(summary['avg_score_delta'], 32.0)
        self.assertEqual(summary['avg_win_delta'], 1.0)


class TestSprtPromotionGate(unittest.TestCase):

    def test_rejects_without_games(self):
        self.assertEqual(sprt_promotion_gate({})['decision'], 'reject')

    def test_accepts_dominant_candidate(self):
        summary = {
            'games': 40,
            'candidate_wins': 32,
            'baseline_wins': 4,
            'draws': 4,
            'avg_score_delta': 5.0,
        }
        self.assertEqual(sprt_promotion_gate(summary)['decision'], 'accept')

    def test_rejects_negative_score_delta(self):
        summary = {
            'games': 40,
            'candidate_wins': 30,
            'baseline_wins': 5,
            'draws': 5,
            'avg_score_delta': -3.0,
        }
        self.assertEqual(sprt_promotion_gate(summary)['decision'], 'reject')

    def test_summary_metrics_path(self):
        summary = {
            'games': 16,
            'candidate_wins': 0,
            'baseline_wins': 0,
            'draws': 0,
            'candidate_win_rate': 0.7,
            'avg_score_delta': 2.0,
        }
        self.assertEqual(sprt_promotion_gate(summary)['decision'], 'accept')


if __name__ == '__main__':
    unittest.main()
