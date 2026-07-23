import os
import sys
import tempfile
import unittest
ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'src'))

from dataset import JsonlTrajectoryReader, JsonlTrajectoryWriter
from local_game import Game, REWARD_SCALE


class TestGameSimulation(unittest.TestCase):

    def test_game_completes_and_scores_are_zero_sum(self):
        game = Game(quan=0, seed=5)
        result = game.run()
        self.assertIn('winner', result)
        self.assertEqual(sum(result['scores']), 0)
        self.assertEqual(len(result['scores']), 4)

    def test_self_drawn_scoring(self):
        game = Game(quan=0, seed=1)
        game._win_fan = 0
        game.players[0].hand = ['W1', 'W1', 'W2', 'W2', 'W3', 'W3', 'B1', 'B1', 'B2', 'B2', 'B3', 'B3', 'T1', 'T1']
        game._resolve_win(0, 'T1', is_self_drawn=True)
        fan = game._win_fan
        self.assertGreaterEqual(fan, 8)
        self.assertEqual(game.scores[0], 3 * (8 + fan))
        for pid in (1, 2, 3):
            self.assertEqual(game.scores[pid], -(8 + fan))

    def test_discard_win_scoring(self):
        game = Game(quan=0, seed=1)
        game.players[1].hand = ['W1', 'W1', 'W2', 'W2', 'W3', 'W3', 'B1', 'B1', 'B2', 'B2', 'B3', 'B3', 'T1']
        game._resolve_win(1, 'T1', is_self_drawn=False, from_player=2)
        fan = game._win_fan
        self.assertGreaterEqual(fan, 8)
        self.assertEqual(game.scores[1], 3 * 8 + fan)
        self.assertEqual(game.scores[2], -(8 + fan))
        self.assertEqual(game.scores[0], -8)
        self.assertEqual(game.scores[3], -8)
        self.assertEqual(sum(game.scores), 0)


class TestDatasetExport(unittest.TestCase):

    def test_records_are_legal_and_rewards_backfilled(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'game.jsonl')
            with JsonlTrajectoryWriter(path) as writer:
                game = Game(quan=0, seed=9, dataset_writer=writer, game_id='g0')
                result = game.run()
            records = list(JsonlTrajectoryReader(path))

        self.assertGreater(len(records), 0)
        done_players = set()
        for record in records:
            self.assertIn(record.action, record.legal_actions)
            self.assertEqual(record.features['schema_version'], 4)
            expected_reward = float(result['scores'][record.player_id]) / REWARD_SCALE
            self.assertAlmostEqual(record.reward, expected_reward, places=9)
            if record.done:
                done_players.add(record.player_id)
        self.assertEqual(done_players, {r.player_id for r in records})


if __name__ == '__main__':
    unittest.main()
