import os
import sys
import tempfile
import unittest
ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'src'))

from dataset import JsonlTrajectoryReader, JsonlTrajectoryWriter, TrajectoryRecord


class TestTrajectoryRoundtrip(unittest.TestCase):

    def test_write_and_read_records(self):
        record = TrajectoryRecord(
            game_id='game-1',
            turn_index=3,
            player_id=2,
            request_type=2,
            request_action=None,
            action='PLAY W1',
            legal_actions=['PLAY W1', 'PLAY W2'],
            reward=0.5,
            done=True,
            features={'schema_version': 5},
            metadata={'note': 'test'},
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'data.jsonl')
            with JsonlTrajectoryWriter(path) as writer:
                writer.write(record)
                writer.write(record)
            loaded = list(JsonlTrajectoryReader(path))
        self.assertEqual(len(loaded), 2)
        first = loaded[0]
        self.assertEqual(first.game_id, 'game-1')
        self.assertEqual(first.action, 'PLAY W1')
        self.assertEqual(first.legal_actions, ['PLAY W1', 'PLAY W2'])
        self.assertEqual(first.reward, 0.5)
        self.assertTrue(first.done)
        self.assertEqual(first.features['schema_version'], 5)


if __name__ == '__main__':
    unittest.main()
