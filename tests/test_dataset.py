import os
import sys
import tempfile
import unittest
ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'src'))
from dataset import JsonlTrajectoryReader, JsonlTrajectoryWriter, TrajectoryRecord

class TestDatasetIO(unittest.TestCase):

    def test_write_then_read_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset_path = os.path.join(tmp, 'sample.jsonl')
            writer = JsonlTrajectoryWriter(dataset_path)
            writer.write(TrajectoryRecord(game_id='g1', turn_index=0, player_id=0, request_type=2, request_action='DRAW', action='PLAY W1', legal_actions=['PASS', 'PLAY W1'], reward=0.0, done=False, features={'meta': [0]}))
            writer.write(TrajectoryRecord(game_id='g1', turn_index=1, player_id=1, request_type=3, request_action='PLAY', action='PASS', legal_actions=['PASS', 'HU'], reward=0.5, done=True, features={'meta': [1]}))
            writer.close()
            records = list(JsonlTrajectoryReader(dataset_path))
            self.assertEqual(len(records), 2)
            self.assertEqual(records[0].action, 'PLAY W1')
            self.assertEqual(records[1].done, True)
if __name__ == '__main__':
    unittest.main()
