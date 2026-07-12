import os
import sys
import unittest
ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'src'))
from state import GameState
from features import FeatureExtractor

class TestFeatureExtractor(unittest.TestCase):

    def _make_state(self):
        gs = GameState()
        gs.my_id = 2
        gs.quan = 1
        gs.hand = ['W1', 'W1', 'B2', 'T9', 'F1']
        gs.packs = [('PENG', 'W3', 1)]
        gs.discards = ['J1', 'W9']
        gs.flowers = 2
        gs.last_request_type = 3
        gs.last_request_action = 'PLAY'
        gs.last_tile = 'W3'
        gs.last_actor = 1
        gs.opponent_discards = {0: ['B1', 'B2'], 1: ['W4'], 3: ['T1', 'T2', 'T3']}
        return gs

    def test_feature_shapes(self):
        extractor = FeatureExtractor()
        features = extractor.extract(self._make_state())
        self.assertEqual(len(features['hand_counts']), 34)
        self.assertEqual(len(features['seen_counts']), 34)
        self.assertEqual(len(features['self_discard_counts']), 34)
        self.assertEqual(len(features['pack_counts']), 34)
        self.assertEqual(len(features['opponent_discard_counts']), 3)
        self.assertEqual(len(features['opponent_discard_counts'][0]), 34)
        self.assertEqual(len(features['meta']), 8)

    def test_deterministic_output(self):
        extractor = FeatureExtractor()
        state = self._make_state()
        f1 = extractor.extract(state)
        f2 = extractor.extract(state)
        self.assertEqual(f1, f2)
if __name__ == '__main__':
    unittest.main()
