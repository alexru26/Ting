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
        self.assertEqual(features['schema_version'], 2)
        self.assertEqual(len(features['hand_counts']), 34)
        self.assertEqual(len(features['seen_counts']), 34)
        self.assertEqual(len(features['self_discard_counts']), 34)
        self.assertEqual(len(features['pack_counts']), 34)
        self.assertEqual(len(features['opponent_discard_counts']), 3)
        self.assertEqual(len(features['opponent_discard_counts'][0]), 34)
        self.assertEqual(len(features['opponent_temporal']), 3)
        self.assertIn('hand_shanten_norm', features)
        self.assertIn('acceptancy_norm', features)
        self.assertIn('action_efficiency_deltas', features)
        self.assertEqual(len(features['meta']), 8)

    def test_normalization_invariants(self):
        extractor = FeatureExtractor()
        features = extractor.extract(self._make_state())
        for key in ['hand_counts_norm', 'seen_counts_norm', 'self_discard_counts_norm', 'pack_counts_norm']:
            self.assertTrue(all(0.0 <= value <= 1.0 for value in features[key]))
        self.assertTrue(-1.0 <= features['action_efficiency_deltas']['PLAY'] <= 1.0)

    def test_temporal_uses_full_history_lengths(self):
        extractor = FeatureExtractor()
        state = self._make_state()
        state.opponent_discards[1].extend(['W1', 'W2', 'W3'])
        features = extractor.extract(state)
        temporal = features['opponent_temporal']
        lengths = [row['full_history_length'] for row in temporal]
        self.assertIn(len(state.opponent_discards[1]), lengths)

    def test_deterministic_output(self):
        extractor = FeatureExtractor()
        state = self._make_state()
        f1 = extractor.extract(state)
        f2 = extractor.extract(state)
        self.assertEqual(f1, f2)
if __name__ == '__main__':
    unittest.main()
