import os
import sys
import unittest
ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'src'))

from features import FeatureExtractor, FEATURE_SCHEMA_VERSION, META_COUNT, PLANE_COUNT
from state import GameState
from tiles import TILE_TO_IDX


def _basic_state():
    state = GameState()
    state.my_id = 0
    state.quan = 1
    state.hand = ['W1', 'W1', 'W2', 'W3', 'B4', 'B5', 'B6', 'T7', 'T8', 'T9', 'J1', 'J1', 'F1', 'F2']
    state.opponent_discards = {1: [], 2: [], 3: []}
    state.opponent_packs = {1: [], 2: [], 3: []}
    state.last_request_type = 2
    state.last_tile = 'F2'
    state.last_actor = 0
    return state


class TestFeatureContract(unittest.TestCase):

    def test_shapes_and_ranges(self):
        features = FeatureExtractor().extract(_basic_state())
        self.assertEqual(features['schema_version'], FEATURE_SCHEMA_VERSION)
        self.assertEqual(len(features['tile_planes']), PLANE_COUNT)
        for plane in features['tile_planes']:
            self.assertEqual(len(plane), 34)
            for value in plane:
                self.assertGreaterEqual(value, 0.0)
                self.assertLessEqual(value, 1.0)
        self.assertEqual(len(features['meta']), META_COUNT)

    def test_planes_are_quantization_safe(self):
        """All plane values must sit on the 0.25 grid so the uint8 cache is lossless."""
        features = FeatureExtractor().extract(_basic_state())
        for plane in features['tile_planes']:
            for value in plane:
                self.assertAlmostEqual(round(value * 4.0), value * 4.0, places=9)

    def test_hand_threshold_planes(self):
        features = FeatureExtractor().extract(_basic_state())
        planes = features['tile_planes']
        w1 = TILE_TO_IDX['W1']
        w2 = TILE_TO_IDX['W2']
        self.assertEqual(planes[0][w1], 1.0)
        self.assertEqual(planes[1][w1], 1.0)
        self.assertEqual(planes[2][w1], 0.0)
        self.assertEqual(planes[0][w2], 1.0)
        self.assertEqual(planes[1][w2], 0.0)

    def test_last_tile_plane(self):
        features = FeatureExtractor().extract(_basic_state())
        last_plane = features['tile_planes'][16]
        self.assertEqual(last_plane[TILE_TO_IDX['F2']], 1.0)
        self.assertEqual(sum(last_plane), 1.0)

    def test_unseen_accounts_for_hand_and_seen(self):
        state = _basic_state()
        state.seen_tiles['W1'] = 2
        features = FeatureExtractor().extract(state)
        planes = features['tile_planes']
        w1 = TILE_TO_IDX['W1']
        # 4 total - 2 seen - 2 in hand = 0 unseen
        self.assertEqual(planes[12][w1], 0.0)
        b9 = TILE_TO_IDX['B9']
        self.assertEqual(planes[12][b9], 1.0)
        self.assertEqual(planes[15][b9], 1.0)

    def test_opponent_planes_use_relative_seats(self):
        state = _basic_state()
        state.my_id = 2
        state.opponent_discards = {3: ['W9'], 0: [], 1: []}
        state.opponent_packs = {3: [], 0: [], 1: []}
        features = FeatureExtractor().extract(state)
        planes = features['tile_planes']
        w9 = TILE_TO_IDX['W9']
        # Player 3 is my_id 2's next player (relative seat 1) -> opp1_discards plane.
        self.assertEqual(planes[9][w9], 0.25)
        self.assertEqual(planes[10][w9], 0.0)

    def test_meld_expansion(self):
        state = _basic_state()
        state.hand = state.hand[:11]
        state.packs = [('PENG', 'T1', 1)]
        features = FeatureExtractor().extract(state)
        own_melds = features['tile_planes'][4]
        self.assertEqual(own_melds[TILE_TO_IDX['T1']], 0.75)

    def test_meta_encodes_phase_and_seat(self):
        features = FeatureExtractor().extract(_basic_state())
        meta = features['meta']
        self.assertEqual(meta[0], 1.0)  # seat 0
        self.assertEqual(meta[5], 1.0)  # quan 1
        self.assertEqual(meta[8], 1.0)  # phase draw-decision

    def test_oracle_planes_default_to_zero(self):
        features = FeatureExtractor().extract(_basic_state())
        for plane in features['tile_planes'][18:]:
            self.assertEqual(sum(plane), 0.0)

    def test_oracle_planes_populated_and_annealed(self):
        state = _basic_state()
        state.oracle_hands = {1: ['W9', 'W9'], 2: [], 3: []}
        state.oracle_scale = 0.5
        features = FeatureExtractor().extract(state)
        oracle_plane = features['tile_planes'][18]
        self.assertAlmostEqual(oracle_plane[TILE_TO_IDX['W9']], 0.5 * 0.5)
        state.oracle_scale = 0.0
        features = FeatureExtractor().extract(state)
        self.assertEqual(sum(features['tile_planes'][18]), 0.0)


if __name__ == '__main__':
    unittest.main()
