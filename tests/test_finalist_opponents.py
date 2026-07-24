import os
import sys
import tempfile
import unittest
ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'src'))

import torch

from finalist_opponents import (
    ACT_SIZE,
    FinalistModel,
    FinalistOpponentPolicy,
    _FinalistNet,
    _chi_action_id,
    build_observation,
)
from state import GameState


def _tiny_checkpoint(tmp):
    path = os.path.join(tmp, 'tiny_finalist.pkl')
    torch.save(_FinalistNet(blocks=1).state_dict(), path)
    return path


def _draw_state():
    state = GameState()
    state.my_id = 1
    state.quan = 2
    state.hand = ['W1', 'W1', 'W2', 'W3', 'B4', 'B5', 'B6', 'T7', 'T8', 'T9', 'J1', 'J1', 'F1', 'F2']
    state.opponent_discards = {0: ['W9'], 2: [], 3: []}
    state.opponent_packs = {0: [('PENG', 'T1', 1)], 2: [], 3: []}
    state.last_request_type = 2
    state.last_tile = 'F2'
    state.last_actor = 1
    return state


class TestObservation(unittest.TestCase):

    def test_observation_shape_and_wind_markers(self):
        obs = build_observation(_draw_state())
        self.assertEqual(obs.shape, (38, 4, 9))
        # Seat wind F2 for seat 1 sits in the honor row (row 3, col 1).
        self.assertEqual(obs[0][3][1], 1.0)
        # Prevalent wind F3 for quan 2 (row 3, col 2).
        self.assertEqual(obs[1][3][2], 1.0)

    def test_hand_thresholds_and_relative_packs(self):
        obs = build_observation(_draw_state())
        # W1 x2 in hand -> planes 2 and 3 set at row 0 col 0.
        self.assertEqual(obs[2][0][0], 1.0)
        self.assertEqual(obs[3][0][0], 1.0)
        self.assertEqual(obs[4][0][0], 0.0)
        # Opponent seat 0 is relative seat 3 for my_id=1: pack planes 6+4*3=18.
        # T1 peng = 3 copies -> planes 18..20 at bamboo row (row 1, col 0).
        self.assertEqual(obs[18][1][0], 1.0)
        self.assertEqual(obs[20][1][0], 1.0)
        self.assertEqual(obs[21][1][0], 0.0)
        # Their W9 discard in history planes 22+4*3=34.
        self.assertEqual(obs[34][0][8], 1.0)

    def test_chi_action_ids(self):
        self.assertEqual(_chi_action_id('W2', 'W1'), 36 + 0 * 3 + 0)
        self.assertEqual(_chi_action_id('W2', 'W2'), 36 + 0 * 3 + 1)
        self.assertEqual(_chi_action_id('B8', 'B9'), 36 + (2 * 7 + 6) * 3 + 2)


class TestFinalistPolicy(unittest.TestCase):

    def test_choose_action_is_legal_on_draw(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = FinalistModel(_tiny_checkpoint(tmp))
        state = _draw_state()
        action = FinalistOpponentPolicy(state, model).choose_action()
        self.assertIn(action, state.enumerate_legal_actions())

    def test_choose_action_resolves_meld_discard(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = FinalistModel(_tiny_checkpoint(tmp))
        state = GameState()
        state.my_id = 1
        state.hand = ['W5', 'W5', 'W5', 'B1', 'B2', 'B3', 'T1', 'T2', 'T3', 'J1', 'J1', 'F1', 'F2']
        state.opponent_discards = {0: [], 2: [], 3: []}
        state.opponent_packs = {0: [], 2: [], 3: []}
        state.last_request_type = 3
        state.last_request_action = 'PLAY'
        state.last_tile = 'W5'
        state.last_actor = 0
        legal = state.enumerate_legal_actions()
        for _ in range(3):
            action = FinalistOpponentPolicy(state, model).choose_action()
            self.assertIn(action, legal)

    def test_masked_logits_respects_mask(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = FinalistModel(_tiny_checkpoint(tmp))
        obs = build_observation(_draw_state())
        masked = model.masked_logits(obs, [0, 1])
        self.assertEqual(int(torch.isfinite(masked).sum().item()), 2)
        self.assertTrue(0 <= model.best_action_id(obs, [0, 1]) <= 1)
        self.assertEqual(masked.shape[0], ACT_SIZE)


if __name__ == '__main__':
    unittest.main()
