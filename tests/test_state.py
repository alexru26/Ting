import os
import sys
import unittest
ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'src'))
from state import GameState


def _draw_state(hand, last_tile, packs=None):
    state = GameState()
    state.my_id = 0
    state.quan = 0
    state.hand = list(hand)
    state.packs = list(packs or [])
    state.last_request_type = 2
    state.last_tile = last_tile
    state.last_actor = 0
    state.last_request_action = None
    return state


def _play_response_state(hand, discarded_tile, actor, my_id=1, packs=None):
    state = GameState()
    state.my_id = my_id
    state.quan = 0
    state.hand = list(hand)
    state.packs = list(packs or [])
    state.last_request_type = 3
    state.last_request_action = 'PLAY'
    state.last_tile = discarded_tile
    state.last_actor = actor
    return state


class TestLegalActionSoundness(unittest.TestCase):
    """Regression tests for the Botzone INVALID-move bug: every action in
    enumerate_legal_actions must be accepted by the judge."""

    def test_pass_is_not_legal_after_draw(self):
        state = _draw_state(
            ['W1', 'W1', 'W2', 'W3', 'B4', 'B5', 'B6', 'T7', 'T8', 'T9', 'J1', 'J1', 'F1', 'F2'],
            'F2',
        )
        self.assertNotIn('PASS', state.enumerate_legal_actions())

    def test_hu_requires_winning_hand(self):
        state = _draw_state(
            ['W1', 'W1', 'W2', 'W3', 'B4', 'B5', 'B6', 'T7', 'T8', 'T9', 'J1', 'J1', 'F1', 'F2'],
            'F2',
        )
        self.assertNotIn('HU', state.enumerate_legal_actions())

    def test_hu_offered_for_seven_pairs_self_draw(self):
        state = _draw_state(
            ['W1', 'W1', 'W2', 'W2', 'W3', 'W3', 'B1', 'B1', 'B2', 'B2', 'B3', 'B3', 'T1', 'T1'],
            'T1',
        )
        actions = state.enumerate_legal_actions()
        self.assertIn('HU', actions)

    def test_concealed_gang_requires_four_copies(self):
        hand = ['W9', 'W9', 'W9', 'W1', 'W2', 'W3', 'B4', 'B5', 'B6', 'T7', 'T8', 'T9', 'J1', 'J1']
        state = _draw_state(hand, 'J1')
        self.assertNotIn('GANG W9', state.enumerate_legal_actions())
        state = _draw_state(hand + ['W9'], 'W9')
        state.hand.remove('J1')
        self.assertIn('GANG W9', state.enumerate_legal_actions())

    def test_bugang_requires_fourth_tile_in_hand(self):
        hand_without = ['W1', 'W2', 'W3', 'B4', 'B5', 'B6', 'T7', 'T8', 'T9', 'J1', 'J1']
        state = _draw_state(hand_without, 'J1', packs=[('PENG', 'W5', 2)])
        self.assertNotIn('BUGANG W5', state.enumerate_legal_actions())
        state = _draw_state(hand_without[:-1] + ['W5'], 'W5', packs=[('PENG', 'W5', 2)])
        self.assertIn('BUGANG W5', state.enumerate_legal_actions())

    def test_peng_cannot_discard_consumed_copies(self):
        hand = ['W5', 'W5', 'B1', 'B2', 'B3', 'T1', 'T2', 'T3', 'J1', 'J1', 'F1', 'F2', 'F3']
        state = _play_response_state(hand, 'W5', actor=0)
        actions = state.enumerate_legal_actions()
        self.assertNotIn('PENG W5', actions)
        self.assertIn('PENG B1', actions)

    def test_peng_can_discard_third_copy_when_held(self):
        hand = ['W5', 'W5', 'W5', 'B1', 'B2', 'B3', 'T1', 'T2', 'T3', 'J1', 'J1', 'F1', 'F2']
        state = _play_response_state(hand, 'W5', actor=0)
        actions = state.enumerate_legal_actions()
        self.assertIn('PENG W5', actions)
        self.assertIn('GANG', actions)

    def test_chi_cannot_discard_consumed_tiles(self):
        hand = ['W1', 'W2', 'B1', 'B2', 'B3', 'T1', 'T2', 'T3', 'J1', 'J1', 'F1', 'F2', 'F3']
        state = _play_response_state(hand, 'W3', actor=0)
        actions = state.enumerate_legal_actions()
        self.assertNotIn('CHI W2 W1', actions)
        self.assertNotIn('CHI W2 W2', actions)
        self.assertIn('CHI W2 B1', actions)

    def test_chi_only_from_left_neighbor(self):
        hand = ['W1', 'W2', 'B1', 'B2', 'B3', 'T1', 'T2', 'T3', 'J1', 'J1', 'F1', 'F2', 'F3']
        state = _play_response_state(hand, 'W3', actor=2, my_id=1)
        actions = state.enumerate_legal_actions()
        self.assertFalse(any(action.startswith('CHI') for action in actions))

    def test_own_discard_allows_only_pass(self):
        hand = ['W5', 'W5', 'B1', 'B2', 'B3', 'T1', 'T2', 'T3', 'J1', 'J1', 'F1', 'F2', 'F3']
        state = _play_response_state(hand, 'W5', actor=1, my_id=1)
        self.assertEqual(state.enumerate_legal_actions(), ['PASS'])

    def test_forced_pass_for_non_actionable_events(self):
        state = GameState()
        state.my_id = 0
        state.hand = ['W1', 'W2', 'W3']
        state.last_request_type = 3
        state.last_request_action = 'DRAW'
        state.last_actor = 2
        self.assertEqual(state.enumerate_legal_actions(), ['PASS'])

    def test_bugang_response_offers_pass_without_win(self):
        hand = ['W1', 'W2', 'W3', 'B4', 'B5', 'B6', 'T7', 'T8', 'T9', 'J1', 'J1', 'F1', 'F2']
        state = GameState()
        state.my_id = 1
        state.hand = hand
        state.last_request_type = 3
        state.last_request_action = 'BUGANG'
        state.last_tile = 'W5'
        state.last_actor = 0
        self.assertEqual(state.enumerate_legal_actions(), ['PASS'])


class TestSeenTileAccounting(unittest.TestCase):

    def test_chi_does_not_double_count_claimed_tile(self):
        state = GameState()
        state.my_id = 0
        state.opponent_discards = {1: [], 2: [], 3: []}
        state.opponent_packs = {1: [], 2: [], 3: []}
        state.seen_tiles['W3'] = 1
        state._last_discard = 'W3'
        state._last_discard_player = 0
        state._apply_type3(['3', '1', 'CHI', 'W3', 'B1'], None)
        self.assertEqual(state.seen_tiles['W3'], 1)
        self.assertEqual(state.seen_tiles['W2'], 1)
        self.assertEqual(state.seen_tiles['W4'], 1)
        self.assertEqual(state.seen_tiles['B1'], 1)

    def test_open_gang_adds_three_newly_seen(self):
        state = GameState()
        state.my_id = 0
        state.opponent_discards = {1: [], 2: [], 3: []}
        state.opponent_packs = {1: [], 2: [], 3: []}
        state.seen_tiles['T5'] = 1
        state._last_discard = 'T5'
        state._last_discard_player = 0
        state._last_drawer = 0
        state._apply_type3(['3', '1', 'GANG'], None)
        self.assertEqual(state.seen_tiles['T5'], 4)

    def test_concealed_opponent_gang_is_hidden(self):
        state = GameState()
        state.my_id = 0
        state.opponent_discards = {1: [], 2: [], 3: []}
        state.opponent_packs = {1: [], 2: [], 3: []}
        state._last_discard = 'T5'
        state._last_discard_player = 0
        state._apply_type3(['3', '1', 'DRAW'], None)
        state._apply_type3(['3', '1', 'GANG'], None)
        self.assertEqual(state.seen_tiles['T5'], 0)
        self.assertEqual(state.opponent_packs[1], [('GANG', None, 1)])


class TestGameStateParsingRobustness(unittest.TestCase):

    def test_parse_from_history_ignores_invalid_first_request_type(self):
        state = GameState()
        state.parse_from_history(['2 W1'], [])
        self.assertEqual(state.last_request_type, -1)
        self.assertEqual(state.hand, [])

    def test_parse_from_history_ignores_malformed_init_and_deal(self):
        state = GameState()
        state.parse_from_history(['0 0', '1 0'], [])
        self.assertEqual(state.last_request_type, -1)
        self.assertEqual(state.hand, [])

    def test_apply_type3_chi_ignores_invalid_mid_tile(self):
        state = GameState()
        state.my_id = 0
        state.hand = ['W1', 'W2', 'W4']
        state._last_discard = 'W3'
        state._last_discard_player = 3
        state._apply_type3(['3', '1', 'CHI', 'W0', 'W4'], None)
        self.assertEqual(state.opponent_packs.get(1, []), [])
        self.assertEqual(state._last_discard, 'W3')

    def test_set_current_request_ignores_malformed_payload(self):
        state = GameState()
        state._set_current_request('3')
        self.assertEqual(state.last_request_type, -1)

    def test_full_history_reconstruction(self):
        requests = [
            '0 0 0',
            '1 0 0 0 0 W1 W2 W3 B4 B5 B6 T7 T8 T9 J1 J1 F1 F2',
            '2 W4',
            '3 0 PLAY F2',
            '3 1 DRAW',
            '3 1 PLAY B9',
        ]
        responses = ['PASS', 'PASS', 'PLAY F2', 'PASS', 'PASS']
        state = GameState.from_history(requests, responses)
        self.assertEqual(state.my_id, 0)
        self.assertEqual(len(state.hand), 13)
        self.assertNotIn('F2', state.hand)
        self.assertIn('W4', state.hand)
        self.assertEqual(state.last_request_action, 'PLAY')
        self.assertEqual(state.last_tile, 'B9')
        self.assertEqual(state.opponent_discards[1], ['B9'])


if __name__ == '__main__':
    unittest.main()
