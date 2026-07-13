import json
import os
import sys
from typing import Any, Dict, List

from state import GameState
from policy import create_policy
from tiles import ALL_TILES, tile_value

class MahjongBot:
    """
    Botzone-compatible entry point.

    Each turn Botzone sends one JSON object on stdin:
        {"requests": [...all requests so far...], "responses": [...all previous responses...]}
    The bot writes one JSON object to stdout:
        {"response": "<action>"}
    """

    def handle_input(self, payload):
        requests = payload.get('requests', [])
        responses = payload.get('responses', [])
        if not requests:
            return 'PASS'
        try:
            state = GameState.from_history(requests, responses)
        except Exception:
            return 'PASS'
        if state.last_request_type in (-1, 0, 1):
            return 'PASS'
        policy_mode = os.getenv('TING_POLICY_MODE')
        policy_model_path = os.getenv('TING_POLICY_MODEL_PATH')
        policy = create_policy(state, mode=policy_mode, model_path=policy_model_path)
        action = policy.choose_action()
        if self._is_legal_action(state, action):
            return action
        return self._fallback_action(state)

    def _is_legal_action(self, state, action):
        if not action:
            return False
        parts = action.split()
        verb = parts[0]
        request_type = state.last_request_type
        request_action = state.last_request_action
        if request_type == 2:
            if verb in ('PASS', 'HU'):
                return True
            if verb == 'PLAY' and len(parts) == 2:
                return parts[1] in state.hand
            if verb == 'GANG' and len(parts) == 2:
                return state.hand.count(parts[1]) >= 4
            if verb == 'BUGANG' and len(parts) == 2:
                return any((ptype == 'PENG' and ptile == parts[1] for ptype, ptile, _ in state.packs))
            return False
        if request_type == 3:
            if request_action == 'PLAY':
                if verb == 'PASS' or verb == 'HU':
                    return True
                if verb == 'GANG' and len(parts) == 1:
                    return state.hand.count(state.last_tile or '') >= 3
                if verb == 'PENG' and len(parts) == 2:
                    return state.hand.count(state.last_tile or '') >= 2 and parts[1] in state.hand
                if verb == 'CHI' and len(parts) == 3:
                    return self._can_chi(state, parts[1], parts[2])
                return False
            if request_action in ('GANG', 'BUGANG', 'PENG', 'CHI', 'DRAW', 'BUHUA'):
                return verb == 'PASS'
        return verb == 'PASS'

    def _can_chi(self, state, mid_tile, discard_tile):
        if not state.last_tile or state.last_tile[0] not in 'WBT':
            return False
        if len(mid_tile) < 2 or len(discard_tile) < 2:
            return False
        if mid_tile not in ALL_TILES or discard_tile not in ALL_TILES:
            return False
        suit = mid_tile[0]
        if suit != state.last_tile[0] or discard_tile not in state.hand:
            return False
        try:
            mid_val = tile_value(mid_tile)
        except ValueError:
            return False
        if mid_val < 2 or mid_val > 8:
            return False
        seq = [f'{suit}{mid_val - 1}', mid_tile, f'{suit}{mid_val + 1}']
        if any((tile not in ALL_TILES for tile in seq)):
            return False
        if state.last_tile not in seq:
            return False
        needed = [t for t in seq if t != state.last_tile]
        return all((t in state.hand for t in needed)) and state.last_actor is not None and ((state.last_actor + 1) % 4 == state.my_id)

    def _fallback_action(self, state):
        if state.last_request_type == 2:
            if state.hand:
                return 'PLAY ' + state.hand[0]
            return 'PASS'
        return 'PASS'

    def run(self):
        raw = sys.stdin.read().strip()
        if not raw:
            print(json.dumps({'response': 'PASS'}))
            return
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            print(json.dumps({'response': 'PASS'}))
            return
        response = self.handle_input(payload)
        print(json.dumps({'response': response}))

        
if __name__ == '__main__':
    MahjongBot().run()
