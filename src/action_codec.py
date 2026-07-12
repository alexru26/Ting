from typing import Dict, List
from tiles import ALL_TILES, NUMBERED_SUITS

class ActionCodec:
    """Deterministic action vocabulary for training and inference."""

    def __init__(self):
        self._actions = self._build_action_space()
        self._action_to_id = {action: idx for idx, action in enumerate(self._actions)}

    @property
    def size(self):
        return len(self._actions)

    def all_actions(self):
        return self._actions[:]

    def encode(self, action):
        normalized = self._normalize_action(action)
        if normalized not in self._action_to_id:
            raise ValueError(f'Unknown action: {action}')
        return self._action_to_id[normalized]

    def decode(self, action_id):
        if action_id < 0 or action_id >= len(self._actions):
            raise ValueError(f'Invalid action id: {action_id}')
        return self._actions[action_id]

    def _build_action_space(self):
        actions = ['PASS', 'HU', 'GANG']
        for tile in ALL_TILES:
            actions.append(f'PLAY {tile}')
        for tile in ALL_TILES:
            actions.append(f'GANG {tile}')
        for tile in ALL_TILES:
            actions.append(f'BUGANG {tile}')
        for tile in ALL_TILES:
            actions.append(f'PENG {tile}')
        for suit in NUMBERED_SUITS:
            for mid_value in range(2, 9):
                mid_tile = f'{suit}{mid_value}'
                for discard in ALL_TILES:
                    actions.append(f'CHI {mid_tile} {discard}')
        return actions

    @staticmethod
    def _normalize_action(action):
        return ' '.join(action.strip().split())
