"""Adapter for the IJCAI Mahjong Competition imitation checkpoints.

The 27 `data/models/imit_*.pkl` files are fused Torch state dicts of the
competition starter-kit architecture: 38 observation planes on a 4x9 tile
grid, a 40-block residual tower (128 channels), and a 235-way action head.
They are frozen evaluation/league opponents - never training labels.

Observation layout (starter-kit `feature.py` conventions):
- plane 0: seat wind marker (F1..F4 cell), plane 1: prevalent wind marker;
- planes 2-5: own hand count thresholds (>=1..>=4);
- planes 6-21: per-player meld tile thresholds, relative seat order
  (self, next, opposite, previous), 4 planes each;
- planes 22-37: per-player discard-history thresholds, same order.
Tiles are laid out W1..W9, T1..T9, B1..B9, F1..F4+J1..J3 on 4 rows of 9.

Action layout: Pass 0, Hu 1, Play 2+t, Chi 36+(suit*7+mid-2)*3+claimed_pos,
Peng 99+t, Gang 133+t, AnGang 167+t, BuGang 201+t.

Claim actions in this scheme do not carry the follow-up discard, so PENG/CHI
are resolved in two stages exactly like the original agent: first the claim
decision, then a Play decision over the post-meld hand.
"""

import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Starter-kit tile order differs from ours (W T B vs W B T).
FINALIST_TILE_LIST = (
    ['W%d' % i for i in range(1, 10)]
    + ['T%d' % i for i in range(1, 10)]
    + ['B%d' % i for i in range(1, 10)]
    + ['F%d' % i for i in range(1, 5)]
    + ['J%d' % i for i in range(1, 4)]
)
_FINALIST_TILE_INDEX = {tile: idx for idx, tile in enumerate(FINALIST_TILE_LIST)}

OBS_PLANES = 38
ACT_SIZE = 235
_OFFSET_HAND = 2
_OFFSET_PACKS = 6
_OFFSET_HISTORY = 22
_ACT_PASS = 0
_ACT_HU = 1
_ACT_PLAY = 2
_ACT_CHI = 36
_ACT_PENG = 99
_ACT_GANG = 133
_ACT_ANGANG = 167
_ACT_BUGANG = 201

_CHI_SUIT_INDEX = {'W': 0, 'T': 1, 'B': 2}


class _FinalistBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.c2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

    def forward(self, x):
        return F.relu(x + self.c2(F.relu(self.c1(x))))


class _FinalistNet(nn.Module):
    def __init__(self, blocks=40):
        super().__init__()
        self.stem = nn.Conv2d(OBS_PLANES, 128, kernel_size=3, padding=1)
        self.body = nn.ModuleList([_FinalistBlock() for _ in range(blocks)])
        self.foot = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 36, 512),
            nn.ReLU(),
            nn.Linear(512, ACT_SIZE),
        )

    def forward(self, x):
        x = F.relu(self.stem(x))
        for block in self.body:
            x = block(x)
        return self.foot(x)


class FinalistModel:
    """A frozen finalist checkpoint with masked-logit action selection."""

    def __init__(self, path, device='cpu'):
        state_dict = torch.load(path, map_location='cpu', weights_only=False)
        block_ids = {int(key.split('.')[1]) for key in state_dict if key.startswith('body.')}
        self.net = _FinalistNet(blocks=max(block_ids) + 1 if block_ids else 40)
        self.net.load_state_dict(state_dict, strict=True)
        self.net.eval()
        self.device = torch.device(device)
        self.net.to(self.device)
        self.name = os.path.splitext(os.path.basename(path))[0]

    def masked_logits(self, observation, action_ids):
        with torch.no_grad():
            obs = torch.from_numpy(observation).unsqueeze(0).to(self.device)
            logits = self.net(obs)[0]
            mask = torch.full((ACT_SIZE,), float('-inf'), device=self.device)
            ids = sorted(set(int(a) for a in action_ids))
            mask[ids] = 0.0
            return (logits + mask).cpu()

    def best_action_id(self, observation, action_ids):
        masked = self.masked_logits(observation, action_ids)
        return int(torch.argmax(masked).item())


def _mark_thresholds(obs, base_plane, tile, count):
    idx = _FINALIST_TILE_INDEX.get(tile)
    if idx is None:
        return
    row, col = divmod(idx, 9)
    for level in range(min(int(count), 4)):
        obs[base_plane + level][row][col] = 1.0


def build_observation(state):
    """Encode our GameState into the finalist 38x4x9 observation."""
    obs = np.zeros((OBS_PLANES, 4, 9), dtype=np.float32)

    seat_tile = 'F%d' % ((int(state.my_id) % 4) + 1)
    prevalent_tile = 'F%d' % ((int(state.quan) % 4) + 1)
    for plane, wind_tile in ((0, seat_tile), (1, prevalent_tile)):
        idx = _FINALIST_TILE_INDEX[wind_tile]
        obs[plane][idx // 9][idx % 9] = 1.0

    hand_counts = {}
    for tile in state.hand:
        hand_counts[tile] = hand_counts.get(tile, 0) + 1
    for tile, count in hand_counts.items():
        _mark_thresholds(obs, _OFFSET_HAND, tile, count)

    def _pack_tiles(packs):
        counts = {}
        for ptype, ptile, _offer in packs:
            if ptile is None:
                continue
            if ptype == 'PENG':
                counts[ptile] = counts.get(ptile, 0) + 3
            elif ptype == 'GANG':
                counts[ptile] = counts.get(ptile, 0) + 4
            elif ptype == 'CHI':
                mid_val = int(ptile[1:])
                for value in (mid_val - 1, mid_val, mid_val + 1):
                    seq_tile = '%s%d' % (ptile[0], value)
                    counts[seq_tile] = counts.get(seq_tile, 0) + 1
        return counts

    relative_packs = {0: state.packs}
    relative_history = {0: state.discards}
    for pid, packs in state.opponent_packs.items():
        relative_packs[(pid - state.my_id) % 4] = packs
    for pid, discards in state.opponent_discards.items():
        relative_history[(pid - state.my_id) % 4] = discards

    for relative_seat in range(4):
        pack_counts = _pack_tiles(relative_packs.get(relative_seat, []))
        for tile, count in pack_counts.items():
            _mark_thresholds(obs, _OFFSET_PACKS + 4 * relative_seat, tile, count)
        history_counts = {}
        for tile in relative_history.get(relative_seat, []):
            history_counts[tile] = history_counts.get(tile, 0) + 1
        for tile, count in history_counts.items():
            _mark_thresholds(obs, _OFFSET_HISTORY + 4 * relative_seat, tile, count)

    return obs


def _chi_action_id(mid_tile, claimed_tile):
    suit = mid_tile[0]
    mid_val = int(mid_tile[1:])
    claimed_val = int(claimed_tile[1:])
    claimed_pos = claimed_val - (mid_val - 1)
    return _ACT_CHI + (_CHI_SUIT_INDEX[suit] * 7 + mid_val - 2) * 3 + claimed_pos


def _tile_id(tile):
    return _FINALIST_TILE_INDEX[tile]


class FinalistOpponentPolicy:
    """Drives a finalist checkpoint through our GameState interface.

    Stage 1 maps our legal actions onto the 235-way claim/act space; if the
    network picks PENG or CHI, stage 2 asks it again for the follow-up
    discard over the post-meld hand, mirroring the original agent's two-step
    protocol.
    """

    def __init__(self, state, model):
        self.state = state
        self.model = model

    def choose_action(self):
        state = self.state
        legal_actions = state.enumerate_legal_actions()
        if not legal_actions:
            raise ValueError('No legal actions available')
        if len(legal_actions) == 1:
            return legal_actions[0]

        observation = build_observation(state)
        stage_one = {}
        for action in legal_actions:
            parts = action.split()
            verb = parts[0]
            if verb == 'PASS':
                stage_one.setdefault(_ACT_PASS, action)
            elif verb == 'HU':
                stage_one.setdefault(_ACT_HU, action)
            elif verb == 'PLAY':
                stage_one.setdefault(_ACT_PLAY + _tile_id(parts[1]), action)
            elif verb == 'GANG' and len(parts) == 1:
                stage_one.setdefault(_ACT_GANG + _tile_id(state.last_tile), action)
            elif verb == 'GANG':
                stage_one.setdefault(_ACT_ANGANG + _tile_id(parts[1]), action)
            elif verb == 'BUGANG':
                stage_one.setdefault(_ACT_BUGANG + _tile_id(parts[1]), action)
            elif verb == 'PENG':
                stage_one.setdefault(_ACT_PENG + _tile_id(state.last_tile), None)
            elif verb == 'CHI':
                stage_one.setdefault(_chi_action_id(parts[1], state.last_tile), None)

        chosen_id = self.model.best_action_id(observation, list(stage_one.keys()))
        direct = stage_one.get(chosen_id)
        if direct is not None:
            return direct

        if _ACT_PENG <= chosen_id < _ACT_GANG:
            meld_actions = [a for a in legal_actions if a.startswith('PENG ')]
            discards = [action.split()[1] for action in meld_actions]
            used = [state.last_tile, state.last_tile]
            mid_tile = None
        else:
            chi_index = chosen_id - _ACT_CHI
            suit = ('W', 'T', 'B')[chi_index // 3 // 7]
            mid_val = (chi_index // 3) % 7 + 2
            mid_tile = '%s%d' % (suit, mid_val)
            meld_actions = [
                a for a in legal_actions if a.startswith('CHI %s ' % mid_tile)
            ]
            discards = [action.split()[2] for action in meld_actions]
            seq = ['%s%d' % (suit, mid_val - 1), mid_tile, '%s%d' % (suit, mid_val + 1)]
            used = [t for t in seq if t != state.last_tile]

        if not meld_actions:
            raise ValueError('Finalist chose meld id %d with no matching action' % chosen_id)

        stage_two_obs = self._post_meld_observation(used, chosen_id, mid_tile)
        play_ids = {_ACT_PLAY + _tile_id(d): d for d in discards}
        play_choice = self.model.best_action_id(stage_two_obs, list(play_ids.keys()))
        discard = play_ids[play_choice]
        if mid_tile is None:
            return 'PENG %s' % discard
        return 'CHI %s %s' % (mid_tile, discard)

    def _post_meld_observation(self, used_tiles, chosen_id, mid_tile):
        state = self.state
        original_hand = state.hand
        original_packs = state.packs
        reduced = list(original_hand)
        for tile in used_tiles:
            reduced.remove(tile)
        new_packs = list(original_packs)
        if mid_tile is None:
            new_packs.append(('PENG', state.last_tile, state.last_actor))
        else:
            mid_val = int(mid_tile[1:])
            seq = ['%s%d' % (mid_tile[0], mid_val - 1), mid_tile, '%s%d' % (mid_tile[0], mid_val + 1)]
            new_packs.append(('CHI', mid_tile, seq.index(state.last_tile)))
        try:
            state.hand = reduced
            state.packs = new_packs
            return build_observation(state)
        finally:
            state.hand = original_hand
            state.packs = original_packs


def load_finalist_models(models_dir, device='cpu', limit=None):
    """Load finalist checkpoints from a directory; bad files fail fast."""
    paths = sorted(
        os.path.join(models_dir, name)
        for name in os.listdir(models_dir)
        if name.endswith('.pkl')
    )
    if limit is not None:
        paths = paths[: int(limit)]
    return [FinalistModel(path, device=device) for path in paths]
