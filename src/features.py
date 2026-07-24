"""Deterministic state-to-feature conversion.

Schema v5 contract (shared by runtime inference, dataset export, and the
training pipeline):

- `tile_planes`: PLANE_COUNT x 34 grid of floats in [0, 1].
- `meta`: flat vector of META_COUNT floats.

The extractor is intentionally cheap: one fan calculation at most (for the
can-win signal) plus cached shanten/acceptance analysis.

The final three planes are oracle planes (opponent hand counts by relative
seat). They are only populated during oracle-guided RL in the simulator -
`state.oracle_hands` / `state.oracle_scale` - and are always zero at
inference time and in supervised data, so the deployed policy never depends
on information it will not have.
"""

from tiles import ALL_TILES, TILE_TO_IDX, min_shanten, useful_tiles

FEATURE_SCHEMA_VERSION = 5

TILE_COUNT = len(ALL_TILES)

# Plane layout (each row is a 34-wide vector).
_PLANE_LAYOUT = (
    'hand_ge1',
    'hand_ge2',
    'hand_ge3',
    'hand_ge4',
    'own_melds',
    'opp1_melds',
    'opp2_melds',
    'opp3_melds',
    'own_discards',
    'opp1_discards',
    'opp2_discards',
    'opp3_discards',
    'unseen_ge1',
    'unseen_ge2',
    'unseen_ge3',
    'unseen_ge4',
    'last_tile',
    'useful_tiles',
    'oracle_opp1_hand',
    'oracle_opp2_hand',
    'oracle_opp3_hand',
)
PLANE_COUNT = len(_PLANE_LAYOUT)
ORACLE_PLANE_COUNT = 3

_LAST_ACTION_VOCAB = ('DRAW', 'PLAY', 'PENG', 'CHI', 'GANG', 'BUGANG')

# Meta layout: seat(4) + quan(4) + phase(3) + relative last actor(4)
# + last action(6+1 none) + 9 scalars.
META_COUNT = 4 + 4 + 3 + 4 + (len(_LAST_ACTION_VOCAB) + 1) + 9


def _count_vector(tiles):
    counts = [0] * TILE_COUNT
    for tile in tiles:
        idx = TILE_TO_IDX.get(tile)
        if idx is not None:
            counts[idx] += 1
    return counts


def _threshold_planes(counts, thresholds=(1, 2, 3, 4)):
    return [[1.0 if count >= threshold else 0.0 for count in counts] for threshold in thresholds]


def _norm_plane(counts, divisor=4.0):
    return [min(1.0, max(0.0, float(count) / divisor)) for count in counts]


def _meld_tile_counts(packs):
    """Expand meld packs into per-tile counts (PENG=3, GANG=4, CHI=1 each)."""
    counts = [0] * TILE_COUNT
    for ptype, ptile, _offer in packs:
        if ptile is None:
            continue
        idx = TILE_TO_IDX.get(ptile)
        if idx is None:
            continue
        if ptype == 'PENG':
            counts[idx] += 3
        elif ptype == 'GANG':
            counts[idx] += 4
        elif ptype == 'CHI':
            suit = ptile[0]
            mid_val = int(ptile[1:])
            for value in (mid_val - 1, mid_val, mid_val + 1):
                seq_idx = TILE_TO_IDX.get('%s%d' % (suit, value))
                if seq_idx is not None:
                    counts[seq_idx] += 1
    return counts


class FeatureExtractor:
    """Converts a GameState into the schema v4 feature dict."""

    def extract(self, state):
        hand = list(state.hand)
        hand_counts = _count_vector(hand)
        meld_count = len(state.packs)

        planes = []
        planes.extend(_threshold_planes(hand_counts))
        planes.append(_norm_plane(_meld_tile_counts(state.packs)))

        opponent_ids = {(pid - state.my_id) % 4: pid for pid in state.opponent_packs}
        for relative_seat in (1, 2, 3):
            pid = opponent_ids.get(relative_seat)
            packs = state.opponent_packs.get(pid, []) if pid is not None else []
            planes.append(_norm_plane(_meld_tile_counts(packs)))

        planes.append(_norm_plane(_count_vector(state.discards)))
        discard_ids = {(pid - state.my_id) % 4: pid for pid in state.opponent_discards}
        for relative_seat in (1, 2, 3):
            pid = discard_ids.get(relative_seat)
            discards = state.opponent_discards.get(pid, []) if pid is not None else []
            planes.append(_norm_plane(_count_vector(discards)))

        unseen = []
        for idx, tile in enumerate(ALL_TILES):
            remaining = 4 - int(state.seen_tiles.get(tile, 0)) - hand_counts[idx]
            unseen.append(max(0, remaining))
        planes.extend(_threshold_planes(unseen))

        last_tile_plane = [0.0] * TILE_COUNT
        last_idx = TILE_TO_IDX.get(state.last_tile) if state.last_tile else None
        if last_idx is not None:
            last_tile_plane[last_idx] = 1.0
        planes.append(last_tile_plane)

        shanten = self._safe_shanten(hand, meld_count)
        useful = self._safe_useful(hand, meld_count)
        useful_plane = [0.0] * TILE_COUNT
        for tile in useful:
            idx = TILE_TO_IDX.get(tile)
            if idx is not None:
                useful_plane[idx] = 1.0
        planes.append(useful_plane)

        planes.extend(self._oracle_planes(state))

        meta = self._meta_vector(state, shanten, len(useful))

        return {
            'schema_version': FEATURE_SCHEMA_VERSION,
            'tile_planes': planes,
            'meta': meta,
            'request_type': int(state.last_request_type),
            'request_action': state.last_request_action,
        }

    def _meta_vector(self, state, shanten, acceptance_count):
        meta = []
        meta.extend(self._one_hot(state.my_id % 4, 4))
        meta.extend(self._one_hot(state.quan % 4, 4))

        request_type = int(state.last_request_type)
        request_action = state.last_request_action
        if request_type == 2:
            phase = 0
        elif request_type == 3 and request_action == 'PLAY':
            phase = 1
        elif request_type == 3 and request_action == 'BUGANG':
            phase = 2
        else:
            phase = None
        meta.extend(self._one_hot(phase, 3))

        if state.last_actor is None:
            relative_actor = 0
        else:
            relative_actor = (int(state.last_actor) - int(state.my_id)) % 4
        meta.extend(self._one_hot(relative_actor, 4))

        try:
            action_index = _LAST_ACTION_VOCAB.index(request_action)
        except ValueError:
            action_index = len(_LAST_ACTION_VOCAB)
        meta.extend(self._one_hot(action_index, len(_LAST_ACTION_VOCAB) + 1))

        win_fan = state.current_win_fan() if hasattr(state, 'current_win_fan') else 0
        total_discards = len(state.discards) + sum(
            len(rows) for rows in state.opponent_discards.values()
        )
        total_seen = sum(state.seen_tiles.values())

        meta.extend(
            [
                min(1.0, float(state.flowers) / 8.0),
                min(1.0, float(len(state.packs)) / 4.0),
                min(1.0, float(total_discards) / 70.0),
                min(1.0, float(total_seen) / 136.0),
                min(1.0, max(0.0, float(shanten) / 8.0)),
                1.0 if shanten <= 0 else 0.0,
                min(1.0, float(acceptance_count) / 34.0),
                1.0 if win_fan >= 8 else 0.0,
                min(1.0, float(win_fan) / 16.0),
            ]
        )
        return meta

    @staticmethod
    def _oracle_planes(state):
        """Opponent hand-count planes, populated only during oracle-guided RL.

        `state.oracle_hands` maps relative seat (1..3) to that opponent's
        hidden hand; `state.oracle_scale` anneals the curriculum from 1 to 0.
        Oracle features never enter the uint8 supervised cache (they are all
        zero outside the RL simulator), so they may leave the 0.25 grid.
        """
        oracle_hands = getattr(state, 'oracle_hands', None)
        scale = min(1.0, max(0.0, float(getattr(state, 'oracle_scale', 1.0) or 0.0)))
        planes = []
        for relative_seat in (1, 2, 3):
            if not oracle_hands or scale <= 0.0:
                planes.append([0.0] * TILE_COUNT)
                continue
            counts = _count_vector(oracle_hands.get(relative_seat, []))
            planes.append([min(1.0, count / 4.0) * scale for count in counts])
        return planes

    @staticmethod
    def _one_hot(index, size):
        vector = [0.0] * size
        if index is not None and 0 <= int(index) < size:
            vector[int(index)] = 1.0
        return vector

    @staticmethod
    def _safe_shanten(hand, meld_count):
        try:
            return int(min_shanten(list(hand), int(meld_count)))
        except Exception:
            return 8

    @staticmethod
    def _safe_useful(hand, meld_count):
        try:
            return useful_tiles(list(hand), int(meld_count))
        except Exception:
            return set()
