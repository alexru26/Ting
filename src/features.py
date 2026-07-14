import math

from tiles import ALL_TILES, TILE_TO_IDX
from tiles import min_shanten, useful_tiles


FEATURE_SCHEMA_VERSION = 2
_COUNT_NORM_DIVISOR = 4.0
_DELTA_NORM_SCALE = 2.0


_LAST_ACTION_TO_ID = {
    None: 0,
    'DRAW': 1,
    'PLAY': 2,
    'PENG': 3,
    'CHI': 4,
    'GANG': 5,
    'BUGANG': 6,
    'BUHUA': 7,
}


class FeatureExtractor:
    """Deterministic state-to-feature conversion for ML pipelines."""

    def extract(self, state):
        hand_counts = self._count_tiles(state.hand)
        seen_counts = [state.seen_tiles.get(tile, 0) for tile in ALL_TILES]
        self_discard_counts = self._count_tiles(state.discards)

        pack_counts = [0] * len(ALL_TILES)
        for _, tile, _ in state.packs:
            idx = TILE_TO_IDX.get(tile)
            if idx is not None:
                pack_counts[idx] += 1

        opponent_discard_counts = []
        for pid in sorted(pid for pid in state.opponent_discards if pid != state.my_id):
            discards = state.opponent_discards.get(pid, [])
            opponent_discard_counts.append(self._count_tiles(discards))

        while len(opponent_discard_counts) < 3:
            opponent_discard_counts.append([0] * len(ALL_TILES))

        opponent_temporal = self._opponent_temporal_features(state)

        hand_shanten = self._safe_shanten(state.hand, len(state.packs))
        hand_acceptancy = self._acceptancy_count(state.hand, len(state.packs))
        efficiency_deltas = self._action_efficiency_deltas(state.hand, len(state.packs))

        hand_counts_norm = self._normalize_count_bucket(hand_counts)
        seen_counts_norm = self._normalize_count_bucket(seen_counts)
        self_discard_counts_norm = self._normalize_count_bucket(self_discard_counts)
        pack_counts_norm = self._normalize_count_bucket(pack_counts)
        opponent_discard_counts_norm = [self._normalize_count_bucket(bucket) for bucket in opponent_discard_counts]

        meta = [
            state.my_id,
            state.quan,
            state.flowers,
            len(state.packs),
            state.last_request_type,
            state.last_actor if state.last_actor is not None else -1,
            _LAST_ACTION_TO_ID.get(state.last_request_action, 0),
            TILE_TO_IDX.get(state.last_tile, -1) if state.last_tile else -1,
        ]

        return {
            'schema_version': FEATURE_SCHEMA_VERSION,
            'hand_counts': hand_counts,
            'seen_counts': seen_counts,
            'self_discard_counts': self_discard_counts,
            'pack_counts': pack_counts,
            'opponent_discard_counts': opponent_discard_counts,
            'opponent_temporal': opponent_temporal,
            'hand_counts_norm': hand_counts_norm,
            'seen_counts_norm': seen_counts_norm,
            'self_discard_counts_norm': self_discard_counts_norm,
            'pack_counts_norm': pack_counts_norm,
            'opponent_discard_counts_norm': opponent_discard_counts_norm,
            'hand_shanten': hand_shanten,
            'hand_shanten_norm': self.normalize_count_like(hand_shanten),
            'acceptancy': hand_acceptancy,
            'acceptancy_norm': self.normalize_count_like(hand_acceptancy),
            'action_efficiency_deltas': efficiency_deltas,
            'meta': meta,
        }

    def _safe_shanten(self, hand, meld_count):
        try:
            return int(min_shanten(list(hand or []), int(meld_count)))
        except Exception:
            return 8

    def _acceptancy_count(self, hand, meld_count):
        try:
            return int(len(useful_tiles(list(hand or []), int(meld_count))))
        except Exception:
            return 0

    def _action_efficiency_deltas(self, hand, meld_count):
        values = {
            'PLAY': 0.0,
            'GANG': 0.0,
            'BUGANG': 0.0,
        }
        if not hand:
            return {key: self.normalize_delta(values[key]) for key in values}

        baseline = self._safe_shanten(hand, meld_count)
        reduced_scores = []
        for tile in set(hand):
            next_hand = list(hand)
            try:
                next_hand.remove(tile)
            except ValueError:
                continue
            reduced_scores.append(self._safe_shanten(next_hand, meld_count))
        if reduced_scores:
            values['PLAY'] = float(baseline - min(reduced_scores))

        gang_candidates = [tile for tile in set(hand) if hand.count(tile) >= 4]
        if gang_candidates:
            best_gang_after = None
            for tile in gang_candidates:
                next_hand = [item for item in hand if item != tile]
                sh_after = self._safe_shanten(next_hand, meld_count + 1)
                if best_gang_after is None or sh_after < best_gang_after:
                    best_gang_after = sh_after
            if best_gang_after is not None:
                values['GANG'] = float(baseline - best_gang_after)

        values['BUGANG'] = values['GANG']
        return {key: self.normalize_delta(values[key]) for key in values}

    def _opponent_temporal_features(self, state):
        summaries = []
        opponent_ids = sorted(pid for pid in state.opponent_discards if pid != state.my_id)
        for pid in opponent_ids[:3]:
            discards = list(state.opponent_discards.get(pid, []))
            packs = list(state.opponent_packs.get(pid, []))
            recent = discards[-6:]
            honors = sum(1 for tile in discards if tile and tile[0] in ('F', 'J'))
            suit_counts = {
                'W': sum(1 for tile in discards if tile and tile[0] == 'W'),
                'B': sum(1 for tile in discards if tile and tile[0] == 'B'),
                'T': sum(1 for tile in discards if tile and tile[0] == 'T'),
            }
            summaries.append(
                {
                    'player_id': pid,
                    'full_history_length': len(discards),
                    'pack_count': len(packs),
                    'recent_honor_ratio': self._ratio(sum(1 for tile in recent if tile and tile[0] in ('F', 'J')), max(1, len(recent))),
                    'honor_ratio': self._ratio(honors, max(1, len(discards))),
                    'suit_ratios': {
                        'W': self._ratio(suit_counts['W'], max(1, len(discards))),
                        'B': self._ratio(suit_counts['B'], max(1, len(discards))),
                        'T': self._ratio(suit_counts['T'], max(1, len(discards))),
                    },
                }
            )
        while len(summaries) < 3:
            summaries.append(
                {
                    'player_id': -1,
                    'full_history_length': 0,
                    'pack_count': 0,
                    'recent_honor_ratio': 0.0,
                    'honor_ratio': 0.0,
                    'suit_ratios': {'W': 0.0, 'B': 0.0, 'T': 0.0},
                }
            )
        return summaries

    @staticmethod
    def _ratio(numerator, denominator):
        if denominator <= 0:
            return 0.0
        return float(numerator) / float(denominator)

    @staticmethod
    def normalize_count_like(value):
        value = float(value)
        return min(1.0, max(0.0, value / _COUNT_NORM_DIVISOR))

    @staticmethod
    def normalize_delta(value, scale=_DELTA_NORM_SCALE):
        value = float(value)
        scale = float(scale) if float(scale) > 0.0 else 1.0
        return float(math.tanh(value / scale))

    @staticmethod
    def renormalize_probabilities(values):
        positives = [max(0.0, float(value)) for value in list(values or [])]
        total = sum(positives)
        if total <= 0.0:
            if not positives:
                return []
            uniform = 1.0 / float(len(positives))
            return [uniform for _ in positives]
        return [value / total for value in positives]

    def _normalize_count_bucket(self, values):
        return [self.normalize_count_like(value) for value in list(values or [])]

    @staticmethod
    def _count_tiles(tiles):
        counts = [0] * len(ALL_TILES)
        for tile in tiles:
            idx = TILE_TO_IDX.get(tile)
            if idx is not None:
                counts[idx] += 1
        return counts
