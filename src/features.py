import math

from tiles import ALL_TILES, TILE_TO_IDX
from tiles import NUMBERED_SUITS, min_shanten, tile_value, useful_tiles
from scoring import calculate_fan


FEATURE_SCHEMA_VERSION = 3
_COUNT_NORM_DIVISOR = 4.0
_DELTA_NORM_SCALE = 2.0
_FAN_NORM_DIVISOR = 16.0


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
        fan_now, can_hu_now = self._fan_now(state)
        tenpai = self._tenpai_profile(state, state.hand, len(state.packs))
        action_fan_values, action_fan_deltas = self._action_conditioned_fan_features(state, tenpai)

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
            'request_type': int(state.last_request_type),
            'seat': int(state.my_id),
            'target_player': True,
            'event_action': state.last_request_action,
            'raw_request': state.last_request_action,
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
            'can_hu_now': 1.0 if can_hu_now else 0.0,
            'fan_if_hu_now': fan_now,
            'fan_if_hu_now_norm': self.normalize_fan(fan_now),
            'fan_gap_to_8': max(0.0, 8.0 - float(fan_now)),
            'fan_gap_to_8_norm': self.normalize_fan(max(0.0, 8.0 - float(fan_now))),
            'is_tenpai': 1.0 if tenpai['is_tenpai'] else 0.0,
            'wait_count': tenpai['wait_count'],
            'wait_count_norm': self.normalize_count_like(tenpai['wait_count']),
            'wait_fan_min': tenpai['wait_fan_min'],
            'wait_fan_max': tenpai['wait_fan_max'],
            'wait_fan_mean': tenpai['wait_fan_mean'],
            'wait_fan_min_norm': self.normalize_fan(tenpai['wait_fan_min']),
            'wait_fan_max_norm': self.normalize_fan(tenpai['wait_fan_max']),
            'wait_fan_mean_norm': self.normalize_fan(tenpai['wait_fan_mean']),
            'action_efficiency_deltas': efficiency_deltas,
            'action_fan_values': action_fan_values,
            'action_fan_deltas': action_fan_deltas,
            'meta': meta,
        }

    def _fan_now(self, state):
        request_type = int(state.last_request_type)
        request_action = state.last_request_action
        tile = state.last_tile
        if not tile:
            return 0.0, False

        hand = list(state.hand)
        is_self_drawn = bool(request_type == 2)
        is_about_kong = bool(request_type == 3 and request_action == 'BUGANG')
        if is_self_drawn and tile in hand:
            hand.remove(tile)
        if not self._can_score_hand_shape(hand, len(state.packs)):
            return 0.0, False

        try:
            fan = float(
                calculate_fan(
                    state.fan_calc_packs(),
                    tuple(hand),
                    tile,
                    int(state.flowers),
                    is_self_drawn,
                    False,
                    is_about_kong,
                    False,
                    int(state.my_id) % 4,
                    int(state.quan),
                )
            )
        except Exception:
            fan = 0.0
        return fan, fan >= 8.0

    def _wait_fans_for_hand(self, state, hand, meld_count):
        shanten = self._safe_shanten(hand, meld_count)
        if shanten > 0:
            return []
        if not self._can_score_hand_shape(hand, meld_count):
            return []

        fans = []
        packs = state.fan_calc_packs()
        for tile in ALL_TILES:
            seen = float(state.seen_tiles.get(tile, 0))
            if seen >= 4.0:
                continue
            try:
                fan = float(
                    calculate_fan(
                        packs,
                        tuple(hand),
                        tile,
                        int(state.flowers),
                        False,
                        False,
                        False,
                        False,
                        int(state.my_id) % 4,
                        int(state.quan),
                    )
                )
            except Exception:
                fan = 0.0
            if fan > 0.0:
                fans.append(fan)
        return fans

    def _tenpai_profile(self, state, hand, meld_count):
        fans = self._wait_fans_for_hand(state, hand, meld_count)
        if not fans:
            return {
                'is_tenpai': False,
                'wait_count': 0,
                'wait_fan_min': 0.0,
                'wait_fan_max': 0.0,
                'wait_fan_mean': 0.0,
            }

        return {
            'is_tenpai': True,
            'wait_count': int(len(fans)),
            'wait_fan_min': float(min(fans)),
            'wait_fan_max': float(max(fans)),
            'wait_fan_mean': float(sum(fans) / float(len(fans))),
        }

    def _best_wait_mean_after_discard(self, state, hand, meld_count):
        if not hand:
            return 0.0
        best = None
        for tile in set(hand):
            reduced = list(hand)
            try:
                reduced.remove(tile)
            except ValueError:
                continue
            profile = self._tenpai_profile(state, reduced, meld_count)
            value = float(profile['wait_fan_mean'])
            if best is None or value > best:
                best = value
        if best is None:
            return 0.0
        return float(best)

    def _best_peng_wait_mean(self, state):
        if state.last_request_type != 3 or state.last_request_action != 'PLAY' or not state.last_tile:
            return 0.0
        tile = state.last_tile
        hand = list(state.hand)
        if hand.count(tile) < 2:
            return 0.0
        hand.remove(tile)
        hand.remove(tile)
        return self._best_wait_mean_after_discard(state, hand, len(state.packs) + 1)

    def _best_chi_wait_mean(self, state):
        if state.last_request_type != 3 or state.last_request_action != 'PLAY' or not state.last_tile:
            return 0.0
        tile = state.last_tile
        if tile[0] not in NUMBERED_SUITS:
            return 0.0
        if state.last_actor is None or (int(state.last_actor) + 1) % 4 != int(state.my_id):
            return 0.0

        suit = tile[0]
        value = tile_value(tile)
        best = None
        for mid in range(max(2, value - 1), min(8, value + 1) + 1):
            seq = ['%s%d' % (suit, mid - 1), '%s%d' % (suit, mid), '%s%d' % (suit, mid + 1)]
            if tile not in seq:
                continue
            needed = [t for t in seq if t != tile]
            hand = list(state.hand)
            if not all(t in hand for t in needed):
                continue
            for needed_tile in needed:
                hand.remove(needed_tile)
            score = self._best_wait_mean_after_discard(state, hand, len(state.packs) + 1)
            if best is None or score > best:
                best = score
        if best is None:
            return 0.0
        return float(best)

    @staticmethod
    def _can_score_hand_shape(hand, meld_count):
        expected = 13 - 3 * int(meld_count)
        if expected < 1:
            expected = 1
        return len(list(hand or [])) == expected

    def _best_gang_wait_mean(self, state):
        hand = list(state.hand)
        base_meld_count = len(state.packs)
        best = None

        for tile in set(hand):
            if hand.count(tile) >= 4:
                reduced = [item for item in hand if item != tile]
                score = self._tenpai_profile(state, reduced, base_meld_count + 1)['wait_fan_mean']
                if best is None or score > best:
                    best = score

        peng_tiles = {ptile for ptype, ptile, _ in state.packs if ptype == 'PENG'}
        for tile in peng_tiles:
            if tile in hand:
                reduced = list(hand)
                reduced.remove(tile)
                score = self._tenpai_profile(state, reduced, base_meld_count + 1)['wait_fan_mean']
                if best is None or score > best:
                    best = score

        if best is None:
            return 0.0
        return float(best)

    def _action_conditioned_fan_features(self, state, tenpai_profile):
        baseline = float(tenpai_profile.get('wait_fan_mean', 0.0))
        fan_now, _can_hu_now = self._fan_now(state)
        values = {
            'PASS': baseline,
            'HU': float(fan_now),
            'PLAY': self._best_wait_mean_after_discard(state, list(state.hand), len(state.packs)),
            'GANG': self._best_gang_wait_mean(state),
            'BUGANG': self._best_gang_wait_mean(state),
            'PENG': self._best_peng_wait_mean(state),
            'CHI': self._best_chi_wait_mean(state),
        }
        deltas = {}
        for family, value in values.items():
            deltas[family] = self.normalize_delta(float(value) - baseline, scale=4.0)
        return values, deltas

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
    def normalize_fan(value):
        value = float(value)
        return min(1.0, max(0.0, value / _FAN_NORM_DIVISOR))

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
