import random
import time

from tiles import ALL_TILES


def _safe_int(value, default_value):
    try:
        return int(value)
    except Exception:
        return default_value


def _safe_float(value, default_value):
    try:
        return float(value)
    except Exception:
        return default_value


def _normalize_probabilities(probabilities):
    values = [_safe_float(value, 0.0) for value in probabilities]
    total = sum(value for value in values if value > 0.0)
    if total <= 0.0:
        if not values:
            return [1.0 / float(len(ALL_TILES)) for _ in ALL_TILES]
        uniform = 1.0 / float(len(values))
        return [uniform for _ in values]
    return [max(0.0, value) / total for value in values]


def sample_hidden_tiles(belief_probabilities, sample_count=8, rng=None):
    rng = rng or random.Random(7)
    sample_count = max(0, _safe_int(sample_count, 0))
    if sample_count <= 0:
        return []

    probabilities = _normalize_probabilities(belief_probabilities)
    tiles = list(ALL_TILES[: len(probabilities)])
    if not tiles:
        return []

    try:
        return rng.choices(tiles, weights=probabilities, k=sample_count)
    except Exception:
        return [tiles[rng.randrange(len(tiles))] for _ in range(sample_count)]


def _copy_feature_list(values):
    if not isinstance(values, list):
        return []
    return [value for value in values]


def _decrement_tile_bucket(bucket, tile_name, amount=1):
    if not isinstance(bucket, list):
        return bucket
    result = list(bucket)
    try:
        tile_index = ALL_TILES.index(tile_name)
    except ValueError:
        return result
    if 0 <= tile_index < len(result):
        result[tile_index] = max(0.0, _safe_float(result[tile_index], 0.0) - float(amount))
    return result


def project_features_for_action(features, action):
    projected = dict(features or {})
    if not action:
        return projected

    parts = action.split()
    verb = parts[0]
    tile = parts[-1] if len(parts) > 1 else None

    if verb == 'PLAY' and tile:
        projected['hand_counts'] = _decrement_tile_bucket(_copy_feature_list(projected.get('hand_counts')), tile, 1)
        projected['self_discard_counts'] = _decrement_tile_bucket(_copy_feature_list(projected.get('self_discard_counts')), tile, -1)
        projected['seen_counts'] = _decrement_tile_bucket(_copy_feature_list(projected.get('seen_counts')), tile, -1)
    elif verb == 'GANG' and tile:
        projected['hand_counts'] = _decrement_tile_bucket(_copy_feature_list(projected.get('hand_counts')), tile, 4)
        projected['pack_counts'] = _decrement_tile_bucket(_copy_feature_list(projected.get('pack_counts')), tile, -1)
        projected['seen_counts'] = _decrement_tile_bucket(_copy_feature_list(projected.get('seen_counts')), tile, -4)
    elif verb == 'BUGANG' and tile:
        projected['hand_counts'] = _decrement_tile_bucket(_copy_feature_list(projected.get('hand_counts')), tile, 1)
        projected['pack_counts'] = _decrement_tile_bucket(_copy_feature_list(projected.get('pack_counts')), tile, -1)
        projected['seen_counts'] = _decrement_tile_bucket(_copy_feature_list(projected.get('seen_counts')), tile, -1)

    return projected


def _action_tile(action):
    if not action:
        return None
    parts = action.split()
    if not parts:
        return None
    if parts[0] in ('PLAY', 'GANG', 'BUGANG', 'PENG') and len(parts) > 1:
        return parts[-1]
    if parts[0] == 'CHI' and len(parts) > 2:
        return parts[2]
    return None


def _hidden_risk(action, hidden_tiles):
    tile = _action_tile(action)
    if not tile:
        return 0.0
    if not hidden_tiles:
        return 0.0
    hidden_hits = sum(1 for hidden_tile in hidden_tiles if hidden_tile == tile)
    return float(hidden_hits) / float(len(hidden_tiles))


class BoundedRolloutPlanner:
    def __init__(self, model, top_k=3, rollout_samples=8, budget_ms=10, disabled=False, belief_weight=0.5, efficiency_weight=0.2, seed=7):
        self.model = model
        self.top_k = max(1, _safe_int(top_k, 3))
        self.rollout_samples = max(0, _safe_int(rollout_samples, 8))
        self.budget_ms = max(0, _safe_int(budget_ms, 10))
        self.disabled = bool(disabled)
        self.belief_weight = max(0.0, _safe_float(belief_weight, 0.5))
        self.efficiency_weight = max(0.0, _safe_float(efficiency_weight, 0.2))
        self.seed = _safe_int(seed, 7)

    def enabled(self):
        return not self.disabled and self.budget_ms > 0 and self.model is not None

    def plan(self, features, legal_actions, belief_weight=0.0):
        if not self.enabled() or not legal_actions:
            return None

        base_info = self.model.policy_info_from_features(features, legal_actions, belief_weight=belief_weight)
        sorted_candidates = sorted(
            zip(base_info.get('actions', list(legal_actions)), base_info.get('probabilities', [0.0] * len(base_info.get('actions', list(legal_actions))))),
            key=lambda item: (-_safe_float(item[1], 0.0), item[0]),
        )

        ranked_actions = [action for action, _ in sorted_candidates[: self.top_k]]
        if not ranked_actions:
            ranked_actions = list(legal_actions)[: self.top_k]

        rng = random.Random(self.seed + _safe_int(features.get('seat', 0) if isinstance(features, dict) else 0, 0))
        deadline = time.monotonic() + (float(self.budget_ms) / 1000.0)
        hidden_tiles = sample_hidden_tiles(base_info.get('belief_probs', []), sample_count=self.rollout_samples, rng=rng)

        best_action = ranked_actions[0]
        best_score = None

        for action in ranked_actions:
            if time.monotonic() >= deadline:
                break

            projected_features = project_features_for_action(features, action)
            info = self.model.policy_info_from_features(projected_features, [action], belief_weight=0.0)
            leaf_value = _safe_float(info.get('value', self.model.estimate_value_from_features(projected_features)), 0.0)
            efficiency_bonus = _safe_float(info.get('efficiency_bonus', 0.0), 0.0)
            risk_penalty = _hidden_risk(action, hidden_tiles)

            risk_weight = self.belief_weight if belief_weight <= 0.0 else max(self.belief_weight, _safe_float(belief_weight, 0.0))
            score = leaf_value + self.efficiency_weight * efficiency_bonus - risk_weight * risk_penalty
            if best_score is None or score > best_score or (score == best_score and action < best_action):
                best_action = action
                best_score = score

        return best_action
