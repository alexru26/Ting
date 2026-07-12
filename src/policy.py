from collections import Counter
import os
from typing import List, Optional, Tuple, TYPE_CHECKING
from tiles import normalize_tiles, is_honor, tile_suit, tile_value, min_shanten, best_discard, shanten_standard, ALL_TILES, NUMBERED_SUITS
from scoring import GoalType, goal_utility, select_goal, best_discard_for_goal, calculate_fan, can_win
from action_codec import ActionCodec
from features import FeatureExtractor
from runtime_model import load_policy_model, choose_action_from_model
from search_planner import BoundedRolloutPlanner
if TYPE_CHECKING:
    from state import GameState


def _safe_float(value, default_value):
    try:
        return float(value)
    except Exception:
        return default_value


def _safe_int(value, default_value):
    try:
        return int(value)
    except Exception:
        return default_value


def _default_model_path():
    src_dir = os.path.dirname(os.path.abspath(__file__))
    candidate = os.path.join(src_dir, 'model.h5')
    if os.path.exists(candidate):
        return candidate
    return None


class GoalBasedPolicy:
    """
    Goal-based Mahjong agent policy.

    Decision flow each turn
    -----------------------
    draw  (type-2 request)
        1. Check self-draw win.
        2. Check concealed kong / supplement kong.
        3. Select goal, choose best discard (with defensive adjustment).

    play-response (type-3 PLAY request)
        1. Check win-on-discard.
        2. Check open kong.
        3. Check peng (if improves goal).
        4. Check chi from left-player (if improves goal).
        5. PASS.

    gang-response (type-3 GANG/BUGANG)
        1. Check rob-the-kong win (BUGANG only).
        2. PASS.
    """
    DEFENSE_WEIGHT = 0.4

    def __init__(self, state):
        self.state = state

    def choose_action(self):
        s = self.state
        rtype = s.last_request_type
        action = s.last_request_action
        if rtype == 2:
            return self._on_draw()
        if rtype == 3:
            if action == 'PLAY':
                return self._on_play(s.last_tile, s.last_actor)
            if action in ('GANG', 'BUGANG'):
                return self._on_gang(action)
            return 'PASS'
        return 'PASS'

    def _on_draw(self):
        s = self.state
        hand = s.hand[:]
        packs = s.packs
        drawn = s.last_tile
        if drawn:
            hand_check = hand[:]
            if drawn in hand_check:
                hand_check.remove(drawn)
            if can_win(hand_check, s.fan_calc_packs(), drawn, s.flowers, True, s.my_id % 4, s.quan):
                return 'HU'
        gang_tile = self._find_concealed_gang(hand, packs)
        if gang_tile and self._should_gang_concealed(gang_tile, hand, packs):
            return f'GANG {gang_tile}'
        bugang_tile = self._find_bugang(hand, packs)
        if bugang_tile and self._should_bugang(bugang_tile, hand, packs):
            return f'BUGANG {bugang_tile}'
        goal = select_goal(hand, packs)
        tile = self._choose_discard(hand, packs, goal)
        return f'PLAY {tile}'

    def _on_play(self, tile, from_player):
        if not tile:
            return 'PASS'
        s = self.state
        hand = s.hand[:]
        packs = s.packs
        if can_win(hand, s.fan_calc_packs(), tile, s.flowers, False, s.my_id % 4, s.quan):
            return 'HU'
        if hand.count(tile) >= 3:
            if self._should_gang_open(tile, hand, packs):
                return 'GANG'
        if hand.count(tile) >= 2:
            discard = self._peng_and_discard(tile, hand, packs)
            if discard is not None:
                return f'PENG {discard}'
        if from_player is not None and (from_player + 1) % 4 == s.my_id:
            result = self._chi_and_discard(tile, hand, packs)
            if result is not None:
                mid, discard = result
                return f'CHI {mid} {discard}'
        return 'PASS'

    def _on_gang(self, action):
        if action != 'BUGANG':
            return 'PASS'
        s = self.state
        tile = s.last_tile
        if not tile:
            return 'PASS'
        if can_win(s.hand, s.fan_calc_packs(), tile, s.flowers, False, s.my_id % 4, s.quan, is_about_kong=True):
            return 'HU'
        return 'PASS'

    def _choose_discard(self, hand, packs, goal):
        """Choose the best tile to discard balancing goal progress and safety."""
        candidates = list(set(hand))
        n = len(packs)
        if not candidates:
            return hand[0]

        def score(tile):
            reduced = hand[:]
            reduced.remove(tile)
            sh = min_shanten(reduced, n)
            offence = -sh
            danger = self._danger_score(tile)
            return offence - self.DEFENSE_WEIGHT * danger
        goal_tile = best_discard_for_goal(hand, packs, goal)
        best_tile = goal_tile
        best_sc = score(goal_tile) + 0.01
        for t in candidates:
            sc = score(t)
            if sc > best_sc:
                best_sc = sc
                best_tile = t
        return best_tile

    def _find_concealed_gang(self, hand, packs):
        counts = Counter(hand)
        for tile, cnt in counts.items():
            if cnt >= 4:
                return tile
        return None

    def _find_bugang(self, hand, packs):
        peng_tiles = {ptile for ptype, ptile, _ in packs if ptype == 'PENG'}
        for tile in hand:
            if tile in peng_tiles:
                return tile
        return None

    def _should_gang_concealed(self, gang_tile, hand, packs):
        """Gang is good if it doesn't hurt our shanten and draws us a new tile."""
        reduced = [t for t in hand if t != gang_tile]
        sh_before = min_shanten(hand, len(packs))
        sh_after = min_shanten(reduced, len(packs) + 1)
        return sh_after <= sh_before

    def _should_bugang(self, tile, hand, packs):
        return True

    def _should_gang_open(self, tile, hand, packs):
        """Open kong from a discard: only do it if our hand profits."""
        reduced = [t for t in hand if t != tile]
        reduced = reduced[1:]
        sh_before = min_shanten(hand, len(packs))
        sh_after = min_shanten(reduced, len(packs) + 1)
        return sh_after <= sh_before

    def _peng_and_discard(self, tile, hand, packs):
        """
        Decide whether to peng and which tile to discard afterwards.
        Returns the discard tile, or None to decline.
        """
        n = len(packs)
        sh_without = min_shanten(hand, n)
        new_hand = hand[:]
        new_hand.remove(tile)
        new_hand.remove(tile)
        goal = select_goal(new_hand, packs + [('PENG', tile, 0)])
        discard = self._choose_discard(new_hand, packs + [('PENG', tile, 0)], goal)
        post_peng = new_hand[:]
        post_peng.remove(discard)
        sh_with = min_shanten(post_peng, n + 1)
        if sh_with < sh_without:
            return discard
        if sh_with == sh_without and sh_without <= 1:
            return discard
        return None

    def _chi_and_discard(self, tile, hand, packs):
        """
        Try all valid chi sequences for `tile` from hand.
        Returns (mid_tile, discard) or None to decline.
        """
        if not tile or tile[0] not in NUMBERED_SUITS:
            return None
        suit = tile[0]
        val = tile_value(tile)
        n = len(packs)
        best_result = None
        best_sh = min_shanten(hand, n)
        for mid_val in range(max(1, val - 1), min(9, val + 1) + 1):
            seq = [f'{suit}{mid_val - 1}', f'{suit}{mid_val}', f'{suit}{mid_val + 1}']
            if any((v < 1 or v > 9 for v in [mid_val - 1, mid_val, mid_val + 1])):
                continue
            from_hand = [t for t in seq if t != tile]
            if all((t in hand for t in from_hand)):
                mid = f'{suit}{mid_val}'
                new_hand = hand[:]
                for t in from_hand:
                    new_hand.remove(t)
                goal = select_goal(new_hand, packs + [('CHI', mid, 0)])
                discard = self._choose_discard(new_hand, packs + [('CHI', mid, 0)], goal)
                post_chi = new_hand[:]
                post_chi.remove(discard)
                sh = min_shanten(post_chi, n + 1)
                if sh < best_sh:
                    best_sh = sh
                    best_result = (mid, discard)
        return best_result

    def _danger_score(self, tile):
        """
        Estimate how dangerous it is to discard `tile`.
        Higher score = more dangerous.

        Heuristics:
        - Tile is in an opponent's "wait zone" based on their discards
        - Opponent has many melds (closer to tenpai)
        - Tile is the 3rd/4th copy visible (fewer copies left = more dangerous)
        """
        s = self.state
        danger = 0.0
        seen = s.seen_tiles.get(tile, 0)
        remaining = 4 - seen
        if remaining <= 1:
            danger += 2.0
        elif remaining == 2:
            danger += 0.5
        for pid, opp_packs in s.opponent_packs.items():
            n_opp = len(opp_packs)
            if n_opp == 0:
                continue
            proximity = n_opp / 4.0
            opp_disc = s.opponent_discards.get(pid, [])
            tile_suit_char = tile[0]
            for ptype, ptile, _ in opp_packs:
                if ptile[0] == tile_suit_char:
                    danger += 0.5 * proximity
            if tile in opp_disc:
                danger -= 1.0
        return max(0.0, danger)

    def choose_discard(self):
        hand = normalize_tiles(self.state.hand)
        if not hand:
            return ''
        goal = select_goal(hand, self.state.packs)
        return self._choose_discard(hand, self.state.packs, goal)


class NeuralPolicy:
    """Neural policy scaffold with strict fallback to rule-based policy."""

    def __init__(self, state, model_path=None, fallback_policy=None, adaptation=None):
        self.state = state
        self.model_path = model_path or os.getenv('TING_POLICY_MODEL_PATH') or _default_model_path()
        self.fallback_policy = fallback_policy or GoalBasedPolicy(state)
        self.codec = ActionCodec()
        self.feature_extractor = FeatureExtractor()
        self.model = None
        self.adaptation = self._build_adaptation_config(adaptation)

        if self.model_path:
            try:
                self.model = load_policy_model(self.model_path)
            except Exception:
                self.model = None

    def _build_adaptation_config(self, adaptation):
        config = {
            'risk_mode': os.getenv('TING_POLICY_RISK_MODE', 'balanced'),
            'temperature': _safe_float(os.getenv('TING_POLICY_TEMPERATURE', 1.0), 1.0),
            'uncertainty_threshold': _safe_float(os.getenv('TING_POLICY_UNCERTAINTY_THRESHOLD', 3.0), 3.0),
            'belief_weight': _safe_float(os.getenv('TING_POLICY_BELIEF_WEIGHT', 0.25), 0.25),
            'fallback_on_uncertain': os.getenv('TING_POLICY_FALLBACK_ON_UNCERTAIN', '1').strip() not in ('0', 'false', 'False'),
            'disable_belief': os.getenv('TING_POLICY_DISABLE_BELIEF', '0').strip() in ('1', 'true', 'True'),
            'search_enabled': os.getenv('TING_POLICY_ENABLE_SEARCH', '0').strip() in ('1', 'true', 'True'),
            'search_disabled': os.getenv('TING_POLICY_SEARCH_DISABLE', '0').strip() in ('1', 'true', 'True'),
            'search_budget_ms': _safe_float(os.getenv('TING_POLICY_SEARCH_BUDGET_MS', 12.0), 12.0),
            'search_top_k': _safe_float(os.getenv('TING_POLICY_SEARCH_TOP_K', 3.0), 3.0),
            'search_rollout_samples': _safe_float(os.getenv('TING_POLICY_SEARCH_SAMPLES', 8.0), 8.0),
            'search_belief_weight': _safe_float(os.getenv('TING_POLICY_SEARCH_BELIEF_WEIGHT', 0.5), 0.5),
        }
        if isinstance(adaptation, dict):
            config.update(adaptation)
        return config

    def _build_search_planner(self):
        if self.model is None:
            return None
        if self.adaptation.get('search_disabled'):
            return None
        if not self.adaptation.get('search_enabled'):
            return None
        if not hasattr(self.model, 'estimate_value_from_features'):
            return None

        return BoundedRolloutPlanner(
            model=self.model,
            top_k=_safe_int(self.adaptation.get('search_top_k', 3), 3),
            rollout_samples=_safe_int(self.adaptation.get('search_rollout_samples', 8), 8),
            budget_ms=_safe_int(self.adaptation.get('search_budget_ms', 12), 12),
            disabled=False,
            belief_weight=_safe_float(self.adaptation.get('search_belief_weight', 0.5), 0.5),
        )

    def _adaptive_temperature(self, info):
        temperature = max(0.05, _safe_float(self.adaptation.get('temperature', 1.0), 1.0))
        risk_mode = str(self.adaptation.get('risk_mode', 'balanced')).strip().lower()
        policy_entropy = _safe_float(info.get('entropy', 0.0), 0.0)
        belief_entropy = _safe_float(info.get('belief_entropy', 0.0), 0.0)
        hand_size = len(getattr(self.state, 'hand', []) or [])
        meld_count = len(getattr(self.state, 'packs', []) or [])
        late_game_factor = max(0.0, (13.0 - float(hand_size)) / 13.0)
        meld_factor = min(1.0, float(meld_count) / 4.0)
        uncertainty_factor = min(1.5, (policy_entropy + belief_entropy) / 5.0)

        if risk_mode == 'conservative':
            temperature *= 0.7
        elif risk_mode == 'aggressive':
            temperature *= 1.25

        temperature *= 1.0 + 0.35 * late_game_factor + 0.15 * meld_factor + 0.25 * uncertainty_factor
        return max(0.05, temperature)

    def _belief_weight(self, info):
        if self.adaptation.get('disable_belief'):
            return 0.0

        risk_mode = str(self.adaptation.get('risk_mode', 'balanced')).strip().lower()
        base = _safe_float(self.adaptation.get('belief_weight', 0.25), 0.25)
        belief_entropy = _safe_float(info.get('belief_entropy', 0.0), 0.0)
        if risk_mode == 'conservative':
            base += 0.15
        elif risk_mode == 'aggressive':
            base *= 0.5
        else:
            base += min(0.2, belief_entropy / 50.0)
        return max(0.0, base)

    def _should_fallback_on_uncertainty(self, info):
        if not self.adaptation.get('fallback_on_uncertain', True):
            return False

        threshold = _safe_float(self.adaptation.get('uncertainty_threshold', 3.0), 3.0)
        risk_mode = str(self.adaptation.get('risk_mode', 'balanced')).strip().lower()
        uncertainty = max(_safe_float(info.get('entropy', 0.0), 0.0), _safe_float(info.get('belief_entropy', 0.0), 0.0))

        if risk_mode == 'aggressive':
            threshold *= 1.5
        elif risk_mode == 'conservative':
            threshold *= 0.75

        return uncertainty >= threshold

    def choose_action(self):
        if self.model is None:
            return self.fallback_policy.choose_action()

        try:
            legal_actions = self.state.enumerate_legal_actions()
            if not legal_actions:
                return self.fallback_policy.choose_action()

            features = self.feature_extractor.extract(self.state)
            base_info = self.model.policy_info_from_features(features, legal_actions, belief_weight=0.0)
            belief_weight = self._belief_weight(base_info)
            info = self.model.policy_info_from_features(features, legal_actions, belief_weight=belief_weight)
            if self._should_fallback_on_uncertainty(info):
                return self.fallback_policy.choose_action()

            action = choose_action_from_model(
                model=self.model,
                features=features,
                legal_actions=legal_actions,
                codec=self.codec,
                belief_weight=belief_weight,
            )

            planner = self._build_search_planner()
            if planner is not None:
                planned_action = planner.plan(features, legal_actions, belief_weight=belief_weight)
                if planned_action and self.state.is_legal_action(planned_action):
                    return planned_action

            if action and self.state.is_legal_action(action):
                return action
        except Exception:
            pass

        return self.fallback_policy.choose_action()


def create_policy(state, mode='neural', model_path=None, adaptation=None):
    normalized_mode = (mode or 'neural').strip().lower()
    if normalized_mode == 'rule':
        return GoalBasedPolicy(state)
    return NeuralPolicy(state, model_path=model_path, adaptation=adaptation)
