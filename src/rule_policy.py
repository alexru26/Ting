"""Rule-based goal policy.

Training-only module: it serves as the dataset-generation teacher and as a
baseline opponent for self-play and evaluation. The Botzone runtime path
(bot.py / policy.py) never imports it - the neural model is the only
runtime decision source.
"""

from collections import Counter

from tiles import normalize_tiles, tile_value, min_shanten, NUMBERED_SUITS
from scoring import select_goal, best_discard_for_goal, can_win


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
            if action == 'PLAY' and s.last_actor != s.my_id:
                return self._on_play(s.last_tile, s.last_actor)
            if action == 'BUGANG' and s.last_actor != s.my_id:
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
        if bugang_tile:
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
        if not hand:
            return ''

        candidates = list(set(hand))
        n = len(packs)
        if not candidates:
            return ''

        def score(tile):
            reduced = hand[:]
            reduced.remove(tile)
            sh = min_shanten(reduced, n)
            offence = -sh
            danger = self._danger_score(tile)
            return offence - self.DEFENSE_WEIGHT * danger
        goal_tile = best_discard_for_goal(hand, packs, goal)
        if goal_tile not in hand:
            goal_tile = candidates[0]
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

    def _should_gang_open(self, tile, hand, packs):
        """Open kong from a discard: only do it if our hand profits."""
        reduced = [t for t in hand if t != tile]
        sh_before = min_shanten(hand, len(packs))
        sh_after = min_shanten(reduced, len(packs) + 1)
        return sh_after <= sh_before

    def _peng_and_discard(self, tile, hand, packs):
        """
        Decide whether to peng and which tile to discard afterwards.
        Returns the discard tile, or None to decline.
        """
        n = len(packs)
        if hand.count(tile) < 2:
            return None

        sh_without = min_shanten(hand, n)
        new_hand = hand[:]
        new_hand.remove(tile)
        new_hand.remove(tile)
        if not new_hand:
            return None
        goal = select_goal(new_hand, packs + [('PENG', tile, 0)])
        discard = self._choose_discard(new_hand, packs + [('PENG', tile, 0)], goal)
        if discard not in new_hand:
            return None
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
        for mid_val in range(max(2, val - 1), min(8, val + 1) + 1):
            seq = [f'{suit}{mid_val - 1}', f'{suit}{mid_val}', f'{suit}{mid_val + 1}']
            from_hand = [t for t in seq if t != tile]
            if all((t in hand for t in from_hand)):
                mid = f'{suit}{mid_val}'
                new_hand = hand[:]
                for t in from_hand:
                    new_hand.remove(t)
                if not new_hand:
                    continue
                goal = select_goal(new_hand, packs + [('CHI', mid, 0)])
                discard = self._choose_discard(new_hand, packs + [('CHI', mid, 0)], goal)
                if discard not in new_hand:
                    continue
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
                if ptile and ptile[0] == tile_suit_char:
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
