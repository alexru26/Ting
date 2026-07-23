from collections import Counter

from tiles import ALL_TILES, NUMBERED_SUITS, normalize_tiles, tile_value
from scoring import calculate_fan

MIN_WIN_FAN = 8


class GameState:
    def __init__(self):
        self.my_id = 0
        self.quan = 0
        self.hand = []
        self.packs = []
        self.discards = []
        self.flowers = 0

        self.opponent_discards = {}
        self.opponent_packs = {}
        self.seen_tiles = Counter()

        self.last_request_type = -1
        self.last_request_action = None
        self.last_tile = None
        self.last_actor = None

        self._last_discard = None
        self._last_discard_player = None
        self._last_drawer = None
        self._pending_gang_tile = None
        self._pending_bugang_tile = None

    @classmethod
    def from_history(cls, requests, responses):
        state = cls()
        state.parse_from_history(requests, responses)
        return state

    def parse_from_history(self, requests, responses):
        if not requests:
            return

        if not self._parse_init(requests[0]):
            return
        if len(requests) == 1:
            self.last_request_type = 0
            return

        if not self._parse_deal(requests[1]):
            return
        if len(requests) == 2:
            self.last_request_type = 1
            return

        for i in range(2, len(requests) - 1):
            resp = responses[i] if i < len(responses) else None
            self._apply_request(requests[i], resp)

        self._set_current_request(requests[-1])

    @staticmethod
    def _parse_int(value):
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _is_valid_tile(tile):
        return tile in ALL_TILES

    def _parse_init(self, req):
        parts = req.split()
        if len(parts) < 3:
            return False

        req_type = self._parse_int(parts[0])
        my_id = self._parse_int(parts[1])
        quan = self._parse_int(parts[2])
        if req_type != 0 or my_id is None or quan is None or my_id < 0 or my_id > 3:
            return False

        self.my_id = my_id
        self.quan = quan
        for pid in range(4):
            if pid != self.my_id:
                self.opponent_discards[pid] = []
                self.opponent_packs[pid] = []
        return True

    def _parse_deal(self, req):
        parts = req.split()
        flower_index = self.my_id + 1
        if len(parts) <= flower_index:
            return False

        req_type = self._parse_int(parts[0])
        flower_count = self._parse_int(parts[flower_index])
        if req_type != 1 or flower_count is None:
            return False

        self.flowers = flower_count
        self.hand = [t for t in parts[5:18] if not t.startswith('H')]
        return True

    def _apply_request(self, req, response):
        parts = req.split()
        if not parts:
            return

        req_type = self._parse_int(parts[0])
        if req_type is None:
            return

        if req_type == 2:
            if len(parts) < 2:
                return
            tile = parts[1]
            if tile.startswith('H'):
                self.flowers += 1
                return
            if not self._is_valid_tile(tile):
                return
            self.hand.append(tile)
            if response:
                resp = response.split()
                if resp[0] == 'GANG' and len(resp) > 1:
                    self._pending_gang_tile = resp[1]
                elif resp[0] == 'BUGANG' and len(resp) > 1:
                    self._pending_bugang_tile = resp[1]

        elif req_type == 3:
            self._apply_type3(parts, response)

    def _apply_type3(self, parts, _response):
        if len(parts) < 3:
            return

        actor = self._parse_int(parts[1])
        if actor is None:
            return
        action = parts[2]

        if action == 'DRAW':
            self._last_drawer = actor
            return
        if action == 'BUHUA':
            return

        if action == 'PLAY':
            if len(parts) < 4:
                return
            tile = parts[3]
            if not self._is_valid_tile(tile):
                return
            self._last_discard = tile
            self._last_discard_player = actor
            if actor == self.my_id:
                self._discard_from_hand(tile)
            else:
                self.opponent_discards.setdefault(actor, []).append(tile)
                self.seen_tiles[tile] += 1
            return

        if action == 'PENG':
            disc = parts[3] if len(parts) > 3 else None
            if disc is not None and not self._is_valid_tile(disc):
                disc = None
            penged = self._last_discard
            giver = self._last_discard_player
            if actor == self.my_id and penged:
                for _ in range(2):
                    if penged in self.hand:
                        self.hand.remove(penged)
                self.packs.append(('PENG', penged, giver))
                self.seen_tiles[penged] += 2
                if disc:
                    self._discard_from_hand(disc)
            elif penged:
                self.opponent_packs.setdefault(actor, []).append(('PENG', penged, giver))
                self.seen_tiles[penged] += 2
                if disc:
                    self.opponent_discards.setdefault(actor, []).append(disc)
                    self.seen_tiles[disc] += 1
            if disc:
                self._last_discard = disc
                self._last_discard_player = actor
            return

        if action == 'CHI':
            mid = parts[3] if len(parts) > 3 else None
            disc = parts[4] if len(parts) > 4 else None
            claimed = self._last_discard
            if not mid or not self._is_valid_tile(mid):
                return
            try:
                mid_val = tile_value(mid)
            except ValueError:
                return
            suit = mid[0]
            if suit not in NUMBERED_SUITS:
                return
            seq = ['%s%d' % (suit, mid_val - 1), mid, '%s%d' % (suit, mid_val + 1)]
            if any((not self._is_valid_tile(t) for t in seq)):
                return
            # The claimed discard was already counted when it was played; only
            # the two tiles contributed from the actor's hand become newly seen.
            claimed_seen = claimed if claimed in seq else None
            for t in seq:
                if t == claimed_seen:
                    claimed_seen = None
                    continue
                self.seen_tiles[t] += 1
            if actor == self.my_id and claimed:
                offer = seq.index(claimed) if claimed in seq else 0
                for t in [x for x in seq if x != claimed]:
                    if t in self.hand:
                        self.hand.remove(t)
                self.packs.append(('CHI', mid, offer))
                if disc:
                    self._discard_from_hand(disc)
            else:
                self.opponent_packs.setdefault(actor, []).append(('CHI', mid, 0))
                if disc and self._is_valid_tile(disc):
                    self.opponent_discards.setdefault(actor, []).append(disc)
                    self.seen_tiles[disc] += 1
            if disc and self._is_valid_tile(disc):
                self._last_discard = disc
                self._last_discard_player = actor
            return

        if action == 'GANG':
            if actor == self.my_id:
                if self._pending_gang_tile:
                    gtile = self._pending_gang_tile
                    for _ in range(4):
                        if gtile in self.hand:
                            self.hand.remove(gtile)
                    self.packs.append(('GANG', gtile, self.my_id))
                    self.seen_tiles[gtile] += 4
                    self._pending_gang_tile = None
                elif self._last_discard and self._last_discard_player != self.my_id:
                    gtile = self._last_discard
                    for _ in range(3):
                        if gtile in self.hand:
                            self.hand.remove(gtile)
                    self.packs.append(('GANG', gtile, self._last_discard_player))
                    self.seen_tiles[gtile] += 3
            elif self._last_drawer == actor:
                # Concealed kong right after the actor's own draw: the four
                # tiles stay hidden, so no seen-tile update is possible.
                self.opponent_packs.setdefault(actor, []).append(('GANG', None, actor))
            else:
                gtile = self._last_discard
                if gtile:
                    self.opponent_packs.setdefault(actor, []).append(('GANG', gtile, self._last_discard_player))
                    # The claimed discard is already in seen_tiles; the three
                    # tiles from the actor's hand are newly revealed.
                    self.seen_tiles[gtile] += 3
            return

        if action == 'BUGANG':
            tile = parts[3] if len(parts) > 3 else None
            if tile and self._is_valid_tile(tile):
                self.seen_tiles[tile] += 1
                if actor == self.my_id:
                    if tile in self.hand:
                        self.hand.remove(tile)
                    for i, (ptype, ptile, poffer) in enumerate(self.packs):
                        if ptype == 'PENG' and ptile == tile:
                            self.packs[i] = ('GANG', tile, poffer)
                            break
                    self._pending_bugang_tile = None
                else:
                    opp = self.opponent_packs.setdefault(actor, [])
                    for i, (ptype, ptile, poffer) in enumerate(opp):
                        if ptype == 'PENG' and ptile == tile:
                            opp[i] = ('GANG', tile, poffer)
                            break

    def _discard_from_hand(self, tile):
        if tile in self.hand:
            self.hand.remove(tile)
        self.discards.append(tile)
        self.seen_tiles[tile] += 1

    def _set_current_request(self, req):
        parts = req.split()
        if not parts:
            return

        req_type = self._parse_int(parts[0])
        if req_type is None:
            return
        self.last_request_type = req_type
        self.last_request_action = None
        self.last_tile = None
        self.last_actor = None

        if req_type == 2:
            if len(parts) < 2:
                self.last_request_type = -1
                return
            tile = parts[1]
            self.last_tile = tile
            if not tile.startswith('H'):
                if self._is_valid_tile(tile):
                    self.hand.append(tile)
            else:
                self.flowers += 1

        elif req_type == 3:
            if len(parts) < 3:
                self.last_request_type = -1
                return
            actor = self._parse_int(parts[1])
            if actor is None:
                self.last_request_type = -1
                return
            self.last_actor = actor
            self.last_request_action = parts[2]
            self.last_tile = parts[3] if len(parts) > 3 else None
            self._apply_type3(parts, None)

    @property
    def n_melded(self):
        return len(self.packs)

    def fan_calc_packs(self):
        result = []
        for ptype, ptile, poffer in self.packs:
            if ptype in ('PENG', 'GANG'):
                relative = (poffer - self.my_id + 4) % 4
                result.append((ptype, ptile, relative))
            else:
                result.append(('CHI', ptile, poffer))
        return tuple(result)

    def to_summary(self):
        return {
            'my_id': self.my_id,
            'quan': self.quan,
            'hand': self.hand,
            'packs': self.packs,
            'discards': self.discards,
            'flowers': self.flowers,
            'seen_tiles': dict(self.seen_tiles),
        }

    def add_initial_hand(self, tiles):
        self.hand = list(tiles)

    def draw_tile(self, tile):
        self.hand.append(tile)

    def play_tile(self, tile):
        if tile in self.hand:
            self.hand.remove(tile)
        self.discards.append(tile)

    def add_pack(self, pack_type, tile, offer):
        self.packs.append((pack_type, tile, offer))

    def current_win_fan(self):
        """Fan value if HU is declared on the current event, else 0.

        Covers self-drawn wins (type 2), wins on another player's discard
        (type 3 PLAY), and robbing a supplement kong (type 3 BUGANG).
        """
        request_type = self.last_request_type
        request_action = self.last_request_action
        tile = self.last_tile
        if not tile or tile not in ALL_TILES:
            return 0

        is_self_drawn = False
        is_about_kong = False
        hand = list(self.hand)
        if request_type == 2:
            is_self_drawn = True
            if tile in hand:
                hand.remove(tile)
        elif request_type == 3 and self.last_actor != self.my_id and request_action == 'PLAY':
            pass
        elif request_type == 3 and self.last_actor != self.my_id and request_action == 'BUGANG':
            is_about_kong = True
        else:
            return 0

        if len(hand) != max(1, 13 - 3 * len(self.packs)):
            return 0

        try:
            return int(
                calculate_fan(
                    self.fan_calc_packs(),
                    tuple(hand),
                    tile,
                    int(self.flowers),
                    is_self_drawn,
                    False,
                    is_about_kong,
                    False,
                    self.my_id % 4,
                    self.quan,
                )
            )
        except Exception:
            # MahjongGB raises when the shape is not a winning hand.
            return 0

    def can_hu(self):
        return self.current_win_fan() >= MIN_WIN_FAN

    def enumerate_legal_actions(self):
        """All actions Botzone will accept for the current request.

        Every action returned here must be valid on the judge side; the
        policy chooses only from this list, so soundness here is what
        guarantees the bot never emits an INVALID move.
        """
        actions = []
        request_type = self.last_request_type
        request_action = self.last_request_action

        if request_type == 2:
            # After a draw the bot must act: PASS is not a legal response.
            if self.can_hu():
                actions.append('HU')
            unique_hand = normalize_tiles(list(set(self.hand)))
            for tile in unique_hand:
                actions.append('PLAY %s' % tile)
            for tile in unique_hand:
                if self.hand.count(tile) >= 4:
                    actions.append('GANG %s' % tile)
            peng_tiles = {tile for ptype, tile, _ in self.packs if ptype == 'PENG'}
            for tile in unique_hand:
                if tile in peng_tiles:
                    actions.append('BUGANG %s' % tile)

        elif request_type == 3:
            if request_action == 'PLAY' and self.last_actor != self.my_id:
                actions.append('PASS')
                if self.can_hu():
                    actions.append('HU')
                last_tile = self.last_tile or ''
                if self.hand.count(last_tile) >= 3:
                    actions.append('GANG')
                if self.hand.count(last_tile) >= 2:
                    for discard in self._discards_after_removal([last_tile, last_tile]):
                        actions.append('PENG %s' % discard)
                if self._chi_allowed():
                    for mid_tile in self._valid_chi_mids(last_tile):
                        used = self._chi_used_tiles(mid_tile, last_tile)
                        for discard in self._discards_after_removal(used):
                            actions.append('CHI %s %s' % (mid_tile, discard))
            elif request_action == 'BUGANG' and self.last_actor != self.my_id:
                actions.append('PASS')
                if self.can_hu():
                    actions.append('HU')
            else:
                actions.append('PASS')
        else:
            actions.append('PASS')

        return self._dedupe_actions(actions)

    def _discards_after_removal(self, used_tiles):
        """Distinct tiles still discardable after `used_tiles` leave the hand."""
        remaining = Counter(self.hand)
        for tile in used_tiles:
            remaining[tile] -= 1
        if any(count < 0 for count in remaining.values()):
            return []
        return normalize_tiles([tile for tile, count in remaining.items() if count > 0])

    @staticmethod
    def _chi_used_tiles(mid_tile, claimed_tile):
        suit = mid_tile[0]
        mid_val = int(mid_tile[1:])
        seq = ['%s%d' % (suit, mid_val - 1), mid_tile, '%s%d' % (suit, mid_val + 1)]
        return [tile for tile in seq if tile != claimed_tile]

    def is_legal_action(self, action):
        return action in set(self.enumerate_legal_actions())

    def _chi_allowed(self):
        if not self.last_tile or self.last_tile[0] not in NUMBERED_SUITS:
            return False
        if self.last_actor is None:
            return False
        return (self.last_actor + 1) % 4 == self.my_id

    def _valid_chi_mids(self, tile):
        if not tile or tile[0] not in NUMBERED_SUITS:
            return []
        suit = tile[0]
        try:
            val = int(tile[1:])
        except ValueError:
            return []

        mids = []
        for mid_val in range(max(2, val - 1), min(8, val + 1) + 1):
            seq = ['%s%d' % (suit, mid_val - 1), '%s%d' % (suit, mid_val), '%s%d' % (suit, mid_val + 1)]
            if tile not in seq:
                continue
            needed = [t for t in seq if t != tile]
            if all(t in self.hand for t in needed):
                mids.append('%s%d' % (suit, mid_val))
        return mids

    @staticmethod
    def _dedupe_actions(actions):
        seen = set()
        result = []
        for action in actions:
            if action in seen:
                continue
            seen.add(action)
            result.append(action)
        return result
