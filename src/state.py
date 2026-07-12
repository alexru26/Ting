from collections import Counter

from tiles import NUMBERED_SUITS, normalize_tiles


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

        self._parse_init(requests[0])
        if len(requests) == 1:
            self.last_request_type = 0
            return

        self._parse_deal(requests[1])
        if len(requests) == 2:
            self.last_request_type = 1
            return

        for i in range(2, len(requests) - 1):
            resp = responses[i] if i < len(responses) else None
            self._apply_request(requests[i], resp)

        self._set_current_request(requests[-1])

    def _parse_init(self, req):
        parts = req.split()
        self.my_id = int(parts[1])
        self.quan = int(parts[2])
        for pid in range(4):
            if pid != self.my_id:
                self.opponent_discards[pid] = []
                self.opponent_packs[pid] = []

    def _parse_deal(self, req):
        parts = req.split()
        self.flowers = int(parts[self.my_id + 1])
        self.hand = [t for t in parts[5:18] if not t.startswith('H')]

    def _apply_request(self, req, response):
        parts = req.split()
        req_type = int(parts[0])

        if req_type == 2:
            tile = parts[1]
            if tile.startswith('H'):
                self.flowers += 1
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
        actor = int(parts[1])
        action = parts[2]

        if action == 'DRAW' or action == 'BUHUA':
            return

        if action == 'PLAY':
            tile = parts[3]
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
            if mid:
                mid_val = int(mid[1:])
                suit = mid[0]
                seq = ['%s%d' % (suit, mid_val - 1), mid, '%s%d' % (suit, mid_val + 1)]
                for t in seq:
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
                    if disc:
                        self.opponent_discards.setdefault(actor, []).append(disc)
                        self.seen_tiles[disc] += 1
            if disc:
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
            else:
                gtile = self._last_discard
                if gtile:
                    self.opponent_packs.setdefault(actor, []).append(('GANG', gtile, self._last_discard_player))
                    self.seen_tiles[gtile] += 4
            return

        if action == 'BUGANG':
            tile = parts[3] if len(parts) > 3 else None
            if tile:
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
        req_type = int(parts[0])
        self.last_request_type = req_type
        self.last_request_action = None
        self.last_tile = None
        self.last_actor = None

        if req_type == 2:
            tile = parts[1]
            self.last_tile = tile
            if not tile.startswith('H'):
                self.hand.append(tile)
            else:
                self.flowers += 1

        elif req_type == 3:
            self.last_actor = int(parts[1])
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

    def enumerate_legal_actions(self):
        actions = []
        request_type = self.last_request_type
        request_action = self.last_request_action

        if request_type == 2:
            actions.extend(['PASS', 'HU'])
            for tile in normalize_tiles(list(set(self.hand))):
                actions.append('PLAY %s' % tile)

            for tile in normalize_tiles(list(set(self.hand))):
                if self.hand.count(tile) >= 4:
                    actions.append('GANG %s' % tile)

            bugang_tiles = {tile for ptype, tile, _ in self.packs if ptype == 'PENG'}
            for tile in normalize_tiles(list(bugang_tiles)):
                actions.append('BUGANG %s' % tile)

        elif request_type == 3:
            if request_action == 'PLAY':
                actions.extend(['PASS', 'HU'])
                last_tile = self.last_tile or ''
                if self.hand.count(last_tile) >= 3:
                    actions.append('GANG')

                if self.hand.count(last_tile) >= 2:
                    for discard in normalize_tiles(list(set(self.hand))):
                        actions.append('PENG %s' % discard)

                if self._chi_allowed():
                    for mid_tile in self._valid_chi_mids(last_tile):
                        for discard in normalize_tiles(list(set(self.hand))):
                            actions.append('CHI %s %s' % (mid_tile, discard))
            elif request_action in ('GANG', 'BUGANG', 'PENG', 'CHI', 'DRAW', 'BUHUA'):
                actions.append('PASS')
        else:
            actions.append('PASS')

        return self._dedupe_actions(actions)

    def is_legal_action(self, action):
        return action in set(self.enumerate_legal_actions())

    def legal_action_mask(self, codec):
        legal = set(self.enumerate_legal_actions())
        action_list = codec.all_actions()
        return [1 if action in legal else 0 for action in action_list]

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
