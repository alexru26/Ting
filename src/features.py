from tiles import ALL_TILES, TILE_TO_IDX


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
            'hand_counts': hand_counts,
            'seen_counts': seen_counts,
            'self_discard_counts': self_discard_counts,
            'pack_counts': pack_counts,
            'opponent_discard_counts': opponent_discard_counts,
            'meta': meta,
        }

    @staticmethod
    def _count_tiles(tiles):
        counts = [0] * len(ALL_TILES)
        for tile in tiles:
            idx = TILE_TO_IDX.get(tile)
            if idx is not None:
                counts[idx] += 1
        return counts
