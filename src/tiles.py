from collections import Counter
from functools import lru_cache


NUMBERED_SUITS = ('W', 'B', 'T')
HONOR_WINDS = ('F1', 'F2', 'F3', 'F4')
HONOR_DRAGONS = ('J1', 'J2', 'J3')
HONORS = HONOR_WINDS + HONOR_DRAGONS

NUMBERED_TILES = [f'{s}{n}' for s in NUMBERED_SUITS for n in range(1, 10)]
ALL_TILES = NUMBERED_TILES + list(HONORS)

TILE_TO_IDX = {t: i for i, t in enumerate(ALL_TILES)}
IDX_TO_TILE = {i: t for i, t in enumerate(ALL_TILES)}


def tile_suit(tile):
    return tile[0]


def tile_value(tile):
    if len(tile) < 2:
        raise ValueError(f'Invalid tile: {tile}')

    try:
        value = int(tile[1:])
    except ValueError as exc:
        raise ValueError(f'Invalid tile: {tile}') from exc

    suit = tile[0]
    if suit in NUMBERED_SUITS and 1 <= value <= 9:
        return value
    if suit == 'F' and 1 <= value <= 4:
        return value
    if suit == 'J' and 1 <= value <= 3:
        return value

    raise ValueError(f'Invalid tile: {tile}')


def is_honor(tile):
    return tile[0] not in NUMBERED_SUITS


def is_terminal(tile):
    return is_honor(tile) or tile_value(tile) in (1, 9)


def is_numbered(tile):
    return tile[0] in NUMBERED_SUITS


def count_tiles(tiles):
    return Counter(tiles)


def is_sequence(tiles):
    if len(tiles) != 3:
        return False

    suits = {tile_suit(t) for t in tiles}
    if len(suits) != 1 or next(iter(suits)) not in NUMBERED_SUITS:
        return False

    vals = sorted(tile_value(t) for t in tiles)
    return vals[1] == vals[0] + 1 and vals[2] == vals[0] + 2


def is_triplet(tiles):
    return len(tiles) == 3 and len(set(tiles)) == 1


def is_pair(tiles):
    return len(tiles) == 2 and len(set(tiles)) == 1


def normalize_tiles(tiles):
    return sorted(tiles, key=lambda x: TILE_TO_IDX.get(x, 99))


def _hand_to_counts(tiles):
    counts = [0] * 34
    for t in tiles:
        idx = TILE_TO_IDX.get(t, -1)
        if idx >= 0:
            counts[idx] += 1
    return counts


def shanten_standard(tiles, n_melded=0):
    """Standard 4-meld + 1-pair shanten. Returns -1 (win) to needed*2."""
    return _shanten_standard_cached(tuple(_hand_to_counts(tiles)), int(n_melded))


@lru_cache(maxsize=1 << 20)
def _shanten_standard_cached(counts_key, n_melded):
    needed = 4 - n_melded
    counts = list(counts_key)
    best = [needed * 2]

    def update(mentsu, taatsu, jantai):
        val = mentsu * 2 + min(taatsu + jantai, needed - mentsu + 1)
        sh = needed * 2 - val
        if sh < best[0]:
            best[0] = sh

    def backtrack(pos, mentsu, taatsu, jantai):
        update(mentsu, taatsu, jantai)

        while pos < 34 and counts[pos] == 0:
            pos += 1
        if pos >= 34:
            return

        orig = counts[pos]
        in_num = pos < 27
        suit_pos = pos % 9

        if orig >= 3:
            counts[pos] -= 3
            backtrack(pos, mentsu + 1, taatsu, jantai)
            counts[pos] += 3

        if in_num and suit_pos <= 6 and counts[pos + 1] >= 1 and counts[pos + 2] >= 1:
            counts[pos] -= 1
            counts[pos + 1] -= 1
            counts[pos + 2] -= 1
            backtrack(pos, mentsu + 1, taatsu, jantai)
            counts[pos] += 1
            counts[pos + 1] += 1
            counts[pos + 2] += 1

        if orig >= 2 and jantai == 0:
            counts[pos] -= 2
            backtrack(pos, mentsu, taatsu, 1)
            counts[pos] += 2

        if orig >= 2:
            counts[pos] -= 2
            backtrack(pos, mentsu, taatsu + 1, jantai)
            counts[pos] += 2

        if in_num and suit_pos <= 7 and counts[pos + 1] >= 1:
            counts[pos] -= 1
            counts[pos + 1] -= 1
            backtrack(pos, mentsu, taatsu + 1, jantai)
            counts[pos] += 1
            counts[pos + 1] += 1

        if in_num and suit_pos <= 6 and counts[pos + 2] >= 1:
            counts[pos] -= 1
            counts[pos + 2] -= 1
            backtrack(pos, mentsu, taatsu + 1, jantai)
            counts[pos] += 1
            counts[pos + 2] += 1

        counts[pos] = 0
        backtrack(pos + 1, mentsu, taatsu, jantai)
        counts[pos] = orig

    backtrack(0, 0, 0, 0)
    return best[0]


def shanten_pairs(tiles):
    """Seven-pairs shanten. Returns -1 (win) to 6."""
    counts = Counter(tiles)
    pairs = sum(1 for c in counts.values() if c >= 2)
    return 6 - min(pairs, 7)


def shanten_orphans(tiles):
    """13-orphans shanten. Returns -1 (win) to 13."""
    orphans = frozenset(
        {
            'W1',
            'W9',
            'B1',
            'B9',
            'T1',
            'T9',
            'F1',
            'F2',
            'F3',
            'F4',
            'J1',
            'J2',
            'J3',
        }
    )
    unique = len(orphans & set(tiles))
    has_pair = any(tiles.count(t) >= 2 for t in orphans)
    return 13 - unique - (1 if has_pair else 0)


def min_shanten(tiles, n_melded=0):
    """Minimum shanten across all winning hand types."""
    s = shanten_standard(tiles, n_melded)
    s = min(s, shanten_pairs(tiles))
    if n_melded == 0:
        s = min(s, shanten_orphans(tiles))
    return s


def best_discard(tiles, n_melded=0):
    """Return (tile_to_discard, resulting_shanten) that minimizes shanten."""
    best_sh = 99
    best_tile = tiles[0] if tiles else ''
    seen = set()

    for i, tile in enumerate(tiles):
        if tile in seen:
            continue
        seen.add(tile)
        reduced = tiles[:i] + tiles[i + 1 :]
        sh = min_shanten(reduced, n_melded)
        if sh < best_sh:
            best_sh = sh
            best_tile = tile

    return best_tile, best_sh


def useful_tiles(tiles, n_melded=0):
    """Tiles that, if drawn, reduce shanten.

    Cached on the hand shape, and decomposed by winning-hand type: a drawn
    tile lowers min_shanten iff it lowers the shanten of a hand type that is
    currently at the minimum. Seven-pairs and thirteen-orphans have closed
    forms; only the standard form needs per-candidate evaluation, and only
    for candidates adjacent to the hand (same tile, or within 2 in suit).
    """
    return _useful_tiles_cached(tuple(_hand_to_counts(tiles)), int(n_melded))


_ORPHAN_INDICES = tuple(
    TILE_TO_IDX[t] for t in ('W1', 'W9', 'B1', 'B9', 'T1', 'T9', 'F1', 'F2', 'F3', 'F4', 'J1', 'J2', 'J3')
)


def _standard_candidate_indices(counts):
    """Candidates that could extend a standard-form block."""
    candidates = set()
    for idx, count in enumerate(counts):
        if count <= 0:
            continue
        if idx >= 27:
            candidates.add(idx)
            continue
        suit_base = (idx // 9) * 9
        suit_pos = idx % 9
        for delta in (-2, -1, 0, 1, 2):
            neighbor = suit_pos + delta
            if 0 <= neighbor <= 8:
                candidates.add(suit_base + neighbor)
    return candidates


@lru_cache(maxsize=1 << 18)
def _useful_tiles_cached(counts_key, n_melded):
    counts = list(counts_key)
    tiles = [IDX_TO_TILE[idx] for idx, count in enumerate(counts) for _ in range(count)]

    s_standard = shanten_standard(tiles, n_melded)
    s_pairs = shanten_pairs(tiles)
    s_orphans = shanten_orphans(tiles) if n_melded == 0 else 99
    current = min(s_standard, s_pairs, s_orphans)

    result = set()

    if s_pairs == current:
        for idx, count in enumerate(counts):
            if count == 1:
                result.add(IDX_TO_TILE[idx])

    if s_orphans == current:
        has_orphan_pair = any(counts[idx] >= 2 for idx in _ORPHAN_INDICES)
        for idx in _ORPHAN_INDICES:
            if counts[idx] == 0:
                result.add(IDX_TO_TILE[idx])
            elif counts[idx] == 1 and not has_orphan_pair:
                result.add(IDX_TO_TILE[idx])

    if s_standard == current:
        for idx in _standard_candidate_indices(counts):
            candidate = IDX_TO_TILE[idx]
            if counts[idx] >= 4 or candidate in result:
                continue
            counts[idx] += 1
            if _shanten_standard_cached(tuple(counts), n_melded) < s_standard:
                result.add(candidate)
            counts[idx] -= 1

    return frozenset(result)
