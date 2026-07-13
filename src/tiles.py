from collections import Counter


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
    needed = 4 - n_melded
    counts = _hand_to_counts(tiles)
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
    """Tiles that, if drawn and optimally discarded, reduce shanten."""
    current = min_shanten(tiles, n_melded)
    result = set()

    for candidate in ALL_TILES:
        extended = tiles + [candidate]
        _, new_sh = best_discard(extended, n_melded)
        if new_sh < current:
            result.add(candidate)

    return result
