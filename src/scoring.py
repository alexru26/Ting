from collections import Counter
from enum import Enum, auto

from tiles import NUMBERED_SUITS, best_discard, is_honor, min_shanten, shanten_pairs, shanten_standard

from MahjongGB import MahjongFanCalculator as _FanCalc


def fan_calculator_available():
    return True


class GoalType(Enum):
    STANDARD = auto()
    SEVEN_PAIRS = auto()
    PURE_FLUSH = auto()
    MIXED_FLUSH = auto()
    ALL_TRIPLETS = auto()


_GOAL_BASE_FAN = {
    GoalType.STANDARD: 8,
    GoalType.SEVEN_PAIRS: 24,
    GoalType.PURE_FLUSH: 24,
    GoalType.MIXED_FLUSH: 6,
    GoalType.ALL_TRIPLETS: 6,
}


def calculate_fan(
    packs,
    hand,
    win_tile,
    flower_count,
    is_self_drawn,
    is_4th_tile,
    is_about_kong,
    is_wall_last,
    seat_wind,
    prevalent_wind,
):
    """Return total fan count using MahjongGB calculator."""
    result = _FanCalc(
        packs,
        hand,
        win_tile,
        flower_count,
        is_self_drawn,
        is_4th_tile,
        is_about_kong,
        is_wall_last,
        seat_wind,
        prevalent_wind,
    )
    return sum(fan_val for fan_val, _ in result)


def can_win(
    hand,
    packs,
    win_tile,
    flower_count,
    is_self_drawn,
    seat_wind,
    prevalent_wind,
    is_about_kong=False,
):
    """Return True if the hand is a valid winning hand (>=8 fan)."""
    try:
        fan = calculate_fan(
            packs,
            tuple(hand),
            win_tile,
            flower_count,
            is_self_drawn,
            False,
            is_about_kong,
            False,
            seat_wind,
            prevalent_wind,
        )
        return fan >= 8
    except Exception:
        return False


def _tiles_for_pure_flush(hand):
    best_suit = ''
    best_count = -1
    for suit in NUMBERED_SUITS:
        cnt = sum(1 for t in hand if t[0] == suit)
        if cnt > best_count:
            best_count = cnt
            best_suit = suit

    return best_suit, [t for t in hand if t[0] == best_suit]


def _tiles_for_mixed_flush(hand):
    best_suit = ''
    best_count = -1
    for suit in NUMBERED_SUITS:
        cnt = sum(1 for t in hand if t[0] == suit)
        if cnt > best_count:
            best_count = cnt
            best_suit = suit

    filtered = [t for t in hand if t[0] == best_suit or is_honor(t)]
    return best_suit, filtered


def _shanten_all_triplets(hand, n_melded):
    needed = 4 - n_melded
    counts = Counter(hand)
    triplets = sum(1 for c in counts.values() if c >= 3)
    pairs = sum(1 for c in counts.values() if c >= 2)

    triplets = min(triplets, needed)
    remaining_pairs = max(0, pairs - triplets)
    jantai = min(1, remaining_pairs)
    return needed * 2 - triplets * 2 - jantai - 1


def _goal_shanten(hand, packs, goal):
    n = len(packs)

    if goal == GoalType.STANDARD:
        return min_shanten(hand, n)

    if goal == GoalType.SEVEN_PAIRS:
        if n > 0:
            return 99
        return shanten_pairs(hand)

    if goal == GoalType.PURE_FLUSH:
        _, filtered = _tiles_for_pure_flush(hand)
        off_count = len(hand) - len(filtered)
        sh = shanten_standard(filtered, n)
        return sh + off_count

    if goal == GoalType.MIXED_FLUSH:
        _, filtered = _tiles_for_mixed_flush(hand)
        off_count = len(hand) - len(filtered)
        sh = shanten_standard(filtered, n)
        return sh + off_count

    if goal == GoalType.ALL_TRIPLETS:
        return _shanten_all_triplets(hand, n)

    return 99


def goal_utility(hand, packs, goal):
    sh = _goal_shanten(hand, packs, goal)
    if sh < 0:
        return float('inf')

    base = _GOAL_BASE_FAN[goal]
    return base / (sh + 1) ** 1.5


def select_goal(hand, packs):
    best_goal = GoalType.STANDARD
    best_util = -1.0

    for goal in GoalType:
        u = goal_utility(hand, packs, goal)
        if u > best_util:
            best_util = u
            best_goal = goal

    return best_goal


def best_discard_for_goal(hand, packs, goal):
    n = len(packs)

    if goal == GoalType.SEVEN_PAIRS:
        counts = Counter(hand)
        isolated = [t for t in hand if counts[t] == 1]
        if isolated:
            return isolated[0]
        return hand[0]

    if goal == GoalType.PURE_FLUSH:
        dominant_suit, _ = _tiles_for_pure_flush(hand)
        off_suit = [t for t in hand if t[0] != dominant_suit]
        if off_suit:
            for t in hand:
                if t[0] != dominant_suit:
                    return t
        return best_discard(hand, n)[0]

    if goal == GoalType.MIXED_FLUSH:
        dominant_suit, _ = _tiles_for_mixed_flush(hand)
        off_suit = [t for t in hand if t[0] != dominant_suit and not is_honor(t)]
        if off_suit:
            return off_suit[0]
        return best_discard(hand, n)[0]

    if goal == GoalType.ALL_TRIPLETS:
        counts = Counter(hand)
        singles = [t for t in hand if counts[t] == 1]
        if singles:
            return singles[0]
        return best_discard(hand, n)[0]

    return best_discard(hand, n)[0]
