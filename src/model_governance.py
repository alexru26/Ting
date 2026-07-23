"""Promotion gating utilities: paired duplicate-wall evaluation and SPRT."""

import math


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


def duplicate_wall_evaluation(seeds, game_factory, candidate_policy_factory, baseline_policy_factory, quan=0, candidate_seat=0):
    """Run candidate and baseline over identical walls and compare outcomes."""
    paired_results = []

    for seed in list(seeds or []):
        candidate_game = game_factory(quan=quan, seed=seed, policy_factory=candidate_policy_factory)
        baseline_game = game_factory(quan=quan, seed=seed, policy_factory=baseline_policy_factory)

        candidate_result = candidate_game.run()
        baseline_result = baseline_game.run()

        paired_results.append(
            {
                'seed': _safe_int(seed, 0),
                'candidate': candidate_result,
                'baseline': baseline_result,
                'score_delta': _paired_score_delta(candidate_result, baseline_result, candidate_seat),
                'win_delta': _paired_win_delta(candidate_result, baseline_result, candidate_seat=candidate_seat),
            }
        )

    summary = {
        'games': len(paired_results),
        'candidate_wins': sum(1 for row in paired_results if row['candidate'].get('winner') == candidate_seat),
        'baseline_wins': sum(1 for row in paired_results if row['baseline'].get('winner') == candidate_seat),
        'draws': sum(1 for row in paired_results if row['candidate'].get('winner') is None and row['baseline'].get('winner') is None),
        'avg_score_delta': 0.0,
        'avg_win_delta': 0.0,
        'paired_results': paired_results,
    }

    if paired_results:
        summary['avg_score_delta'] = sum(row['score_delta'] for row in paired_results) / float(len(paired_results))
        summary['avg_win_delta'] = sum(row['win_delta'] for row in paired_results) / float(len(paired_results))

    return summary


def _paired_score_delta(candidate_result, baseline_result, candidate_seat=0):
    candidate_scores = list(candidate_result.get('scores', [0, 0, 0, 0]))
    baseline_scores = list(baseline_result.get('scores', [0, 0, 0, 0]))
    return _safe_float(candidate_scores[candidate_seat], 0.0) - _safe_float(baseline_scores[candidate_seat], 0.0)


def _paired_win_delta(candidate_result, baseline_result, candidate_seat=0):
    candidate_score = 1.0 if candidate_result.get('winner') == candidate_seat else 0.0
    baseline_score = 1.0 if baseline_result.get('winner') == candidate_seat else 0.0
    return candidate_score - baseline_score


def sprt_promotion_gate(paired_summary, min_win_rate=0.55, alpha=0.05, beta=0.2, min_avg_score_delta=0.0):
    games = _safe_int(paired_summary.get('games', 0), 0)
    if games <= 0:
        return {'decision': 'reject', 'reason': 'no_games', 'llr': float('-inf')}

    candidate_wins = _safe_int(paired_summary.get('candidate_wins', 0), 0)
    baseline_wins = _safe_int(paired_summary.get('baseline_wins', 0), 0)
    draws = _safe_int(paired_summary.get('draws', 0), 0)
    candidate_win_rate = paired_summary.get('candidate_win_rate', None)
    if candidate_win_rate is not None and candidate_wins == 0 and baseline_wins == 0:
        win_rate = _safe_float(candidate_win_rate, 0.0)
        avg_score_delta = _safe_float(paired_summary.get('avg_score_delta', 0.0), 0.0)
        if win_rate >= float(min_win_rate) and avg_score_delta >= float(min_avg_score_delta):
            return {'decision': 'accept', 'reason': 'summary_metrics', 'win_rate': win_rate, 'avg_score_delta': avg_score_delta, 'llr': 0.0, 'upper': None, 'lower': None}
        if win_rate < float(min_win_rate) or avg_score_delta < float(min_avg_score_delta):
            return {'decision': 'reject', 'reason': 'summary_metrics', 'win_rate': win_rate, 'avg_score_delta': avg_score_delta, 'llr': float('-inf'), 'upper': None, 'lower': None}
        return {'decision': 'continue', 'reason': 'summary_metrics', 'win_rate': win_rate, 'avg_score_delta': avg_score_delta, 'llr': 0.0, 'upper': None, 'lower': None}
    else:
        decisive_games = max(1, candidate_wins + baseline_wins)
        win_rate = candidate_wins / float(decisive_games)
    avg_score_delta = _safe_float(paired_summary.get('avg_score_delta', 0.0), 0.0)

    if avg_score_delta < float(min_avg_score_delta):
        return {'decision': 'reject', 'reason': 'score_delta', 'win_rate': win_rate, 'avg_score_delta': avg_score_delta, 'llr': float('-inf')}

    p0 = 0.5
    p1 = max(float(min_win_rate) + 0.15, 0.7)
    llr = candidate_wins * math.log(p1 / p0)
    llr += baseline_wins * math.log((1.0 - p1) / (1.0 - p0))
    llr += draws * math.log(0.5)

    upper = math.log((1.0 - float(beta)) / float(alpha))
    lower = math.log(float(beta) / (1.0 - float(alpha)))

    if llr >= upper and win_rate >= float(min_win_rate):
        return {'decision': 'accept', 'reason': 'llr', 'win_rate': win_rate, 'avg_score_delta': avg_score_delta, 'llr': llr, 'upper': upper, 'lower': lower}
    if llr <= lower:
        return {'decision': 'reject', 'reason': 'llr', 'win_rate': win_rate, 'avg_score_delta': avg_score_delta, 'llr': llr, 'upper': upper, 'lower': lower}
    return {'decision': 'continue', 'reason': 'insufficient_evidence', 'win_rate': win_rate, 'avg_score_delta': avg_score_delta, 'llr': llr, 'upper': upper, 'lower': lower}
