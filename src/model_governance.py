import hashlib
import json
import math
import os


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


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        while True:
            chunk = handle.read(65536)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def build_model_registry_entry(model_path, version, training_corpus, metrics=None, extra_metadata=None):
    entry = {
        'model_path': model_path,
        'version': str(version),
        'checksum_sha256': sha256_file(model_path) if os.path.exists(model_path) else None,
        'training_corpus': list(training_corpus or []),
        'metrics': dict(metrics or {}),
        'extra_metadata': dict(extra_metadata or {}),
    }
    return entry


def write_model_registry(output_path, entries):
    parent = os.path.dirname(output_path)
    if parent and not os.path.exists(parent):
        os.makedirs(parent)

    payload = {
        'count': len(entries),
        'entries': list(entries),
    }
    with open(output_path, 'w', encoding='utf-8') as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2, sort_keys=True)
    return payload


def duplicate_wall_evaluation(seeds, game_factory, candidate_policy_factory, baseline_policy_factory, quan=0, candidate_seat=0):
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
                'score_delta': _paired_score_delta(candidate_result, baseline_result),
                'win_delta': _paired_win_delta(candidate_result, baseline_result, candidate_seat=candidate_seat),
            }
        )

    summary = {
        'games': len(paired_results),
        'candidate_wins': sum(1 for row in paired_results if row['candidate'].get('winner') is not None),
        'baseline_wins': sum(1 for row in paired_results if row['baseline'].get('winner') is not None),
        'draws': sum(1 for row in paired_results if row['candidate'].get('winner') is None and row['baseline'].get('winner') is None),
        'avg_score_delta': 0.0,
        'avg_win_delta': 0.0,
        'paired_results': paired_results,
    }

    if paired_results:
        summary['avg_score_delta'] = sum(row['score_delta'] for row in paired_results) / float(len(paired_results))
        summary['avg_win_delta'] = sum(row['win_delta'] for row in paired_results) / float(len(paired_results))

    return summary


def _paired_score_delta(candidate_result, baseline_result):
    candidate_scores = list(candidate_result.get('scores', [0, 0, 0, 0]))
    baseline_scores = list(baseline_result.get('scores', [0, 0, 0, 0]))
    candidate_margin = max(_safe_float(score, 0.0) for score in candidate_scores) - min(_safe_float(score, 0.0) for score in candidate_scores)
    baseline_margin = max(_safe_float(score, 0.0) for score in baseline_scores) - min(_safe_float(score, 0.0) for score in baseline_scores)
    return candidate_margin - baseline_margin


def _paired_win_delta(candidate_result, baseline_result, candidate_seat=0):
    candidate_winner = candidate_result.get('winner')
    baseline_winner = baseline_result.get('winner')
    candidate_score = 1.0 if candidate_winner == candidate_seat else 0.0
    baseline_score = 1.0 if baseline_winner == candidate_seat else 0.0
    return candidate_score - baseline_score


def elo_expected_score(rating_a, rating_b):
    return 1.0 / (1.0 + 10.0 ** ((_safe_float(rating_b, 0.0) - _safe_float(rating_a, 0.0)) / 400.0))


def elo_update(rating_a, rating_b, score_a, k_factor=16.0):
    expected_a = elo_expected_score(rating_a, rating_b)
    expected_b = 1.0 - expected_a
    score_a = min(1.0, max(0.0, _safe_float(score_a, 0.0)))
    score_b = 1.0 - score_a
    updated_a = _safe_float(rating_a, 0.0) + float(k_factor) * (score_a - expected_a)
    updated_b = _safe_float(rating_b, 0.0) + float(k_factor) * (score_b - expected_b)
    return updated_a, updated_b


def update_elo_ladder(ratings, match_results, k_factor=16.0, default_rating=1500.0):
    ladder = {str(key): _safe_float(value, default_rating) for key, value in dict(ratings or {}).items()}
    for match in list(match_results or []):
        player_a = str(match.get('player_a'))
        player_b = str(match.get('player_b'))
        if not player_a or not player_b or player_a == 'None' or player_b == 'None':
            continue
        rating_a = ladder.get(player_a, float(default_rating))
        rating_b = ladder.get(player_b, float(default_rating))
        score_a = _safe_float(match.get('score_a', 0.5), 0.5)
        updated_a, updated_b = elo_update(rating_a, rating_b, score_a, k_factor=k_factor)
        ladder[player_a] = updated_a
        ladder[player_b] = updated_b

    ordered = sorted(ladder.items(), key=lambda item: (-item[1], item[0]))
    return [{'player': player, 'rating': rating} for player, rating in ordered]


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
    llr = 0.0
    for _ in range(candidate_wins):
        llr += math.log(p1 / p0)
    for _ in range(baseline_wins):
        llr += math.log((1.0 - p1) / (1.0 - p0))
    for _ in range(draws):
        llr += math.log(0.5)

    upper = math.log((1.0 - float(beta)) / float(alpha))
    lower = math.log(float(beta) / (1.0 - float(alpha)))

    if llr >= upper and win_rate >= float(min_win_rate):
        return {'decision': 'accept', 'reason': 'llr', 'win_rate': win_rate, 'avg_score_delta': avg_score_delta, 'llr': llr, 'upper': upper, 'lower': lower}
    if llr <= lower:
        return {'decision': 'reject', 'reason': 'llr', 'win_rate': win_rate, 'avg_score_delta': avg_score_delta, 'llr': llr, 'upper': upper, 'lower': lower}
    return {'decision': 'continue', 'reason': 'insufficient_evidence', 'win_rate': win_rate, 'avg_score_delta': avg_score_delta, 'llr': llr, 'upper': upper, 'lower': lower}
