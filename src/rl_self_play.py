import argparse
import json
import os
import random

from local_game import Game
from policy import GoalBasedPolicy
from model import CnnPolicyValueModel
from features import FeatureExtractor
from model_governance import build_model_registry_entry, duplicate_wall_evaluation, sprt_promotion_gate, write_model_registry
from action_codec import ActionCodec
import runtime_model


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


def _repo_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _resolve_registry_model_path(path):
    if not path:
        return None
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(_repo_root(), path))


class _LoadedOpponentPolicy:
    def __init__(self, state, model, temperature=1.0, belief_weight=0.25, efficiency_weight=0.2):
        self.state = state
        self.model = model
        self.temperature = float(temperature)
        self.belief_weight = float(belief_weight)
        self.efficiency_weight = float(efficiency_weight)
        self.feature_extractor = FeatureExtractor()

    def choose_action(self):
        legal_actions = self.state.enumerate_legal_actions()
        if not legal_actions:
            return 'PASS'

        features = self.feature_extractor.extract(self.state)
        action = runtime_model.choose_action_from_model(
            model=self.model,
            features=features,
            legal_actions=legal_actions,
            codec=ActionCodec(),
            belief_weight=self.belief_weight,
            efficiency_weight=self.efficiency_weight,
            temperature=self.temperature,
        )
        if action and self.state.is_legal_action(action):
            return action
        return GoalBasedPolicy(self.state).choose_action()


def _load_registry_model(row, model_cache):
    path = _resolve_registry_model_path(row.get('path'))
    if not path:
        return None
    if path in model_cache:
        return model_cache[path]

    try:
        model_cache[path] = runtime_model.load_policy_model(path)
    except Exception:
        model_cache[path] = None
    return model_cache[path]


def _sample_opponent_rows(pool, opponent_count, seed):
    opponent_count = max(0, _safe_int(opponent_count, 0))
    rows = [row for row in list(pool or []) if row.get('kind') != 'candidate']
    if not rows:
        rows = [{'id': 'baseline-rule', 'kind': 'baseline', 'policy_mode': 'rule', 'path': None}]

    rng = random.Random(int(seed))
    rng.shuffle(rows)

    selected = []
    while len(selected) < opponent_count:
        for row in rows:
            selected.append(row)
            if len(selected) >= opponent_count:
                break
        if len(rows) > 1:
            rng.shuffle(rows)
    return selected[:opponent_count]


def _build_episode_policy_factory(candidate_model, candidate_seat, opponent_rows, candidate_buffer, model_cache):
    seat_rows = {int(seat): row for seat, row in opponent_rows.items()}

    def policy_factory(state):
        seat = int(state.my_id)
        if seat == int(candidate_seat):
            return PpoPolicy(state, candidate_model, buffer=candidate_buffer, explore=True, temperature=1.0)

        row = seat_rows.get(seat)
        if not row:
            return GoalBasedPolicy(state)

        policy_mode = str(row.get('policy_mode', 'rule')).strip().lower()
        if policy_mode == 'rule' or row.get('kind') == 'baseline':
            return GoalBasedPolicy(state)

        model = _load_registry_model(row, model_cache)
        if model is None:
            return GoalBasedPolicy(state)

        return _LoadedOpponentPolicy(state, model)

    return policy_factory


def _print_progress_bar(prefix, current, total, width=32):
    total_value = max(1, int(total))
    current_value = max(0, min(int(current), total_value))
    ratio = float(current_value) / float(total_value)
    filled = int(float(width) * ratio)
    bar = ('#' * filled) + ('-' * (width - filled))
    print('\r%s [%s] %d/%d (%.1f%%)' % (prefix, bar, current_value, total_value, ratio * 100.0), end='', flush=True)
    if current_value >= total_value:
        print('')


def _load_external_opponents(registry_path):
    if not registry_path:
        return []
    if not os.path.exists(registry_path):
        return []

    with open(registry_path, 'r', encoding='utf-8') as handle:
        payload = json.load(handle)

    rows = payload.get('opponents', []) if isinstance(payload, dict) else []
    result = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        result.append(
            {
                'id': row.get('id') or row.get('file_name') or 'external-opponent',
                'kind': 'external',
                'path': row.get('path') or row.get('file_name'),
                'policy_mode': row.get('policy_mode', 'imitation'),
            }
        )
    return result


def build_opponent_league_pool(
    baseline_id='baseline-rule',
    candidate_path=None,
    historical_checkpoints=None,
    external_registry_path=None,
    max_external=27,
):
    pool = []
    pool.append({'id': baseline_id, 'kind': 'baseline', 'policy_mode': 'rule', 'path': None})

    if candidate_path:
        pool.append(
            {
                'id': os.path.basename(candidate_path),
                'kind': 'candidate',
                'policy_mode': 'neural',
                'path': candidate_path,
            }
        )

    for checkpoint_path in list(historical_checkpoints or []):
        pool.append(
            {
                'id': os.path.basename(checkpoint_path),
                'kind': 'historical',
                'policy_mode': 'neural',
                'path': checkpoint_path,
            }
        )

    external = _load_external_opponents(external_registry_path)
    for row in external[: max(0, _safe_int(max_external, 27))]:
        pool.append(row)

    return pool


def _placement_proxy(scores):
    ordered = sorted([(score, pid) for pid, score in enumerate(scores)], reverse=True)
    placement = [0.0, 0.0, 0.0, 0.0]
    proxy_by_rank = [1.0, 0.33, -0.33, -1.0]
    for rank, (_score, pid) in enumerate(ordered):
        placement[pid] = proxy_by_rank[rank]
    return placement


def shape_rewards(result, score_delta_weight=1.0, fan_weight=0.0, placement_weight=0.0):
    scores = list(result.get('scores', [0, 0, 0, 0]))
    winner = result.get('winner')
    fan = _safe_float(result.get('fan', 0.0), 0.0)
    placement = _placement_proxy(scores)

    rewards = []
    for pid, score in enumerate(scores):
        reward = _safe_float(score_delta_weight, 1.0) * _safe_float(score, 0.0)
        if winner is not None and pid == winner:
            reward += _safe_float(fan_weight, 0.0) * fan
        reward += _safe_float(placement_weight, 0.0) * placement[pid]
        rewards.append(float(reward))
    return rewards


class EpisodeBuffer:
    def __init__(self):
        self.transitions = []

    def add(self, transition):
        self.transitions.append(dict(transition))


class PpoPolicy:
    def __init__(self, state, model, buffer=None, explore=True, temperature=1.0):
        self.state = state
        self.model = model
        self.buffer = buffer
        self.explore = bool(explore)
        self.temperature = float(temperature)
        self.feature_extractor = FeatureExtractor()
        self.decision_info = None

    def choose_action(self):
        legal_actions = self.state.enumerate_legal_actions()
        if not legal_actions:
            self.decision_info = {
                'actions': [],
                'selected_action': 'PASS',
                'selected_index': 0,
                'selected_log_prob': 0.0,
                'selected_probability': 1.0,
                'value': 0.0,
                'entropy': 0.0,
            }
            return 'PASS'

        features = self.feature_extractor.extract(self.state)
        action, info = self.model.sample_action_from_features(
            features,
            legal_actions,
            temperature=self.temperature,
            greedy=not self.explore,
        )
        if action is None:
            action = 'PASS'
            info = {
                'actions': legal_actions,
                'selected_action': 'PASS',
                'selected_index': 0,
                'selected_log_prob': 0.0,
                'selected_probability': 1.0,
                'value': 0.0,
                'entropy': 0.0,
            }

        self.decision_info = info
        if self.buffer is not None:
            self.buffer.add(
                {
                    'features': features,
                    'legal_actions': legal_actions,
                    'action': action,
                    'old_log_prob': info.get('selected_log_prob', 0.0),
                    'value': info.get('value', 0.0),
                    'entropy': info.get('entropy', 0.0),
                    'player_id': self.state.my_id,
                }
            )
        return action


def run_self_play_worker(
    games,
    seed=42,
    quan=0,
    game_factory=None,
    score_delta_weight=1.0,
    fan_weight=0.0,
    placement_weight=0.0,
):
    game_factory = game_factory or Game
    games = max(0, _safe_int(games, 0))

    summary = {
        'games': games,
        'wins': [0, 0, 0, 0],
        'draws': 0,
        'avg_fan': 0.0,
        'reward_totals': [0.0, 0.0, 0.0, 0.0],
        'mean_scores': [0.0, 0.0, 0.0, 0.0],
    }

    fan_sum = 0.0
    score_sum = [0.0, 0.0, 0.0, 0.0]

    for game_index in range(games):
        game = game_factory(quan=quan, seed=seed + game_index)
        result = game.run()

        winner = result.get('winner')
        if winner is None:
            summary['draws'] += 1
        else:
            summary['wins'][winner] += 1

        fan = _safe_float(result.get('fan', 0.0), 0.0)
        fan_sum += fan

        scores = list(result.get('scores', [0, 0, 0, 0]))
        for pid in range(4):
            score_sum[pid] += _safe_float(scores[pid], 0.0)

        rewards = shape_rewards(
            result,
            score_delta_weight=score_delta_weight,
            fan_weight=fan_weight,
            placement_weight=placement_weight,
        )
        for pid in range(4):
            summary['reward_totals'][pid] += rewards[pid]

    if games > 0:
        summary['avg_fan'] = fan_sum / float(games)
        summary['mean_scores'] = [value / float(games) for value in score_sum]

    return summary


def run_ppo_fine_tuning(
    model_path,
    games=32,
    seed=42,
    quan=0,
    candidate_seat=0,
    eval_games=16,
    promote_min_win_rate=0.55,
    promote_min_avg_score_delta=0.0,
    clip_range=0.2,
    entropy_coef=0.01,
    value_coef=0.5,
    score_delta_weight=1.0,
    fan_weight=0.0,
    placement_weight=0.0,
    game_factory=None,
    device='cpu',
    opponent_registry_path=None,
):
    model = CnnPolicyValueModel.load(model_path, device=device)
    game_factory = game_factory or Game
    registry_path = opponent_registry_path or _resolve_registry_model_path(os.path.join('data', 'opponents_registry.json'))
    opponent_pool = build_opponent_league_pool(
        candidate_path=model_path,
        external_registry_path=registry_path,
    )
    model_cache = {}

    train_summary = {
        'model_path': model_path,
        'requested_device': model.requested_device,
        'resolved_device': model.resolved_device,
        'episodes': 0,
        'policy_updates': 0,
        'policy_loss': 0.0,
        'value_loss': 0.0,
        'entropy': 0.0,
        'ratio': 0.0,
        'reward_total': 0.0,
        'wins': [0, 0, 0, 0],
        'draws': 0,
    }

    total_episodes = max(0, _safe_int(games, 0))

    for episode_index in range(total_episodes):
        buffer = EpisodeBuffer()
        opponent_rows = _sample_opponent_rows(opponent_pool, 4 - 1, seed + episode_index)
        opponent_seats = [seat for seat in range(4) if seat != int(candidate_seat)]
        seat_opponents = {seat: row for seat, row in zip(opponent_seats, opponent_rows)}
        policy_factory = _build_episode_policy_factory(
            candidate_model=model,
            candidate_seat=candidate_seat,
            opponent_rows=seat_opponents,
            candidate_buffer=buffer,
            model_cache=model_cache,
        )

        game = game_factory(quan=quan, seed=seed + episode_index, policy_factory=policy_factory)
        result = game.run()
        rewards = shape_rewards(
            result,
            score_delta_weight=score_delta_weight,
            fan_weight=fan_weight,
            placement_weight=placement_weight,
        )

        winner = result.get('winner')
        if winner is None:
            train_summary['draws'] += 1
        else:
            train_summary['wins'][winner] += 1

        episode_reward = float(rewards[int(candidate_seat)])
        train_summary['reward_total'] += episode_reward

        for transition in buffer.transitions:
            advantage = episode_reward - _safe_float(transition.get('value', 0.0), 0.0)
            update = model.ppo_train_step(
                transition['features'],
                transition['legal_actions'],
                transition['action'],
                advantage,
                episode_reward,
                old_log_prob=transition.get('old_log_prob', 0.0),
                clip_range=clip_range,
                entropy_coef=entropy_coef,
                value_coef=value_coef,
            )
            train_summary['policy_updates'] += 1
            train_summary['policy_loss'] += update['policy_loss']
            train_summary['value_loss'] += update['value_loss']
            train_summary['entropy'] += update['entropy']
            train_summary['ratio'] += update['ratio']

        train_summary['episodes'] += 1
    _print_progress_bar('ppo-train', train_summary['episodes'], total_episodes)

    model.save(model_path)
    train_summary['evaluation'] = evaluate_against_baseline(
        model_path=model_path,
        games=eval_games,
        seed=seed + 10000,
        quan=quan,
        candidate_seat=candidate_seat,
        game_factory=game_factory,
        device=device,
    )
    train_summary['promoted'] = promotion_gate(
        train_summary['evaluation'],
        min_win_rate=promote_min_win_rate,
        min_avg_score_delta=promote_min_avg_score_delta,
    )
    return train_summary


def evaluate_against_baseline(model_path, games=16, seed=42, quan=0, candidate_seat=0, game_factory=None, device='cpu'):
    model = CnnPolicyValueModel.load(model_path, device=device)
    game_factory = game_factory or Game

    summary = {
        'games': 0,
        'wins': [0, 0, 0, 0],
        'draws': 0,
        'mean_scores': [0.0, 0.0, 0.0, 0.0],
        'avg_score_delta': 0.0,
        'candidate_win_rate': 0.0,
        'candidate_seat': int(candidate_seat),
    }

    score_sum = [0.0, 0.0, 0.0, 0.0]
    score_delta_sum = 0.0

    for episode_index in range(max(0, _safe_int(games, 0))):
        def policy_factory(state):
            if state.my_id == int(candidate_seat):
                return PpoPolicy(state, model, buffer=None, explore=False, temperature=0.0)
            return GoalBasedPolicy(state)

        game = game_factory(quan=quan, seed=seed + episode_index, policy_factory=policy_factory)
        result = game.run()
        summary['games'] += 1

        winner = result.get('winner')
        if winner is None:
            summary['draws'] += 1
        else:
            summary['wins'][winner] += 1

        scores = list(result.get('scores', [0, 0, 0, 0]))
        for pid in range(4):
            score_sum[pid] += _safe_float(scores[pid], 0.0)
        score_delta_sum += _safe_float(scores[int(candidate_seat)], 0.0) - sum(
            _safe_float(scores[pid], 0.0) for pid in range(4) if pid != int(candidate_seat)
        ) / 3.0

    if summary['games'] > 0:
        summary['mean_scores'] = [value / float(summary['games']) for value in score_sum]
        summary['avg_score_delta'] = score_delta_sum / float(summary['games'])
        summary['candidate_win_rate'] = summary['wins'][int(candidate_seat)] / float(summary['games'])

    return summary


def promotion_gate(evaluation_summary, min_win_rate=0.55, min_avg_score_delta=0.0):
    if not evaluation_summary:
        return False
    result = sprt_promotion_gate(
        evaluation_summary,
        min_win_rate=min_win_rate,
        min_avg_score_delta=min_avg_score_delta,
    )
    return result.get('decision') == 'accept'


def evaluate_duplicate_wall(model_path, games=16, seed=42, quan=0, candidate_seat=0, game_factory=None):
    model = CnnPolicyValueModel.load(model_path)
    game_factory = game_factory or Game

    def candidate_policy_factory(state):
        if state.my_id == int(candidate_seat):
            return PpoPolicy(state, model, buffer=None, explore=False, temperature=0.0)
        return GoalBasedPolicy(state)

    def baseline_policy_factory(state):
        return GoalBasedPolicy(state)

    seeds = [seed + index for index in range(max(0, _safe_int(games, 0)))]
    summary = duplicate_wall_evaluation(
        seeds=seeds,
        game_factory=game_factory,
        candidate_policy_factory=candidate_policy_factory,
        baseline_policy_factory=baseline_policy_factory,
        quan=quan,
    )
    summary['model_path'] = model_path
    summary['candidate_seat'] = int(candidate_seat)
    return summary


def main():
    parser = argparse.ArgumentParser(description='Phase 4 self-play worker and league utilities')
    sub = parser.add_subparsers(dest='command', required=True)

    worker_parser = sub.add_parser('self-play', help='Run self-play worker loop')
    worker_parser.add_argument('--games', type=int, default=100, help='Number of self-play games')
    worker_parser.add_argument('--seed', type=int, default=42, help='Base random seed')
    worker_parser.add_argument('--quan', type=int, default=0, help='Prevalent wind')
    worker_parser.add_argument('--score-delta-weight', type=float, default=1.0, help='Weight for score-delta reward')
    worker_parser.add_argument('--fan-weight', type=float, default=0.0, help='Weight for winner fan bonus')
    worker_parser.add_argument('--placement-weight', type=float, default=0.0, help='Weight for placement proxy')

    league_parser = sub.add_parser('league-pool', help='Build opponent league pool descriptor')
    league_parser.add_argument('--candidate', default=None, help='Candidate model path')
    league_parser.add_argument('--historical', nargs='*', default=[], help='Historical checkpoint model paths')
    league_parser.add_argument('--external-registry', default=None, help='Path to external opponents registry JSON')
    league_parser.add_argument('--max-external', type=int, default=27, help='Max external opponents to include')

    ppo_train_parser = sub.add_parser('ppo-train', help='Run PPO fine-tuning against rule-based opponents')
    ppo_train_parser.add_argument('--model', required=True, help='Model checkpoint path to fine-tune')
    ppo_train_parser.add_argument('--games', type=int, default=32, help='Training games to run')
    ppo_train_parser.add_argument('--eval-games', type=int, default=16, help='Evaluation games for promotion gate')
    ppo_train_parser.add_argument('--seed', type=int, default=42, help='Base random seed')
    ppo_train_parser.add_argument('--quan', type=int, default=0, help='Prevalent wind')
    ppo_train_parser.add_argument('--candidate-seat', type=int, default=0, help='Seat index controlled by the candidate')
    ppo_train_parser.add_argument('--promote-min-win-rate', type=float, default=0.55, help='Minimum win rate for promotion')
    ppo_train_parser.add_argument('--promote-min-avg-score-delta', type=float, default=0.0, help='Minimum average score delta for promotion')
    ppo_train_parser.add_argument('--clip-range', type=float, default=0.2, help='PPO clipping range')
    ppo_train_parser.add_argument('--entropy-coef', type=float, default=0.01, help='Entropy regularization coefficient')
    ppo_train_parser.add_argument('--value-coef', type=float, default=0.5, help='Value loss coefficient')
    ppo_train_parser.add_argument('--score-delta-weight', type=float, default=1.0, help='Reward weight for score delta')
    ppo_train_parser.add_argument('--fan-weight', type=float, default=0.0, help='Reward weight for fan bonus')
    ppo_train_parser.add_argument('--placement-weight', type=float, default=0.0, help='Reward weight for placement proxy')
    ppo_train_parser.add_argument('--device', default='cpu', help='Torch device: cpu, cuda, cuda:0, or auto')
    ppo_train_parser.add_argument('--opponent-registry', default=None, help='Path to opponents_registry.json for external self-play opponents')

    ppo_eval_parser = sub.add_parser('ppo-eval', help='Evaluate a candidate checkpoint against baseline opponents')
    ppo_eval_parser.add_argument('--model', required=True, help='Model checkpoint path to evaluate')
    ppo_eval_parser.add_argument('--games', type=int, default=16, help='Evaluation games to run')
    ppo_eval_parser.add_argument('--seed', type=int, default=42, help='Base random seed')
    ppo_eval_parser.add_argument('--quan', type=int, default=0, help='Prevalent wind')
    ppo_eval_parser.add_argument('--candidate-seat', type=int, default=0, help='Seat index controlled by the candidate')
    ppo_eval_parser.add_argument('--device', default='cpu', help='Torch device: cpu, cuda, cuda:0, or auto')

    args = parser.parse_args()

    if args.command == 'self-play':
        summary = run_self_play_worker(
            games=args.games,
            seed=args.seed,
            quan=args.quan,
            score_delta_weight=args.score_delta_weight,
            fan_weight=args.fan_weight,
            placement_weight=args.placement_weight,
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
        return

    if args.command == 'ppo-train':
        summary = run_ppo_fine_tuning(
            model_path=args.model,
            games=args.games,
            seed=args.seed,
            quan=args.quan,
            candidate_seat=args.candidate_seat,
            eval_games=args.eval_games,
            promote_min_win_rate=args.promote_min_win_rate,
            promote_min_avg_score_delta=args.promote_min_avg_score_delta,
            clip_range=args.clip_range,
            entropy_coef=args.entropy_coef,
            value_coef=args.value_coef,
            score_delta_weight=args.score_delta_weight,
            fan_weight=args.fan_weight,
            placement_weight=args.placement_weight,
            device=args.device,
            opponent_registry_path=args.opponent_registry,
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
        return

    if args.command == 'ppo-eval':
        summary = evaluate_against_baseline(
            model_path=args.model,
            games=args.games,
            seed=args.seed,
            quan=args.quan,
            candidate_seat=args.candidate_seat,
            device=args.device,
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
        return

    if args.command == 'duplicate-wall':
        summary = evaluate_duplicate_wall(
            model_path=args.model,
            games=args.games,
            seed=args.seed,
            quan=args.quan,
            candidate_seat=args.candidate_seat,
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
        return

    pool = build_opponent_league_pool(
        candidate_path=args.candidate,
        historical_checkpoints=args.historical,
        external_registry_path=args.external_registry,
        max_external=args.max_external,
    )
    print(json.dumps({'count': len(pool), 'pool': pool}, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()