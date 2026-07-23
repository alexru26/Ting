"""PPO fine-tuning and evaluation for the Ting Mahjong policy.

`ppo-train` fine-tunes a checkpoint with batched PPO updates collected over
whole games against rule-based and optional historical neural opponents.
`ppo-eval` measures a candidate greedily against rule-based baselines, and
`duplicate-wall` runs paired same-wall comparisons for low-variance gating.
"""

import argparse
import json
import random

from local_game import Game, REWARD_SCALE
from model import CnnPolicyValueModel
from model_governance import duplicate_wall_evaluation, sprt_promotion_gate
from features import FeatureExtractor
from rule_policy import GoalBasedPolicy


def _print_progress_bar(prefix, current, total, width=32):
    total_value = max(1, int(total))
    current_value = max(0, min(int(current), total_value))
    ratio = float(current_value) / float(total_value)
    filled = int(float(width) * ratio)
    bar = ('#' * filled) + ('-' * (width - filled))
    print('\r%s [%s] %d/%d (%.1f%%)' % (prefix, bar, current_value, total_value, ratio * 100.0), end='', flush=True)
    if current_value >= total_value:
        print('')


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
    fan = float(result.get('fan', 0.0) or 0.0)
    placement = _placement_proxy(scores)

    rewards = []
    for pid, score in enumerate(scores):
        reward = float(score_delta_weight) * float(score)
        if winner is not None and pid == winner:
            reward += float(fan_weight) * fan
        reward += float(placement_weight) * placement[pid]
        rewards.append(float(reward))
    return rewards


class NeuralGreedyPolicy:
    """Deterministic neural policy used for opponents and evaluation."""

    def __init__(self, state, model):
        self.state = state
        self.model = model
        self.feature_extractor = FeatureExtractor()

    def choose_action(self):
        legal_actions = self.state.enumerate_legal_actions()
        if not legal_actions:
            raise ValueError('No legal actions available')
        if len(legal_actions) == 1:
            return legal_actions[0]
        features = self.feature_extractor.extract(self.state)
        return self.model.choose_action_from_features(features, legal_actions)


class PpoPolicy:
    """Sampling policy that records transitions for PPO."""

    def __init__(self, state, model, buffer=None, temperature=1.0):
        self.state = state
        self.model = model
        self.buffer = buffer
        self.temperature = float(temperature)

    def choose_action(self):
        legal_actions = self.state.enumerate_legal_actions()
        if not legal_actions:
            raise ValueError('No legal actions available')
        if len(legal_actions) == 1:
            return legal_actions[0]

        features = FeatureExtractor().extract(self.state)
        action, info = self.model.sample_action_from_features(
            features, legal_actions, temperature=self.temperature
        )
        if self.buffer is not None:
            self.buffer.append(
                {
                    'features': features,
                    'legal_actions': legal_actions,
                    'action': action,
                    'old_log_prob': info['selected_log_prob'],
                    'value': info['value'],
                }
            )
        return action


def _build_opponent_models(opponent_paths, device='cpu'):
    """Load historical neural opponents; a bad path fails fast."""
    return [CnnPolicyValueModel.load(path, device=device) for path in list(opponent_paths or [])]


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
    ppo_epochs=4,
    minibatch_size=256,
    update_every=8,
    temperature=1.0,
    score_delta_weight=1.0,
    fan_weight=0.0,
    placement_weight=0.0,
    game_factory=None,
    device='cpu',
    opponent_paths=None,
    out_path=None,
):
    model = CnnPolicyValueModel.load(model_path, device=device)
    opponent_models = _build_opponent_models(opponent_paths, device=device)
    game_factory = game_factory or Game
    rng = random.Random(int(seed))

    summary = {
        'model_path': model_path,
        'resolved_device': model.resolved_device,
        'episodes': 0,
        'updates': 0,
        'policy_loss': 0.0,
        'value_loss': 0.0,
        'entropy': 0.0,
        'reward_total': 0.0,
        'wins': [0, 0, 0, 0],
        'draws': 0,
    }

    total_episodes = max(0, int(games))
    if total_episodes > 0:
        _print_progress_bar('ppo-train', 0, total_episodes)

    pending_transitions = []

    def _flush_updates():
        if not pending_transitions:
            return
        stats = model.ppo_update(
            pending_transitions,
            clip_range=clip_range,
            entropy_coef=entropy_coef,
            value_coef=value_coef,
            epochs=ppo_epochs,
            minibatch_size=minibatch_size,
        )
        summary['updates'] += stats['updates']
        summary['policy_loss'] += stats['policy_loss']
        summary['value_loss'] += stats['value_loss']
        summary['entropy'] += stats['entropy']
        pending_transitions.clear()

    for episode_index in range(total_episodes):
        buffer = []
        seat_models = {}
        for seat in range(4):
            if seat == int(candidate_seat):
                continue
            if opponent_models and rng.random() < 0.5:
                seat_models[seat] = rng.choice(opponent_models)

        def policy_factory(state):
            if state.my_id == int(candidate_seat):
                return PpoPolicy(state, model, buffer=buffer, temperature=temperature)
            opponent_model = seat_models.get(state.my_id)
            if opponent_model is not None:
                return NeuralGreedyPolicy(state, opponent_model)
            return GoalBasedPolicy(state)

        game = game_factory(quan=quan, seed=seed + episode_index, policy_factory=policy_factory)
        result = game.run()

        winner = result.get('winner')
        if winner is None:
            summary['draws'] += 1
        else:
            summary['wins'][winner] += 1

        rewards = shape_rewards(
            result,
            score_delta_weight=score_delta_weight,
            fan_weight=fan_weight,
            placement_weight=placement_weight,
        )
        episode_return = float(rewards[int(candidate_seat)]) / REWARD_SCALE
        summary['reward_total'] += episode_return

        for transition in buffer:
            transition['return_target'] = episode_return
            transition['advantage'] = episode_return - float(transition['value'])
        pending_transitions.extend(buffer)

        summary['episodes'] += 1
        if summary['episodes'] % max(1, int(update_every)) == 0:
            _flush_updates()
        _print_progress_bar('ppo-train', summary['episodes'], total_episodes)

    _flush_updates()

    save_path = out_path or model_path
    model.save(save_path)
    summary['saved_model_path'] = save_path
    summary['evaluation'] = evaluate_against_baseline(
        model_path=save_path,
        games=eval_games,
        seed=seed + 10000,
        quan=quan,
        candidate_seat=candidate_seat,
        game_factory=game_factory,
        device=device,
    )
    summary['promoted'] = promotion_gate(
        summary['evaluation'],
        min_win_rate=promote_min_win_rate,
        min_avg_score_delta=promote_min_avg_score_delta,
    )
    return summary


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

    for episode_index in range(max(0, int(games))):
        def policy_factory(state):
            if state.my_id == int(candidate_seat):
                return NeuralGreedyPolicy(state, model)
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
            score_sum[pid] += float(scores[pid])
        score_delta_sum += float(scores[int(candidate_seat)]) - sum(
            float(scores[pid]) for pid in range(4) if pid != int(candidate_seat)
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


def evaluate_duplicate_wall(model_path, games=16, seed=42, quan=0, candidate_seat=0, game_factory=None, device='cpu'):
    model = CnnPolicyValueModel.load(model_path, device=device)
    game_factory = game_factory or Game

    def candidate_policy_factory(state):
        if state.my_id == int(candidate_seat):
            return NeuralGreedyPolicy(state, model)
        return GoalBasedPolicy(state)

    def baseline_policy_factory(state):
        return GoalBasedPolicy(state)

    seeds = [seed + index for index in range(max(0, int(games)))]
    summary = duplicate_wall_evaluation(
        seeds=seeds,
        game_factory=game_factory,
        candidate_policy_factory=candidate_policy_factory,
        baseline_policy_factory=baseline_policy_factory,
        quan=quan,
        candidate_seat=candidate_seat,
    )
    summary['model_path'] = model_path
    summary['candidate_seat'] = int(candidate_seat)
    return summary


def main():
    parser = argparse.ArgumentParser(description='PPO fine-tuning and evaluation utilities')
    sub = parser.add_subparsers(dest='command', required=True)

    ppo_train_parser = sub.add_parser('ppo-train', help='Run PPO fine-tuning against rule-based and neural opponents')
    ppo_train_parser.add_argument('--model', required=True, help='Model checkpoint path to fine-tune')
    ppo_train_parser.add_argument('--out', default=None, help='Output checkpoint path (default: overwrite --model)')
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
    ppo_train_parser.add_argument('--ppo-epochs', type=int, default=4, help='PPO epochs per update')
    ppo_train_parser.add_argument('--minibatch-size', type=int, default=256, help='PPO minibatch size')
    ppo_train_parser.add_argument('--update-every', type=int, default=8, help='Games collected per PPO update')
    ppo_train_parser.add_argument('--temperature', type=float, default=1.0, help='Sampling temperature for exploration')
    ppo_train_parser.add_argument('--score-delta-weight', type=float, default=1.0, help='Reward weight for score delta')
    ppo_train_parser.add_argument('--fan-weight', type=float, default=0.0, help='Reward weight for fan bonus')
    ppo_train_parser.add_argument('--placement-weight', type=float, default=0.0, help='Reward weight for placement proxy')
    ppo_train_parser.add_argument('--device', default='cpu', help='Torch device: cpu, cuda, cuda:0, or auto')
    ppo_train_parser.add_argument('--opponents', nargs='*', default=[], help='Historical checkpoint paths used as neural opponents')

    ppo_eval_parser = sub.add_parser('ppo-eval', help='Evaluate a candidate checkpoint against rule-based baselines')
    ppo_eval_parser.add_argument('--model', required=True, help='Model checkpoint path to evaluate')
    ppo_eval_parser.add_argument('--games', type=int, default=16, help='Evaluation games to run')
    ppo_eval_parser.add_argument('--seed', type=int, default=42, help='Base random seed')
    ppo_eval_parser.add_argument('--quan', type=int, default=0, help='Prevalent wind')
    ppo_eval_parser.add_argument('--candidate-seat', type=int, default=0, help='Seat index controlled by the candidate')
    ppo_eval_parser.add_argument('--device', default='cpu', help='Torch device')

    duplicate_parser = sub.add_parser('duplicate-wall', help='Paired same-wall candidate vs baseline evaluation')
    duplicate_parser.add_argument('--model', required=True, help='Model checkpoint path to evaluate')
    duplicate_parser.add_argument('--games', type=int, default=16, help='Paired games to run')
    duplicate_parser.add_argument('--seed', type=int, default=42, help='Base random seed')
    duplicate_parser.add_argument('--quan', type=int, default=0, help='Prevalent wind')
    duplicate_parser.add_argument('--candidate-seat', type=int, default=0, help='Seat index controlled by the candidate')
    duplicate_parser.add_argument('--device', default='cpu', help='Torch device')

    args = parser.parse_args()

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
            ppo_epochs=args.ppo_epochs,
            minibatch_size=args.minibatch_size,
            update_every=args.update_every,
            temperature=args.temperature,
            score_delta_weight=args.score_delta_weight,
            fan_weight=args.fan_weight,
            placement_weight=args.placement_weight,
            device=args.device,
            opponent_paths=args.opponents,
            out_path=args.out,
        )
    elif args.command == 'ppo-eval':
        summary = evaluate_against_baseline(
            model_path=args.model,
            games=args.games,
            seed=args.seed,
            quan=args.quan,
            candidate_seat=args.candidate_seat,
            device=args.device,
        )
    else:
        summary = evaluate_duplicate_wall(
            model_path=args.model,
            games=args.games,
            seed=args.seed,
            quan=args.quan,
            candidate_seat=args.candidate_seat,
            device=args.device,
        )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
