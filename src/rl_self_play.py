"""PPO fine-tuning and evaluation for the Ting Mahjong policy.

`ppo-train` fine-tunes a checkpoint with batched PPO updates collected over
whole games against a sampled opponent league (rule baseline, the frozen
IJCAI finalist imitation checkpoints, historical h5 checkpoints, and mirror
copies of the current candidate). Supports the Suphx-style oracle-feature
curriculum and Tjong-style fan-backward credit decay. `ppo-eval` measures a
candidate greedily against rule-based baselines, and `duplicate-wall` runs
paired same-wall comparisons gated by SPRT.
"""

import argparse
import glob
import json
import os
import random

from local_game import Game, REWARD_SCALE
from model import CnnPolicyValueModel
from model_governance import duplicate_wall_evaluation, sprt_promotion_gate
from features import FeatureExtractor
from finalist_opponents import FinalistOpponentPolicy, load_finalist_models
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
    """Deterministic neural policy used for opponents and evaluation.

    Opponents never see oracle features: their checkpoints were trained with
    zeroed oracle planes, so the curriculum applies to the candidate only.
    """

    def __init__(self, state, model):
        self.state = state
        self.model = model
        self.feature_extractor = FeatureExtractor()

    def choose_action(self):
        self.state.oracle_scale = 0.0
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


class OpponentLeague:
    """Samples opponents per seat: finalists, historical checkpoints, a
    frozen mirror of the current candidate, or the rule baseline."""

    def __init__(self, rng, finalist_models=None, historical_models=None, candidate_model=None,
                 finalist_prob=0.3, historical_prob=0.2, self_play_prob=0.2):
        self.rng = rng
        self.finalist_models = list(finalist_models or [])
        self.historical_models = list(historical_models or [])
        self.candidate_model = candidate_model
        self.finalist_prob = float(finalist_prob) if self.finalist_models else 0.0
        self.historical_prob = float(historical_prob) if self.historical_models else 0.0
        self.self_play_prob = float(self_play_prob) if candidate_model is not None else 0.0

    def sample_seat(self):
        """Return (kind, model) for one opponent seat."""
        roll = self.rng.random()
        if roll < self.finalist_prob:
            return 'finalist', self.rng.choice(self.finalist_models)
        roll -= self.finalist_prob
        if roll < self.historical_prob:
            return 'historical', self.rng.choice(self.historical_models)
        roll -= self.historical_prob
        if roll < self.self_play_prob:
            return 'self', self.candidate_model
        return 'rule', None

    def build_policy(self, state, kind, opponent_model):
        if kind == 'finalist':
            return FinalistOpponentPolicy(state, opponent_model)
        if kind in ('historical', 'self'):
            return NeuralGreedyPolicy(state, opponent_model)
        return GoalBasedPolicy(state)


def _collect_league_models(opponent_paths, league_dir, device):
    paths = list(opponent_paths or [])
    if league_dir and os.path.isdir(league_dir):
        paths.extend(sorted(glob.glob(os.path.join(league_dir, '*.h5'))))
    unique_paths = sorted(set(os.path.abspath(path) for path in paths))
    return _build_opponent_models(unique_paths, device=device), unique_paths


def backfill_decayed_returns(buffer, episode_return, credit_gamma):
    """Tjong-style fan-backward credit: later decisions carry more of the
    final outcome; every transition still gets a nonzero learning signal."""
    total = len(buffer)
    gamma = float(credit_gamma)
    win_target = 1.0 if episode_return > 0.0 else 0.0
    for position, transition in enumerate(buffer):
        decayed = episode_return * (gamma ** (total - 1 - position))
        transition['return_target'] = decayed
        transition['advantage'] = decayed - float(transition['value'])
        transition['win_target'] = win_target


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
    win_coef=0.1,
    ppo_epochs=4,
    minibatch_size=256,
    update_every=8,
    temperature=1.0,
    score_delta_weight=1.0,
    fan_weight=0.0,
    placement_weight=0.0,
    credit_gamma=0.97,
    oracle_start=0.0,
    oracle_end=0.0,
    finalist_dir=None,
    finalist_limit=None,
    finalist_prob=0.3,
    historical_prob=0.2,
    self_play_prob=0.2,
    league_dir=None,
    game_factory=None,
    device='cpu',
    opponent_paths=None,
    out_path=None,
    learning_rate=None,
    target_kl=None,
    snapshot_every=0,
):
    model = CnnPolicyValueModel.load(model_path, device=device)
    if learning_rate is not None:
        model.set_learning_rate(learning_rate)
    historical_models, historical_paths = _collect_league_models(opponent_paths, league_dir, device)
    finalist_models = []
    if finalist_dir:
        finalist_models = load_finalist_models(finalist_dir, device=device, limit=finalist_limit)
    game_factory = game_factory or Game
    rng = random.Random(int(seed))
    league = OpponentLeague(
        rng,
        finalist_models=finalist_models,
        historical_models=historical_models,
        candidate_model=model,
        finalist_prob=finalist_prob,
        historical_prob=historical_prob,
        self_play_prob=self_play_prob,
    )

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
        'league': {
            'finalists': [m.name for m in finalist_models],
            'historical': historical_paths,
            'opponent_seats': {'finalist': 0, 'historical': 0, 'self': 0, 'rule': 0},
        },
        'oracle': {'start': float(oracle_start), 'end': float(oracle_end)},
        'credit_gamma': float(credit_gamma),
        'learning_rate': float(model.learning_rate),
        'target_kl': None if target_kl is None else float(target_kl),
        'kl_stops': 0,
        'last_approx_kl': 0.0,
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
            win_coef=win_coef,
            epochs=ppo_epochs,
            minibatch_size=minibatch_size,
            target_kl=target_kl,
        )
        summary['updates'] += stats['updates']
        summary['policy_loss'] += stats['policy_loss']
        summary['value_loss'] += stats['value_loss']
        summary['entropy'] += stats['entropy']
        summary['last_approx_kl'] = stats.get('approx_kl', 0.0)
        if stats.get('kl_stopped'):
            summary['kl_stops'] += 1
        pending_transitions.clear()

    for episode_index in range(total_episodes):
        buffer = []
        seat_choices = {}
        for seat in range(4):
            if seat == int(candidate_seat):
                continue
            kind, opponent_model = league.sample_seat()
            seat_choices[seat] = (kind, opponent_model)
            summary['league']['opponent_seats'][kind] += 1

        if total_episodes > 1:
            progress = episode_index / float(total_episodes - 1)
        else:
            progress = 1.0
        oracle_scale = float(oracle_start) + (float(oracle_end) - float(oracle_start)) * progress

        def policy_factory(state):
            if state.my_id == int(candidate_seat):
                return PpoPolicy(state, model, buffer=buffer, temperature=temperature)
            kind, opponent_model = seat_choices[state.my_id]
            return league.build_policy(state, kind, opponent_model)

        game = game_factory(
            quan=quan,
            seed=seed + episode_index,
            policy_factory=policy_factory,
            oracle_scale=oracle_scale,
        )
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

        backfill_decayed_returns(buffer, episode_return, credit_gamma)
        pending_transitions.extend(buffer)

        summary['episodes'] += 1
        if summary['episodes'] % max(1, int(update_every)) == 0:
            _flush_updates()
        if int(snapshot_every) > 0 and summary['episodes'] % int(snapshot_every) == 0:
            snapshot_target = (out_path or model_path) + '.snapshot.h5'
            model.save(snapshot_target)
            summary['last_snapshot'] = snapshot_target
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
    summary['duplicate_wall'] = evaluate_duplicate_wall(
        model_path=save_path,
        games=max(2, int(eval_games) // 2),
        seed=seed + 20000,
        quan=quan,
        candidate_seat=candidate_seat,
        game_factory=game_factory,
        device=device,
        include_paired_results=False,
    )
    # Promotion needs both the raw-baseline gate and the paired
    # duplicate-wall gate (score-aware, low variance).
    summary['promoted'] = bool(
        promotion_gate(
            summary['evaluation'],
            min_win_rate=promote_min_win_rate,
            min_avg_score_delta=promote_min_avg_score_delta,
        )
        and summary['duplicate_wall']['avg_score_delta'] >= float(promote_min_avg_score_delta)
    )
    if summary['promoted'] and league_dir:
        os.makedirs(league_dir, exist_ok=True)
        snapshot_path = os.path.join(
            league_dir, 'league_seed%d_ep%d.h5' % (int(seed), int(summary['episodes']))
        )
        model.save(snapshot_path)
        summary['league_snapshot'] = snapshot_path
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


def evaluate_against_finalists(
    model_path,
    finalist_dir,
    games=64,
    seed=42,
    quan=0,
    game_factory=None,
    device='cpu',
    finalist_limit=None,
    show_progress=False,
):
    """Play the candidate against tables of frozen IJCAI finalists.

    The candidate seat rotates every game and the three opponents are drawn
    without replacement per game, so the summary reflects strength across
    seats and across the whole finalist pool - the metric the Botzone ladder
    actually rewards.
    """
    model = CnnPolicyValueModel.load(model_path, device=device)
    finalist_models = load_finalist_models(finalist_dir, device=device, limit=finalist_limit)
    if not finalist_models:
        raise ValueError('No finalist checkpoints found in %r' % finalist_dir)
    game_factory = game_factory or Game
    rng = random.Random(int(seed))

    summary = {
        'model_path': model_path,
        'games': 0,
        'candidate_wins': 0,
        'opponent_wins': 0,
        'draws': 0,
        'candidate_win_rate': 0.0,
        'avg_candidate_score': 0.0,
        'avg_score_delta': 0.0,
        'finalists': [m.name for m in finalist_models],
    }
    score_sum = 0.0
    delta_sum = 0.0

    total_games = max(0, int(games))
    for episode_index in range(total_games):
        candidate_seat = episode_index % 4
        opponents = rng.sample(finalist_models, 3)
        seat_models = {}
        opponent_index = 0
        for seat in range(4):
            if seat == candidate_seat:
                continue
            seat_models[seat] = opponents[opponent_index]
            opponent_index += 1

        def policy_factory(state):
            if state.my_id == candidate_seat:
                return NeuralGreedyPolicy(state, model)
            return FinalistOpponentPolicy(state, seat_models[state.my_id])

        game = game_factory(quan=quan, seed=seed + episode_index, policy_factory=policy_factory)
        result = game.run()
        summary['games'] += 1

        winner = result.get('winner')
        if winner is None:
            summary['draws'] += 1
        elif winner == candidate_seat:
            summary['candidate_wins'] += 1
        else:
            summary['opponent_wins'] += 1

        scores = list(result.get('scores', [0, 0, 0, 0]))
        score_sum += float(scores[candidate_seat])
        delta_sum += float(scores[candidate_seat]) - sum(
            float(scores[pid]) for pid in range(4) if pid != candidate_seat
        ) / 3.0
        if show_progress:
            _print_progress_bar('finalist-eval', summary['games'], total_games)

    if summary['games'] > 0:
        summary['candidate_win_rate'] = summary['candidate_wins'] / float(summary['games'])
        summary['avg_candidate_score'] = score_sum / float(summary['games'])
        summary['avg_score_delta'] = delta_sum / float(summary['games'])
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


def evaluate_duplicate_wall(model_path, games=16, seed=42, quan=0, candidate_seat=0, game_factory=None, device='cpu', include_paired_results=True):
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
    if not include_paired_results:
        summary.pop('paired_results', None)
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
    ppo_train_parser.add_argument('--win-coef', type=float, default=0.1, help='Auxiliary win-head loss coefficient')
    ppo_train_parser.add_argument('--credit-gamma', type=float, default=0.97, help='Fan-backward credit decay per decision from episode end')
    ppo_train_parser.add_argument('--oracle-start', type=float, default=0.0, help='Oracle plane scale at the first episode (Suphx-style curriculum)')
    ppo_train_parser.add_argument('--oracle-end', type=float, default=0.0, help='Oracle plane scale at the last episode')
    ppo_train_parser.add_argument('--finalist-dir', default=None, help='Directory of IJCAI finalist .pkl checkpoints (e.g. data/models)')
    ppo_train_parser.add_argument('--finalist-limit', type=int, default=None, help='Load at most N finalist checkpoints')
    ppo_train_parser.add_argument('--finalist-prob', type=float, default=0.3, help='Per-seat probability of a finalist opponent')
    ppo_train_parser.add_argument('--historical-prob', type=float, default=0.2, help='Per-seat probability of a historical checkpoint opponent')
    ppo_train_parser.add_argument('--self-play-prob', type=float, default=0.2, help='Per-seat probability of mirroring the current candidate')
    ppo_train_parser.add_argument('--league-dir', default=None, help='Directory of league snapshots; promoted candidates are saved here')
    ppo_train_parser.add_argument('--learning-rate', type=float, default=None, help='Override the checkpoint learning rate for PPO (default: keep stored value)')
    ppo_train_parser.add_argument('--target-kl', type=float, default=None, help='Stop PPO epochs early once mean approx KL exceeds this')
    ppo_train_parser.add_argument('--snapshot-every', type=int, default=0, help='Save a .snapshot.h5 checkpoint every N games (0 disables)')

    finalist_eval_parser = sub.add_parser('finalist-eval', help='Evaluate a candidate checkpoint against tables of finalist opponents')
    finalist_eval_parser.add_argument('--model', required=True, help='Model checkpoint path to evaluate')
    finalist_eval_parser.add_argument('--finalist-dir', required=True, help='Directory of IJCAI finalist .pkl checkpoints')
    finalist_eval_parser.add_argument('--finalist-limit', type=int, default=None, help='Load at most N finalist checkpoints')
    finalist_eval_parser.add_argument('--games', type=int, default=64, help='Evaluation games to run')
    finalist_eval_parser.add_argument('--seed', type=int, default=42, help='Base random seed')
    finalist_eval_parser.add_argument('--quan', type=int, default=0, help='Prevalent wind')
    finalist_eval_parser.add_argument('--device', default='cpu', help='Torch device')
    finalist_eval_parser.add_argument('--verbose', action='store_true', help='Print per-game progress')

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
            win_coef=args.win_coef,
            ppo_epochs=args.ppo_epochs,
            minibatch_size=args.minibatch_size,
            update_every=args.update_every,
            temperature=args.temperature,
            score_delta_weight=args.score_delta_weight,
            fan_weight=args.fan_weight,
            placement_weight=args.placement_weight,
            credit_gamma=args.credit_gamma,
            oracle_start=args.oracle_start,
            oracle_end=args.oracle_end,
            finalist_dir=args.finalist_dir,
            finalist_limit=args.finalist_limit,
            finalist_prob=args.finalist_prob,
            historical_prob=args.historical_prob,
            self_play_prob=args.self_play_prob,
            league_dir=args.league_dir,
            device=args.device,
            opponent_paths=args.opponents,
            out_path=args.out,
            learning_rate=args.learning_rate,
            target_kl=args.target_kl,
            snapshot_every=args.snapshot_every,
        )
    elif args.command == 'finalist-eval':
        summary = evaluate_against_finalists(
            model_path=args.model,
            finalist_dir=args.finalist_dir,
            games=args.games,
            seed=args.seed,
            quan=args.quan,
            device=args.device,
            finalist_limit=args.finalist_limit,
            show_progress=args.verbose,
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
