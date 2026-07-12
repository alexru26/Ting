import argparse
import json
import math
import os

from action_codec import ActionCodec
from dataset import JsonlTrajectoryReader, create_mixed_split_manifest, ingest_external_jsonl
from model import CnnPolicyValueModel
from external_ingest import build_opponent_registry, ingest_games_directory
from ml_packages import package_profile


def make_feature_key(features):
    return json.dumps(features, sort_keys=True, separators=(',', ':'))


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


def _resolve_model_out_path(model_out_path):
    if os.path.isabs(model_out_path):
        return model_out_path

    src_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(src_dir)
    normalized = os.path.normpath(model_out_path)
    src_prefix = 'src' + os.sep

    if normalized == 'src':
        return src_dir

    if normalized.startswith(src_prefix):
        return os.path.join(repo_root, normalized)

    return os.path.join(src_dir, normalized)


def _ece(confidences, outcomes, bin_count=10):
    if not confidences:
        return 0.0

    total = float(len(confidences))
    ece_value = 0.0
    for i in range(bin_count):
        lo = float(i) / float(bin_count)
        hi = float(i + 1) / float(bin_count)
        members = []
        for idx, conf in enumerate(confidences):
            in_bin = (conf >= lo and conf < hi) or (i == bin_count - 1 and conf == 1.0)
            if in_bin:
                members.append(idx)
        if not members:
            continue
        avg_conf = sum(confidences[idx] for idx in members) / float(len(members))
        avg_acc = sum(outcomes[idx] for idx in members) / float(len(members))
        ece_value += abs(avg_conf - avg_acc) * (float(len(members)) / total)
    return ece_value


def load_policy_model(path):
    with open(path, 'rb') as handle:
        header = handle.read(8)

    if header == b'\x89HDF\r\n\x1a\n':
        return CnnPolicyValueModel.load(path)

    with open(path, 'r', encoding='utf-8') as handle:
        payload = json.load(handle)

    if not isinstance(payload, dict):
        raise ValueError('Invalid checkpoint payload.')
    return CnnPolicyValueModel.from_dict(payload)


def _iter_records(dataset_path, max_records=None):
    count = 0
    for record in JsonlTrajectoryReader(dataset_path):
        yield record
        count += 1
        if max_records is not None and count >= int(max_records):
            break


def _iter_training_records(dataset_path, max_records=None, decision_only=False):
    count = 0
    dropped_forced = 0
    for record in JsonlTrajectoryReader(dataset_path):
        legal_actions = list(record.legal_actions or [])
        if record.action not in legal_actions:
            legal_actions.append(record.action)

        if decision_only and len(legal_actions) <= 1:
            dropped_forced += 1
            continue

        yield record
        count += 1
        if max_records is not None and count >= int(max_records):
            break


def _count_dropped_forced(dataset_path):
    dropped = 0
    for record in JsonlTrajectoryReader(dataset_path):
        legal_actions = list(record.legal_actions or [])
        if record.action not in legal_actions:
            legal_actions.append(record.action)
        if len(legal_actions) <= 1:
            dropped += 1
    return dropped


def train_cnn(
    dataset_path,
    model_out_path,
    epochs=3,
    learning_rate=0.01,
    hidden_size=32,
    max_records=None,
    verbose=False,
    decision_only=False,
    policy_weight=1.0,
    value_weight=0.5,
    belief_weight=0.25,
    forced_policy_weight=0.0,
):
    model_out_path = _resolve_model_out_path(model_out_path)
    codec = ActionCodec()
    model = CnnPolicyValueModel(
        action_space_size=codec.size,
        hidden_size=hidden_size,
        learning_rate=learning_rate,
    )

    dropped_forced = 0
    if decision_only and max_records is None:
        dropped_forced = _count_dropped_forced(dataset_path)

    if decision_only:
        records = list(_iter_training_records(dataset_path, max_records=max_records, decision_only=True))
        if max_records is not None:
            total_in_window = 0
            for _ in _iter_records(dataset_path, max_records=max_records):
                total_in_window += 1
            dropped_forced = max(0, total_in_window - len(records))
    elif max_records is None:
        records = JsonlTrajectoryReader(dataset_path)
    else:
        records = list(_iter_records(dataset_path, max_records=max_records))

    stats = model.fit(
        records,
        epochs=epochs,
        max_records=None,
        shuffle=False,
        verbose=verbose,
        policy_weight=policy_weight,
        value_weight=value_weight,
        belief_weight=belief_weight,
        forced_policy_weight=forced_policy_weight,
    )
    model.metadata.update(
        {
            'dataset_path': dataset_path,
            'sample_count': stats['samples'],
            'epochs': int(epochs),
            'hidden_size': int(hidden_size),
            'learning_rate': float(learning_rate),
            'max_records': None if max_records is None else int(max_records),
            'verbose': bool(verbose),
            'decision_only': bool(decision_only),
            'policy_weight': float(policy_weight),
            'value_weight': float(value_weight),
            'belief_weight': float(belief_weight),
            'forced_policy_weight': float(forced_policy_weight),
            'dropped_forced_records': int(dropped_forced),
            'package_profile': package_profile(),
        }
    )
    model.save(model_out_path)
    result = model.to_dict()
    result['training_stats'] = stats
    return result


def evaluate_cnn(dataset_path, model_path, top_ks=None, max_records=None):
    model = CnnPolicyValueModel.load(model_path)

    if top_ks is None:
        top_ks = [1, 3, 5]

    top_ks = sorted(set(int(k) for k in top_ks if int(k) > 0))

    total = 0
    hit_counts = {k: 0 for k in top_ks}
    confidences = []
    outcomes = []
    masked_xent_sum = 0.0
    value_mse_sum = 0.0

    for record in _iter_records(dataset_path, max_records=max_records):
        legal_actions = list(record.legal_actions or [])
        if not legal_actions:
            continue

        if record.action not in legal_actions:
            legal_actions.append(record.action)

        probs = model.action_distribution_from_features(record.features, legal_actions)
        if not probs:
            continue

        ranked = sorted(probs.items(), key=lambda item: (-item[1], item[0]))
        ranked_actions = [action for action, _ in ranked]

        total += 1
        for k in top_ks:
            if record.action in ranked_actions[:k]:
                hit_counts[k] += 1

        p_true = max(1e-12, float(probs.get(record.action, 0.0)))
        masked_xent_sum += -math.log(p_true)

        top1 = ranked_actions[0]
        confidences.append(float(probs.get(top1, 0.0)))
        outcomes.append(1 if top1 == record.action else 0)

        value_pred = model.estimate_value_from_features(record.features)
        value_true = _safe_float(record.reward, 0.0)
        diff = value_pred - value_true
        value_mse_sum += diff * diff

    metrics = {
        'total_evaluated': total,
        'topk_accuracy': {},
        'masked_cross_entropy': 0.0,
        'value_mse': 0.0,
        'ece': _ece(confidences, outcomes),
    }

    for k in top_ks:
        if total == 0:
            metrics['topk_accuracy'][str(k)] = 0.0
        else:
            metrics['topk_accuracy'][str(k)] = float(hit_counts[k]) / float(total)

    if total > 0:
        metrics['masked_cross_entropy'] = masked_xent_sum / float(total)
        metrics['value_mse'] = value_mse_sum / float(total)

    return metrics


def choose_action_from_model(model, features, legal_actions, codec=None, belief_weight=0.0):
    if hasattr(model, 'choose_action_from_features'):
        try:
            return model.choose_action_from_features(features, legal_actions, belief_weight=belief_weight)
        except Exception:
            try:
                return model.choose_action_from_features(features, legal_actions)
            except Exception:
                return None

    return None


def _parse_topk(raw_value):
    values = []
    for part in raw_value.split(','):
        part = part.strip()
        if not part:
            continue
        values.append(int(part))
    return values


def _cmd_train_cnn(args):
    model_out = args.out or 'model.h5'
    result = train_cnn(
        dataset_path=args.dataset,
        model_out_path=model_out,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        hidden_size=args.hidden_size,
        max_records=args.max_records,
        verbose=args.verbose,
        decision_only=args.decision_only,
        policy_weight=args.policy_weight,
        value_weight=args.value_weight,
        belief_weight=args.belief_weight,
        forced_policy_weight=args.forced_policy_weight,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


def _cmd_eval_cnn(args):
    metrics = evaluate_cnn(
        dataset_path=args.dataset,
        model_path=args.model,
        top_ks=_parse_topk(args.topk),
        max_records=args.max_records,
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))


def _cmd_ingest_jsonl(args):
    stats = ingest_external_jsonl(
        input_path=args.input,
        output_path=args.output,
        source_name=args.source,
        drop_invalid=(not args.strict),
    )
    print(json.dumps(stats, indent=2, sort_keys=True))


def _cmd_ingest_games(args):
    stats = ingest_games_directory(
        games_dir=args.games_dir,
        output_path=args.output,
        source_name=args.source,
    )
    print(json.dumps(stats, indent=2, sort_keys=True))


def _cmd_make_split(args):
    manifest = create_mixed_split_manifest(
        local_dataset_path=args.local,
        external_dataset_path=args.external,
        out_manifest_path=args.output,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    summary = {
        'output': args.output,
        'seed': manifest['seed'],
        'train_local': len(manifest['splits']['train']['local_game_ids']),
        'train_external': len(manifest['splits']['train']['external_game_ids']),
        'val_local': len(manifest['splits']['val']['local_game_ids']),
        'val_external': len(manifest['splits']['val']['external_game_ids']),
        'test_local': len(manifest['splits']['test']['local_game_ids']),
        'test_external': len(manifest['splits']['test']['external_game_ids']),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


def _cmd_build_opponents(args):
    registry = build_opponent_registry(models_dir=args.models_dir, output_path=args.output)
    print(json.dumps({'output': args.output, 'count': registry['count']}, indent=2, sort_keys=True))


def main():
    parser = argparse.ArgumentParser(description='Neural CNN training tools for Ting Mahjong bot')
    sub = parser.add_subparsers(dest='command', required=True)

    train_cnn_parser = sub.add_parser('train-cnn', help='Train CNN policy-value model from JSONL trajectory dataset')
    train_cnn_parser.add_argument('--dataset', required=True, help='Input JSONL trajectory dataset path')
    train_cnn_parser.add_argument('--out', default='model.h5', help='Output model path (default: model.h5 under src/)')
    train_cnn_parser.add_argument('--epochs', type=int, default=3, help='Training epochs')
    train_cnn_parser.add_argument('--learning-rate', type=float, default=0.01, help='Learning rate')
    train_cnn_parser.add_argument('--hidden-size', type=int, default=32, help='Hidden layer size')
    train_cnn_parser.add_argument('--max-records', type=int, default=None, help='Optional record cap for smoke training')
    train_cnn_parser.add_argument('--verbose', action='store_true', help='Print per-epoch training metrics')
    train_cnn_parser.add_argument('--decision-only', action='store_true', help='Train only on records with more than one legal action')
    train_cnn_parser.add_argument('--policy-weight', type=float, default=1.0, help='Policy loss multiplier for decision states')
    train_cnn_parser.add_argument('--value-weight', type=float, default=0.5, help='Value loss multiplier')
    train_cnn_parser.add_argument('--belief-weight', type=float, default=0.25, help='Belief KL loss multiplier')
    train_cnn_parser.add_argument('--forced-policy-weight', type=float, default=0.0, help='Policy loss multiplier for forced states')
    train_cnn_parser.set_defaults(func=_cmd_train_cnn)

    eval_cnn_parser = sub.add_parser('eval-cnn', help='Evaluate CNN policy-value model on JSONL trajectory dataset')
    eval_cnn_parser.add_argument('--dataset', required=True, help='Input JSONL trajectory dataset path')
    eval_cnn_parser.add_argument('--model', required=True, help='Model JSON path')
    eval_cnn_parser.add_argument('--topk', default='1,3,5', help='Comma-separated top-k list')
    eval_cnn_parser.add_argument('--max-records', type=int, default=None, help='Optional record cap for smoke evaluation')
    eval_cnn_parser.set_defaults(func=_cmd_eval_cnn)

    ingest_jsonl_parser = sub.add_parser('ingest-jsonl', help='Normalize external JSONL records into trajectory JSONL')
    ingest_jsonl_parser.add_argument('--input', required=True, help='Input external JSONL path')
    ingest_jsonl_parser.add_argument('--output', required=True, help='Output trajectory JSONL path')
    ingest_jsonl_parser.add_argument('--source', default='external', help='Source name stored in metadata')
    ingest_jsonl_parser.add_argument('--strict', action='store_true', help='Fail on first invalid line')
    ingest_jsonl_parser.set_defaults(func=_cmd_ingest_jsonl)

    ingest_games_parser = sub.add_parser('ingest-games', help='Ingest Botzone match JSON files under a directory')
    ingest_games_parser.add_argument('--games-dir', required=True, help='Directory containing match .json files')
    ingest_games_parser.add_argument('--output', required=True, help='Output trajectory JSONL path')
    ingest_games_parser.add_argument('--source', default='external-games', help='Source name stored in metadata')
    ingest_games_parser.set_defaults(func=_cmd_ingest_games)

    split_parser = sub.add_parser('make-split', help='Create reproducible mixed local/external split manifest')
    split_parser.add_argument('--local', required=True, help='Local trajectory JSONL path')
    split_parser.add_argument('--external', required=True, help='External trajectory JSONL path')
    split_parser.add_argument('--output', required=True, help='Output split manifest JSON path')
    split_parser.add_argument('--train-ratio', type=float, default=0.8, help='Train ratio (0,1)')
    split_parser.add_argument('--val-ratio', type=float, default=0.1, help='Validation ratio [0,1)')
    split_parser.add_argument('--seed', type=int, default=42, help='Deterministic split seed')
    split_parser.set_defaults(func=_cmd_make_split)

    opponents_parser = sub.add_parser('build-opponents', help='Build opponent model registry from model folder')
    opponents_parser.add_argument('--models-dir', required=True, help='Directory containing opponent model files')
    opponents_parser.add_argument('--output', required=True, help='Output registry JSON path')
    opponents_parser.set_defaults(func=_cmd_build_opponents)

    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
