import argparse
import json
import math
import os
import time

from action_codec import ActionCodec
from dataset import JsonlTrajectoryReader, create_mixed_split_manifest, ingest_external_jsonl
from model import CnnPolicyValueModel
from external_ingest import build_opponent_registry, ingest_games_directory
from ml_packages import package_profile
import runtime_model


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


def _brier(confidences, outcomes):
    if not confidences:
        return 0.0
    total = 0.0
    for conf, outcome in zip(confidences, outcomes):
        delta = float(conf) - float(outcome)
        total += delta * delta
    return total / float(len(confidences))


def _temperature_rescale_distribution(probs, temperature):
    if not probs:
        return {}
    t = max(1e-3, float(temperature))
    weighted = {}
    total = 0.0
    for action, prob in probs.items():
        p = max(1e-12, float(prob))
        value = p ** (1.0 / t)
        weighted[action] = value
        total += value
    if total <= 0.0:
        uniform = 1.0 / float(len(weighted))
        return {action: uniform for action in weighted}
    return {action: value / total for action, value in weighted.items()}


def _select_calibration_temperature(model, dataset_path, max_records=None, verbose=False):
    candidate_temperatures = [0.7, 0.85, 1.0, 1.15, 1.3, 1.5]
    best_temperature = 1.0
    best_nll = None
    sweep_start = time.perf_counter()

    if verbose:
        print('starting calibration sweep over %d temperatures' % len(candidate_temperatures))

    for temperature in candidate_temperatures:
        candidate_start = time.perf_counter()
        nll_sum = 0.0
        evaluated = 0
        for record in _iter_records(dataset_path, max_records=max_records):
            legal_actions = list(record.legal_actions or [])
            if not legal_actions:
                continue
            if record.action not in legal_actions:
                legal_actions.append(record.action)

            probs = model.action_distribution_from_features(record.features, legal_actions)
            if not probs:
                continue
            scaled = _temperature_rescale_distribution(probs, temperature)
            p_true = max(1e-12, float(scaled.get(record.action, 0.0)))
            nll_sum += -math.log(p_true)
            evaluated += 1

        if evaluated <= 0:
            if verbose:
                elapsed = time.perf_counter() - candidate_start
                print('calibration temperature=%.2f skipped (evaluated=0, elapsed=%.2fs)' % (temperature, elapsed))
            continue
        nll = nll_sum / float(evaluated)
        if best_nll is None or nll < best_nll:
            best_nll = nll
            best_temperature = temperature
        if verbose:
            elapsed = time.perf_counter() - candidate_start
            print('calibration temperature=%.2f nll=%.6f evaluated=%d elapsed=%.2fs' % (temperature, nll, evaluated, elapsed))

    if verbose:
        total_elapsed = time.perf_counter() - sweep_start
        print('completed calibration sweep in %.2fs, selected_temperature=%.2f' % (total_elapsed, best_temperature))

    return float(best_temperature)


def load_policy_model(path):
    return runtime_model.load_policy_model(path)


def _iter_records(dataset_path, max_records=None):
    count = 0
    for record in JsonlTrajectoryReader(dataset_path):
        yield record
        count += 1
        if max_records is not None and count >= int(max_records):
            break


def _collect_training_records(dataset_path, max_records=None, decision_only=False):
    records = []
    kept = 0
    dropped_forced = 0
    limit = None if max_records is None else int(max_records)

    for record in JsonlTrajectoryReader(dataset_path):
        legal_actions = list(record.legal_actions or [])
        if record.action not in legal_actions:
            legal_actions.append(record.action)

        if decision_only and len(legal_actions) <= 1:
            dropped_forced += 1
            continue

        records.append(record)
        kept += 1
        if limit is not None and kept >= limit:
            break

    return records, dropped_forced


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
    ablate_encoder=False,
    ablate_features=False,
    ablate_belief=False,
    ablate_efficiency=False,
    ablate_search=False,
    device='auto',
):
    train_start = time.perf_counter()

    if verbose:
        print('starting train_cnn dataset=%s out=%s epochs=%d hidden_size=%d decision_only=%s device=%s max_records=%s'
              % (dataset_path, model_out_path, int(epochs), int(hidden_size), str(bool(decision_only)), str(device), str(max_records)))

    model_init_start = time.perf_counter()
    model_out_path = _resolve_model_out_path(model_out_path)
    codec = ActionCodec()
    model = CnnPolicyValueModel(
        action_space_size=codec.size,
        hidden_size=hidden_size,
        learning_rate=learning_rate,
        device=device,
    )
    if verbose:
        print('model initialized resolved_device=%s elapsed=%.2fs' % (model.resolved_device, time.perf_counter() - model_init_start))

    data_prep_start = time.perf_counter()
    if decision_only:
        records, dropped_forced = _collect_training_records(
            dataset_path,
            max_records=max_records,
            decision_only=True,
        )
    elif max_records is None:
        records = JsonlTrajectoryReader(dataset_path)
        dropped_forced = 0
    else:
        records = list(_iter_records(dataset_path, max_records=max_records))
        dropped_forced = 0
    if verbose:
        prepared_count = len(records) if isinstance(records, list) else 'stream'
        print('data preparation done records=%s dropped_forced=%d elapsed=%.2fs' % (str(prepared_count), int(dropped_forced), time.perf_counter() - data_prep_start))

    fit_start = time.perf_counter()
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
        aux_value_weight=0.15,
        efficiency_weight=0.0 if ablate_efficiency else 0.1,
        belief_consistency_weight=0.0 if ablate_belief else 0.1,
    )
    if verbose:
        print('model.fit completed elapsed=%.2fs samples=%d' % (time.perf_counter() - fit_start, int(stats.get('samples', 0))))

    calibration_start = time.perf_counter()
    selected_temperature = _select_calibration_temperature(model, dataset_path, max_records=max_records, verbose=verbose)
    if verbose:
        print('calibration selection completed elapsed=%.2fs selected_temperature=%.2f' % (time.perf_counter() - calibration_start, selected_temperature))

    model.set_calibration_temperature(selected_temperature)

    metadata_start = time.perf_counter()
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
            'promotion_metric': 'top1_masked_accuracy',
            'calibration_temperature': float(selected_temperature),
            'ablation': {
                'encoder': bool(ablate_encoder),
                'features': bool(ablate_features),
                'belief': bool(ablate_belief),
                'efficiency': bool(ablate_efficiency),
                'search': bool(ablate_search),
            },
            'dropped_forced_records': int(dropped_forced),
            'package_profile': package_profile(),
            'requested_device': model.requested_device,
            'resolved_device': model.resolved_device,
        }
    )
    if verbose:
        print('metadata update completed elapsed=%.2fs' % (time.perf_counter() - metadata_start))

    save_start = time.perf_counter()
    model.save(model_out_path)
    if verbose:
        print('model.save completed path=%s elapsed=%.2fs' % (model_out_path, time.perf_counter() - save_start))

    result = model.to_dict()
    result['training_stats'] = stats
    if verbose:
        print('train_cnn finished total_elapsed=%.2fs' % (time.perf_counter() - train_start))

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
    nll_sum = 0.0
    brier_sum = 0.0
    calibration_temperature = _safe_float(model.metadata.get('calibration_temperature', 1.0), 1.0)

    for record in _iter_records(dataset_path, max_records=max_records):
        legal_actions = list(record.legal_actions or [])
        if not legal_actions:
            continue

        if record.action not in legal_actions:
            legal_actions.append(record.action)

        probs = model.action_distribution_from_features(record.features, legal_actions)
        if not probs:
            continue

        probs = _temperature_rescale_distribution(probs, calibration_temperature)

        ranked = sorted(probs.items(), key=lambda item: (-item[1], item[0]))
        ranked_actions = [action for action, _ in ranked]

        total += 1
        for k in top_ks:
            if record.action in ranked_actions[:k]:
                hit_counts[k] += 1

        p_true = max(1e-12, float(probs.get(record.action, 0.0)))
        masked_xent_sum += -math.log(p_true)
        nll_sum += -math.log(p_true)

        top1 = ranked_actions[0]
        top1_conf = float(probs.get(top1, 0.0))
        top1_outcome = 1 if top1 == record.action else 0
        confidences.append(top1_conf)
        outcomes.append(top1_outcome)
        brier_sum += (top1_conf - float(top1_outcome)) * (top1_conf - float(top1_outcome))

        value_pred = model.estimate_value_from_features(record.features)
        value_true = _safe_float(record.reward, 0.0)
        diff = value_pred - value_true
        value_mse_sum += diff * diff

    metrics = {
        'total_evaluated': total,
        'topk_accuracy': {},
        'masked_cross_entropy': 0.0,
        'nll': 0.0,
        'value_mse': 0.0,
        'ece': _ece(confidences, outcomes),
        'brier': _brier(confidences, outcomes),
        'calibration_temperature': calibration_temperature,
    }

    for k in top_ks:
        if total == 0:
            metrics['topk_accuracy'][str(k)] = 0.0
        else:
            metrics['topk_accuracy'][str(k)] = float(hit_counts[k]) / float(total)

    if total > 0:
        metrics['masked_cross_entropy'] = masked_xent_sum / float(total)
        metrics['nll'] = nll_sum / float(total)
        metrics['value_mse'] = value_mse_sum / float(total)
        metrics['brier'] = brier_sum / float(total)

    return metrics


def choose_action_from_model(model, features, legal_actions, codec=None, belief_weight=0.0, efficiency_weight=0.0, temperature=None):
    if hasattr(model, 'choose_action_from_features'):
        try:
            return model.choose_action_from_features(
                features,
                legal_actions,
                belief_weight=belief_weight,
                efficiency_weight=efficiency_weight,
                temperature=temperature,
            )
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
        ablate_encoder=args.ablate_encoder,
        ablate_features=args.ablate_features,
        ablate_belief=args.ablate_belief,
        ablate_efficiency=args.ablate_efficiency,
        ablate_search=args.ablate_search,
        device=args.device,
    )
    summary = {
        'model_type': result.get('model_type'),
        'backend': result.get('backend'),
        'action_space_size': result.get('action_space_size'),
        'hidden_size': result.get('hidden_size'),
        'learning_rate': result.get('learning_rate'),
        'seed': result.get('seed'),
        'metadata': result.get('metadata', {}),
        'training_stats': result.get('training_stats', {}),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


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
    train_cnn_parser.add_argument('--ablate-encoder', action='store_true', help='Disable encoder-related upgrades (metadata/ablation only)')
    train_cnn_parser.add_argument('--ablate-features', action='store_true', help='Disable feature upgrades (metadata/ablation only)')
    train_cnn_parser.add_argument('--ablate-belief', action='store_true', help='Ablate belief-consistency loss terms')
    train_cnn_parser.add_argument('--ablate-efficiency', action='store_true', help='Ablate efficiency bonus usage in training')
    train_cnn_parser.add_argument('--ablate-search', action='store_true', help='Record search ablation setting in metadata')
    train_cnn_parser.add_argument('--device', default='auto', help='Torch device: cpu, cuda, cuda:0, or auto')
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
