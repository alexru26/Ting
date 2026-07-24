"""Supervised training tools for the Ting Mahjong policy-value model.

Pipeline:
1. `preencode-cnn` - one streaming pass over trajectory JSONL that quantizes
   tile planes, encodes legal-action index tensors, and writes a compact
   .npz cache (v3: adds per-record game keys, steps-from-end, win flags).
2. `train-cnn` - batched training over one or more caches (auto-built when
   missing or stale). Supports data mixing (`--dataset path:weight`,
   repeatable), match-level train/validation splits, fan-backward credit
   decay for value targets, and outcome-sensitive sample weighting for the
   masked legal-action cross-entropy.
3. `eval-cnn` - batched evaluation: top-k masked accuracy, masked CE,
   value MSE, win-head accuracy, and ECE calibration, reported separately
   for decision and forced states.

Oracle planes are never stored in caches (they are all zero outside the RL
simulator); the model pads them back with zeros at batch time.
"""

import argparse
import hashlib
import json
import os
import time

import numpy as np

from dataset import JsonlTrajectoryReader
from features import FEATURE_SCHEMA_VERSION, META_COUNT, ORACLE_PLANE_COUNT, PLANE_COUNT
from model import CnnPolicyValueModel, encode_action, _NONE_INDEX

CACHE_FORMAT_VERSION = 3
BASE_PLANE_COUNT = PLANE_COUNT - ORACLE_PLANE_COUNT

DEFAULT_CREDIT_GAMMA = 0.97
DEFAULT_OUTCOME_SCALE = 1.0
DEFAULT_DRAW_WEIGHT = 0.7


def _print_progress(prefix, count):
    print('\r%s %d records' % (prefix, count), end='', flush=True)


def default_cache_path(dataset_path):
    return dataset_path + '.cache.npz'


def _game_key(game_id):
    digest = hashlib.blake2b(str(game_id).encode('utf-8'), digest_size=8).digest()
    return int.from_bytes(digest, 'big', signed=False)


def preencode_cnn(dataset_path, cache_out_path, max_records=None, verbose=False):
    """Single streaming pass: JSONL records -> compact tensor cache."""
    start = time.perf_counter()
    limit = None if max_records is None else int(max_records)

    planes_rows = []
    meta_rows = []
    legal_rows = []
    target_rows = []
    reward_rows = []
    game_keys = []
    steps_rows = []

    for record in JsonlTrajectoryReader(dataset_path):
        features = record.features
        version = features.get('schema_version') if isinstance(features, dict) else None
        if version != FEATURE_SCHEMA_VERSION:
            raise ValueError(
                'Record %d has feature schema %r; expected v%d. Regenerate the dataset.'
                % (len(target_rows), version, FEATURE_SCHEMA_VERSION)
            )

        legal_actions = list(record.legal_actions)
        if record.action not in legal_actions:
            raise ValueError(
                'Record %d action %r not in its legal actions.' % (len(target_rows), record.action)
            )

        planes = np.asarray(features['tile_planes'], dtype=np.float32)[:BASE_PLANE_COUNT]
        planes_rows.append(np.round(planes * 4.0).astype(np.uint8))
        meta_rows.append(np.asarray(features['meta'], dtype=np.float32))
        legal_rows.append([encode_action(action) for action in legal_actions])
        target_rows.append(legal_actions.index(record.action))
        reward_rows.append(float(record.reward))
        game_keys.append(_game_key(record.game_id))
        steps_rows.append(int(record.metadata.get('steps_from_end', 0)))

        if verbose and len(target_rows) % 5000 == 0:
            _print_progress('preencode', len(target_rows))
        if limit is not None and len(target_rows) >= limit:
            break

    count = len(target_rows)
    if count <= 0:
        raise ValueError('No records available to pre-encode from %s' % dataset_path)
    if verbose:
        _print_progress('preencode', count)
        print('')

    width = max(len(row) for row in legal_rows)
    legal_family = np.zeros((count, width), dtype=np.uint8)
    legal_arg1 = np.full((count, width), _NONE_INDEX, dtype=np.uint8)
    legal_arg2 = np.full((count, width), _NONE_INDEX, dtype=np.uint8)
    legal_len = np.zeros((count,), dtype=np.int16)
    for row_idx, row in enumerate(legal_rows):
        legal_len[row_idx] = len(row)
        for col, (fam, a1, a2) in enumerate(row):
            legal_family[row_idx, col] = fam
            legal_arg1[row_idx, col] = a1
            legal_arg2[row_idx, col] = a2

    reward = np.asarray(reward_rows, dtype=np.float32)
    payload = {
        'tile_planes_q': np.stack(planes_rows),
        'meta': np.stack(meta_rows),
        'legal_family': legal_family,
        'legal_arg1': legal_arg1,
        'legal_arg2': legal_arg2,
        'legal_len': legal_len,
        'target_index': np.asarray(target_rows, dtype=np.int16),
        'reward': reward,
        'win_flag': (reward > 0.0).astype(np.uint8),
        'game_key': np.asarray(game_keys, dtype=np.uint64),
        'steps_from_end': np.asarray(steps_rows, dtype=np.int16),
        'cache_format': np.asarray(
            [CACHE_FORMAT_VERSION, FEATURE_SCHEMA_VERSION, BASE_PLANE_COUNT, META_COUNT],
            dtype=np.int64,
        ),
    }

    cache_dir = os.path.dirname(cache_out_path)
    if cache_dir and not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    temp_output = cache_out_path + '.tmp.npz'
    np.savez(temp_output, **payload)
    os.replace(temp_output, cache_out_path)

    return {
        'cache_out_path': cache_out_path,
        'dataset_path': dataset_path,
        'record_count': int(count),
        'legal_width': int(width),
        'elapsed_seconds': float(time.perf_counter() - start),
    }


def load_cache(cache_path):
    with np.load(cache_path) as handle:
        payload = {key: handle[key] for key in handle.files}
    header = payload.get('cache_format')
    if header is None or int(header[0]) != CACHE_FORMAT_VERSION:
        raise ValueError('Unsupported cache format in %s; re-run preencode-cnn.' % cache_path)
    if int(header[1]) != FEATURE_SCHEMA_VERSION or int(header[2]) != BASE_PLANE_COUNT or int(header[3]) != META_COUNT:
        raise ValueError('Cache %s was built for a different feature contract; re-run preencode-cnn.' % cache_path)
    return payload


def ensure_cache(dataset_path, cache_path=None, max_records=None, verbose=False):
    """Return a loaded cache, building it when missing or older than the dataset."""
    cache_path = cache_path or default_cache_path(dataset_path)
    needs_build = not os.path.exists(cache_path)
    if not needs_build and os.path.exists(dataset_path):
        needs_build = os.path.getmtime(cache_path) < os.path.getmtime(dataset_path)
    if not needs_build:
        try:
            return load_cache(cache_path)
        except ValueError:
            needs_build = True
    summary = preencode_cnn(dataset_path, cache_path, max_records=max_records, verbose=verbose)
    if verbose:
        print('built cache %s records=%d elapsed=%.2fs' % (cache_path, summary['record_count'], summary['elapsed_seconds']))
    return load_cache(cache_path)


def parse_dataset_spec(spec):
    """'path[:weight]' -> (path, weight). Windows drive letters are not a concern here."""
    if ':' in spec:
        path, _, raw_weight = spec.rpartition(':')
        try:
            return path, float(raw_weight)
        except ValueError:
            return spec, 1.0
    return spec, 1.0


def merge_caches(cache_payloads, source_weights):
    """Concatenate caches (padding legal widths) and attach per-record source weights."""
    if not cache_payloads:
        raise ValueError('No caches to merge.')
    width = max(int(payload['legal_family'].shape[1]) for payload in cache_payloads)

    def _pad(array, fill):
        if array.shape[1] == width:
            return array
        padded = np.full((array.shape[0], width), fill, dtype=array.dtype)
        padded[:, : array.shape[1]] = array
        return padded

    merged = {
        'tile_planes_q': np.concatenate([p['tile_planes_q'] for p in cache_payloads]),
        'meta': np.concatenate([p['meta'] for p in cache_payloads]),
        'legal_family': np.concatenate([_pad(p['legal_family'], 0) for p in cache_payloads]),
        'legal_arg1': np.concatenate([_pad(p['legal_arg1'], _NONE_INDEX) for p in cache_payloads]),
        'legal_arg2': np.concatenate([_pad(p['legal_arg2'], _NONE_INDEX) for p in cache_payloads]),
        'legal_len': np.concatenate([p['legal_len'] for p in cache_payloads]),
        'target_index': np.concatenate([p['target_index'] for p in cache_payloads]),
        'reward': np.concatenate([p['reward'] for p in cache_payloads]),
        'win_flag': np.concatenate([p['win_flag'] for p in cache_payloads]),
        'game_key': np.concatenate([p['game_key'] for p in cache_payloads]),
        'steps_from_end': np.concatenate([p['steps_from_end'] for p in cache_payloads]),
    }
    merged['source_weight'] = np.concatenate(
        [
            np.full((len(payload['target_index']),), float(weight), dtype=np.float32)
            for payload, weight in zip(cache_payloads, source_weights)
        ]
    )
    return merged


def apply_training_signals(preencoded, credit_gamma=DEFAULT_CREDIT_GAMMA, outcome_scale=DEFAULT_OUTCOME_SCALE, draw_weight=DEFAULT_DRAW_WEIGHT):
    """Derive decayed value targets and outcome-sensitive sample weights.

    Fan-backward credit (Tjong): decisions closer to the end of a trajectory
    carry more of the final outcome; return_t = reward * gamma^steps_from_end.
    Sample weighting: winning/high-scoring trajectories contribute more, and
    drawn games are down-weighted as low-information.
    """
    reward = preencoded['reward'].astype(np.float32)
    steps = preencoded['steps_from_end'].astype(np.float32)
    gamma = float(credit_gamma)
    decayed = reward * np.power(gamma, steps, dtype=np.float32)
    preencoded['return_target'] = decayed.astype(np.float32)

    weights = np.ones(len(reward), dtype=np.float32)
    weights *= np.where(reward == 0.0, float(draw_weight), 1.0).astype(np.float32)
    weights *= (1.0 + float(outcome_scale) * np.clip(decayed, 0.0, None)).astype(np.float32)
    if 'source_weight' in preencoded:
        weights *= preencoded['source_weight'].astype(np.float32)
    preencoded['sample_weight'] = weights
    return preencoded


def split_by_game(preencoded, train_ratio=0.9, seed=7):
    """Match-level split: all records of one game land on the same side."""
    game_keys = preencoded['game_key']
    unique_keys = np.unique(game_keys)
    rng = np.random.default_rng(int(seed))
    rng.shuffle(unique_keys)
    split = int(float(train_ratio) * float(len(unique_keys)))
    split = max(1, min(split, len(unique_keys) - 1)) if len(unique_keys) > 1 else 1
    train_keys = set(unique_keys[:split].tolist())
    indices = np.arange(len(game_keys), dtype=np.int64)
    train_mask = np.isin(game_keys, np.asarray(list(train_keys), dtype=np.uint64))
    return indices[train_mask], indices[~train_mask]


def train_cnn(
    dataset_specs,
    model_out_path,
    epochs=10,
    learning_rate=0.001,
    channels=32,
    blocks=3,
    hidden_size=256,
    max_records=None,
    verbose=False,
    policy_weight=1.0,
    value_weight=0.5,
    win_weight=0.2,
    credit_gamma=DEFAULT_CREDIT_GAMMA,
    outcome_scale=DEFAULT_OUTCOME_SCALE,
    draw_weight=DEFAULT_DRAW_WEIGHT,
    device='auto',
    early_stopping_patience=3,
    batch_size=512,
    seed=7,
):
    train_start = time.perf_counter()
    if isinstance(dataset_specs, str):
        dataset_specs = [dataset_specs]
    parsed_specs = [parse_dataset_spec(spec) for spec in dataset_specs]

    payloads = []
    weights = []
    for path, weight in parsed_specs:
        payloads.append(ensure_cache(path, max_records=max_records, verbose=verbose))
        weights.append(weight)
    preencoded = merge_caches(payloads, weights)
    preencoded = apply_training_signals(
        preencoded,
        credit_gamma=credit_gamma,
        outcome_scale=outcome_scale,
        draw_weight=draw_weight,
    )

    total = int(len(preencoded['target_index']))
    if max_records is not None and total > int(max_records):
        keep = np.arange(int(max_records), dtype=np.int64)
        for key, value in list(preencoded.items()):
            if isinstance(value, np.ndarray) and value.shape[:1] == (total,):
                preencoded[key] = value[keep]
        total = int(max_records)

    train_indices, validation_indices = split_by_game(preencoded, train_ratio=0.9, seed=seed)

    model = CnnPolicyValueModel(
        channels=channels,
        blocks=blocks,
        hidden_size=hidden_size,
        learning_rate=learning_rate,
        seed=seed,
        device=device,
    )
    if verbose:
        print(
            'train-cnn records=%d train=%d val=%d device=%s channels=%d blocks=%d hidden=%d sources=%s'
            % (
                total,
                len(train_indices),
                len(validation_indices),
                model.resolved_device,
                model.channels,
                model.blocks,
                model.hidden_size,
                ','.join('%s:%.2f' % spec for spec in parsed_specs),
            )
        )

    stats_total = {
        'samples': 0,
        'policy_loss': 0.0,
        'value_loss': 0.0,
        'win_loss': 0.0,
        'action_hits': 0,
        'decision_samples': 0,
        'decision_hits': 0,
    }
    best_metric = None
    best_state = None
    best_epoch = 0
    patience = max(0, int(early_stopping_patience))
    without_improvement = 0
    epochs_trained = 0

    for epoch_idx in range(max(1, int(epochs))):
        epoch_stats = model.fit_preencoded(
            preencoded,
            epochs=1,
            batch_size=batch_size,
            shuffle=True,
            sample_indices=train_indices,
            policy_weight=policy_weight,
            value_weight=value_weight,
            win_weight=win_weight,
            show_progress=bool(verbose),
        )
        for key in stats_total:
            stats_total[key] += epoch_stats[key]
        epochs_trained += 1

        if len(validation_indices) <= 0:
            continue
        val_metrics = model.evaluate_preencoded(preencoded, sample_indices=validation_indices)
        val_ce = val_metrics['masked_cross_entropy']
        if verbose:
            print(
                'epoch %d/%d val_masked_ce=%.6f val_top1=%.4f val_value_mse=%.6f val_ece=%.4f'
                % (
                    epoch_idx + 1,
                    int(epochs),
                    val_ce,
                    val_metrics['topk_accuracy'].get('1', 0.0),
                    val_metrics['value_mse'],
                    val_metrics['ece'],
                )
            )
        if val_metrics['decision_evaluated'] <= 0:
            continue
        if best_metric is None or val_ce < best_metric:
            best_metric = val_ce
            best_epoch = epoch_idx + 1
            best_state = model.state_dict()
            without_improvement = 0
        else:
            without_improvement += 1
            if patience > 0 and without_improvement >= patience:
                if verbose:
                    print('early stopping at epoch %d (best epoch %d)' % (epoch_idx + 1, best_epoch))
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.metadata.update(
        {
            'dataset_sources': ['%s:%s' % spec for spec in parsed_specs],
            'record_count': int(total),
            'train_record_count': int(len(train_indices)),
            'validation_record_count': int(len(validation_indices)),
            'split': 'by_game',
            'epochs_requested': int(epochs),
            'epochs_trained': int(epochs_trained),
            'best_epoch': int(best_epoch),
            'best_validation_masked_ce': None if best_metric is None else float(best_metric),
            'batch_size': int(batch_size),
            'policy_weight': float(policy_weight),
            'value_weight': float(value_weight),
            'win_weight': float(win_weight),
            'credit_gamma': float(credit_gamma),
            'outcome_scale': float(outcome_scale),
            'draw_weight': float(draw_weight),
            'learning_rate': float(learning_rate),
            'seed': int(seed),
        }
    )
    model.save(model_out_path)

    return {
        'model_out_path': model_out_path,
        'metadata': model.metadata,
        'training_stats': stats_total,
        'elapsed_seconds': float(time.perf_counter() - train_start),
    }


def evaluate_cnn(dataset_path, model_path, top_ks=(1, 3, 5), max_records=None, cache_path=None, device='cpu', credit_gamma=DEFAULT_CREDIT_GAMMA):
    model = CnnPolicyValueModel.load(model_path, device=device)
    preencoded = ensure_cache(dataset_path, cache_path=cache_path, max_records=max_records)
    preencoded = apply_training_signals(preencoded, credit_gamma=credit_gamma)
    total = int(len(preencoded['target_index']))
    if max_records is not None:
        total = min(total, int(max_records))
    indices = np.arange(total, dtype=np.int64)
    metrics = model.evaluate_preencoded(preencoded, sample_indices=indices, top_ks=top_ks)
    metrics['model_path'] = model_path
    metrics['dataset_path'] = dataset_path
    return metrics


def _parse_topk(raw_value):
    return [int(part) for part in raw_value.split(',') if part.strip()]


def _cmd_preencode_cnn(args):
    summary = preencode_cnn(
        dataset_path=args.dataset,
        cache_out_path=args.output or default_cache_path(args.dataset),
        max_records=args.max_records,
        verbose=args.verbose,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


def _cmd_train_cnn(args):
    result = train_cnn(
        dataset_specs=args.dataset,
        model_out_path=args.out,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        channels=args.channels,
        blocks=args.blocks,
        hidden_size=args.hidden_size,
        max_records=args.max_records,
        verbose=args.verbose,
        policy_weight=args.policy_weight,
        value_weight=args.value_weight,
        win_weight=args.win_weight,
        credit_gamma=args.credit_gamma,
        outcome_scale=args.outcome_scale,
        draw_weight=args.draw_weight,
        device=args.device,
        early_stopping_patience=args.early_stopping_patience,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


def _cmd_eval_cnn(args):
    metrics = evaluate_cnn(
        dataset_path=args.dataset,
        model_path=args.model,
        top_ks=_parse_topk(args.topk),
        max_records=args.max_records,
        cache_path=args.cache,
        device=args.device,
        credit_gamma=args.credit_gamma,
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))


def main():
    parser = argparse.ArgumentParser(description='Supervised training tools for the Ting Mahjong bot')
    sub = parser.add_subparsers(dest='command', required=True)

    preencode_parser = sub.add_parser('preencode-cnn', help='Pre-encode dataset tensors into an .npz cache')
    preencode_parser.add_argument('--dataset', required=True, help='Input JSONL trajectory dataset path')
    preencode_parser.add_argument('--output', default=None, help='Output .npz cache path (default: <dataset>.cache.npz)')
    preencode_parser.add_argument('--max-records', type=int, default=None, help='Optional cap for pre-encoding')
    preencode_parser.add_argument('--verbose', action='store_true', help='Print pre-encoding progress')
    preencode_parser.set_defaults(func=_cmd_preencode_cnn)

    train_parser = sub.add_parser('train-cnn', help='Train the policy-value model from trajectory datasets')
    train_parser.add_argument('--dataset', action='append', required=True, help='Dataset spec path[:weight]; repeat to mix sources')
    train_parser.add_argument('--out', default='src/model.h5', help='Output model path')
    train_parser.add_argument('--epochs', type=int, default=10, help='Maximum training epochs')
    train_parser.add_argument('--learning-rate', type=float, default=0.001, help='Adam learning rate')
    train_parser.add_argument('--channels', type=int, default=32, help='Residual trunk channels')
    train_parser.add_argument('--blocks', type=int, default=3, help='Residual block count')
    train_parser.add_argument('--hidden-size', type=int, default=256, help='Fused hidden width')
    train_parser.add_argument('--batch-size', type=int, default=512, help='Training batch size')
    train_parser.add_argument('--max-records', type=int, default=None, help='Optional record cap for smoke training')
    train_parser.add_argument('--policy-weight', type=float, default=1.0, help='Policy loss multiplier')
    train_parser.add_argument('--value-weight', type=float, default=0.5, help='Value loss multiplier')
    train_parser.add_argument('--win-weight', type=float, default=0.2, help='Auxiliary win-head loss multiplier')
    train_parser.add_argument('--credit-gamma', type=float, default=DEFAULT_CREDIT_GAMMA, help='Fan-backward credit decay per step from trajectory end')
    train_parser.add_argument('--outcome-scale', type=float, default=DEFAULT_OUTCOME_SCALE, help='Extra policy weight per unit of positive decayed return')
    train_parser.add_argument('--draw-weight', type=float, default=DEFAULT_DRAW_WEIGHT, help='Sample weight multiplier for drawn games')
    train_parser.add_argument('--early-stopping-patience', type=int, default=3, help='Stop when validation CE stalls for N epochs (0 disables)')
    train_parser.add_argument('--device', default='auto', help='Torch device: cpu, cuda, cuda:0, or auto')
    train_parser.add_argument('--seed', type=int, default=7, help='Training seed')
    train_parser.add_argument('--verbose', action='store_true', help='Print per-epoch metrics')
    train_parser.set_defaults(func=_cmd_train_cnn)

    eval_parser = sub.add_parser('eval-cnn', help='Evaluate a checkpoint on a JSONL trajectory dataset')
    eval_parser.add_argument('--dataset', required=True, help='Input JSONL trajectory dataset path')
    eval_parser.add_argument('--model', required=True, help='Model checkpoint path')
    eval_parser.add_argument('--topk', default='1,3,5', help='Comma-separated top-k list')
    eval_parser.add_argument('--max-records', type=int, default=None, help='Optional record cap')
    eval_parser.add_argument('--cache', default=None, help='Pre-encoded cache path (default: <dataset>.cache.npz, auto-built)')
    eval_parser.add_argument('--credit-gamma', type=float, default=DEFAULT_CREDIT_GAMMA, help='Credit decay used for value targets during eval')
    eval_parser.add_argument('--device', default='cpu', help='Torch device')
    eval_parser.set_defaults(func=_cmd_eval_cnn)

    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
