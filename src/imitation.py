import argparse
import json
import math
import os
import random
import shutil
import tempfile
import time

import numpy as np

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


def _print_progress_bar(prefix, current, total, width=32):
    total_value = max(1, int(total))
    current_value = max(0, min(int(current), total_value))
    ratio = float(current_value) / float(total_value)
    filled = int(float(width) * ratio)
    bar = ('#' * filled) + ('-' * (width - filled))
    print('\r%s [%s] %d/%d (%.1f%%)' % (prefix, bar, current_value, total_value, ratio * 100.0), end='', flush=True)
    if current_value >= total_value:
        print('')


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


def _is_decision_record(record):
    legal_actions = list(record.legal_actions or [])
    if record.action not in legal_actions:
        legal_actions.append(record.action)
    return len(legal_actions) > 1


def _collect_training_records(dataset_path, max_records=None, decision_only=False):
    records = []
    kept = 0
    dropped_forced = 0
    limit = None if max_records is None else int(max_records)

    for record in JsonlTrajectoryReader(dataset_path):
        if decision_only and not _is_decision_record(record):
            dropped_forced += 1
            continue

        records.append(record)
        kept += 1
        if limit is not None and kept >= limit:
            break

    return records, dropped_forced


def _split_records(records, train_ratio=0.8, seed=7):
    if not records:
        return [], []

    indices = list(range(len(records)))
    rng = random.Random(int(seed))
    rng.shuffle(indices)

    split_index = int(float(train_ratio) * float(len(indices)))
    if len(indices) > 1:
        split_index = max(1, min(split_index, len(indices) - 1))
    else:
        split_index = 1

    train_indices = set(indices[:split_index])
    train_records = []
    validation_records = []
    for idx, record in enumerate(records):
        if idx in train_indices:
            train_records.append(record)
        else:
            validation_records.append(record)
    return train_records, validation_records


def _evaluate_masked_cross_entropy(model, records, decision_only=False):
    nll_sum = 0.0
    evaluated = 0
    for record in records:
        if decision_only and not _is_decision_record(record):
            continue

        legal_actions = list(record.legal_actions or [])
        if not legal_actions:
            continue

        if record.action not in legal_actions:
            legal_actions.append(record.action)

        probs = model.action_distribution_from_features(record.features, legal_actions)
        if not probs:
            continue

        p_true = max(1e-12, float(probs.get(record.action, 0.0)))
        nll_sum += -math.log(p_true)
        evaluated += 1

    if evaluated <= 0:
        return None, 0
    return nll_sum / float(evaluated), int(evaluated)


def _empty_training_stats():
    return {
        'samples': 0,
        'action_loss': 0.0,
        'value_loss': 0.0,
        'aux_value_loss': 0.0,
        'belief_loss': 0.0,
        'belief_consistency_loss': 0.0,
        'efficiency_bonus': 0.0,
        'action_hits': 0,
        'weighted_total_loss': 0.0,
        'decision_samples': 0,
        'decision_hits': 0,
        'forced_samples': 0,
    }


def _merge_training_stats(accumulated, delta):
    merged = dict(accumulated)
    for key, value in delta.items():
        base = merged.get(key, 0)
        merged[key] = base + value
    return merged


def _split_index_array(total_count, train_ratio=0.8, seed=7):
    indices = np.arange(int(total_count), dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    rng.shuffle(indices)
    split_index = int(float(train_ratio) * float(total_count))
    if total_count > 1:
        split_index = max(1, min(split_index, total_count - 1))
    else:
        split_index = 1
    return indices[:split_index], indices[split_index:]


def _slice_preencoded(preencoded, indices):
    idx = np.asarray(indices, dtype=np.int64)
    return {
        'tile_tensor': preencoded['tile_tensor'][idx],
        'meta_tensor': preencoded['meta_tensor'][idx],
        'family_target': preencoded['family_target'][idx],
        'arg1_target': preencoded['arg1_target'][idx],
        'arg2_target': preencoded['arg2_target'][idx],
        'reward': preencoded['reward'][idx],
        'decision_mask': preencoded['decision_mask'][idx],
        'belief_target': preencoded['belief_target'][idx],
        'seen_full_mask': preencoded['seen_full_mask'][idx],
    }


def preencode_cnn(dataset_path, cache_out_path, max_records=None, decision_only=False, device='cpu', verbose=False):
    start = time.perf_counter()
    codec = ActionCodec()
    model = CnnPolicyValueModel(action_space_size=codec.size, hidden_size=32, learning_rate=0.001, device=device)

    limit = None if max_records is None else int(max_records)

    def _iter_filtered_records():
        kept_local = 0
        dropped_local = 0
        for record in JsonlTrajectoryReader(dataset_path):
            legal_actions = list(record.legal_actions or [])
            if record.action not in legal_actions:
                legal_actions.append(record.action)
            if decision_only and len(legal_actions) <= 1:
                dropped_local += 1
                continue
            yield record, legal_actions, dropped_local
            kept_local += 1
            if limit is not None and kept_local >= limit:
                break

    kept_count = 0
    dropped_forced = 0
    for _record, _legal_actions, dropped_marker in _iter_filtered_records():
        kept_count += 1
        dropped_forced = dropped_marker

    if kept_count <= 0:
        raise ValueError('No records available to pre-encode.')

    if verbose:
        print('preencode starting records=%d decision_only=%s' % (kept_count, str(bool(decision_only))))

    first_record = None
    first_legal_actions = None
    for record, legal_actions, _dropped in _iter_filtered_records():
        first_record = record
        first_legal_actions = legal_actions
        break
    if first_record is None or first_legal_actions is None:
        raise ValueError('No records available to pre-encode.')

    tile_probe, meta_probe = model._encode_features(first_record.features)
    tile_shape = tuple(tile_probe.squeeze(0).shape)
    meta_shape = tuple(meta_probe.squeeze(0).shape)
    belief_shape = tuple(np.asarray(model._belief_target_from_features(first_record.features), dtype=np.float32).shape)
    seen_shape = (len(model.arg_vocab) - 1,)

    cache_dir = os.path.dirname(cache_out_path)
    if cache_dir and not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    temp_root = cache_dir if cache_dir else os.getcwd()
    scratch_dir = tempfile.mkdtemp(prefix='preencode_', dir=temp_root)

    try:
        tile_path = os.path.join(scratch_dir, 'tile_tensor.dat')
        meta_path = os.path.join(scratch_dir, 'meta_tensor.dat')
        family_path = os.path.join(scratch_dir, 'family_target.dat')
        arg1_path = os.path.join(scratch_dir, 'arg1_target.dat')
        arg2_path = os.path.join(scratch_dir, 'arg2_target.dat')
        reward_path = os.path.join(scratch_dir, 'reward.dat')
        decision_path = os.path.join(scratch_dir, 'decision_mask.dat')
        belief_path = os.path.join(scratch_dir, 'belief_target.dat')
        seen_path = os.path.join(scratch_dir, 'seen_full_mask.dat')

        tile_tensor = np.memmap(tile_path, dtype=np.float32, mode='w+', shape=(kept_count,) + tile_shape)
        meta_tensor = np.memmap(meta_path, dtype=np.float32, mode='w+', shape=(kept_count,) + meta_shape)
        family_target = np.memmap(family_path, dtype=np.int64, mode='w+', shape=(kept_count,))
        arg1_target = np.memmap(arg1_path, dtype=np.int64, mode='w+', shape=(kept_count,))
        arg2_target = np.memmap(arg2_path, dtype=np.int64, mode='w+', shape=(kept_count,))
        reward = np.memmap(reward_path, dtype=np.float32, mode='w+', shape=(kept_count,))
        decision_mask = np.memmap(decision_path, dtype=np.uint8, mode='w+', shape=(kept_count,))
        belief_target = np.memmap(belief_path, dtype=np.float32, mode='w+', shape=(kept_count,) + belief_shape)
        seen_full_mask = np.memmap(seen_path, dtype=np.float32, mode='w+', shape=(kept_count,) + seen_shape)

        write_idx = 0
        for record, legal_actions, dropped_marker in _iter_filtered_records():
            is_decision = len(legal_actions) > 1
            tile_encoded, meta_encoded = model._encode_features(record.features)
            tile_tensor[write_idx] = tile_encoded.squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False)
            meta_tensor[write_idx] = meta_encoded.squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False)

            family_idx, arg1_idx, arg2_idx = model.encode_action_targets(record.action)
            family_target[write_idx] = family_idx
            arg1_target[write_idx] = arg1_idx
            arg2_target[write_idx] = arg2_idx
            reward[write_idx] = float(_safe_float(record.reward, 0.0))
            decision_mask[write_idx] = 1 if is_decision else 0
            belief_target[write_idx] = np.asarray(model._belief_target_from_features(record.features), dtype=np.float32)

            seen = record.features.get('seen_counts', []) if isinstance(record.features, dict) else []
            seen_vec = np.zeros(seen_shape, dtype=np.float32)
            if isinstance(seen, list):
                seen_limit = min(len(seen), seen_vec.shape[0])
                if seen_limit > 0:
                    seen_slice = np.asarray(seen[:seen_limit], dtype=np.float32)
                    seen_vec[:seen_limit] = (seen_slice >= 4.0).astype(np.float32)
            seen_full_mask[write_idx] = seen_vec

            write_idx += 1
            dropped_forced = dropped_marker
            if verbose and (write_idx == 1 or write_idx % 1000 == 0 or write_idx == kept_count):
                _print_progress_bar('preencode', write_idx, kept_count)

        tile_tensor.flush()
        meta_tensor.flush()
        family_target.flush()
        arg1_target.flush()
        arg2_target.flush()
        reward.flush()
        decision_mask.flush()
        belief_target.flush()
        seen_full_mask.flush()

        temp_output = cache_out_path + '.tmp.npz'
        np.savez(
            temp_output,
            tile_tensor=np.asarray(tile_tensor),
            meta_tensor=np.asarray(meta_tensor),
            family_target=np.asarray(family_target),
            arg1_target=np.asarray(arg1_target),
            arg2_target=np.asarray(arg2_target),
            reward=np.asarray(reward),
            decision_mask=np.asarray(decision_mask),
            belief_target=np.asarray(belief_target),
            seen_full_mask=np.asarray(seen_full_mask),
        )
        os.replace(temp_output, cache_out_path)

    finally:
        shutil.rmtree(scratch_dir, ignore_errors=True)

    summary = {
        'cache_out_path': cache_out_path,
        'dataset_path': dataset_path,
        'record_count': int(kept_count),
        'dropped_forced_records': int(dropped_forced),
        'decision_only': bool(decision_only),
        'elapsed_seconds': float(time.perf_counter() - start),
    }
    return summary


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
    early_stopping_patience=2,
    early_stopping_min_delta=0.0,
    batch_size=256,
    cache_path=None,
):
    train_start = time.perf_counter()

    if verbose:
        print('starting train_cnn dataset=%s out=%s epochs=%d hidden_size=%d decision_only=%s device=%s max_records=%s'
              % (dataset_path, model_out_path, int(epochs), int(hidden_size), str(bool(decision_only)), str(device), str(max_records)))
        print('training configuration batch_size=%d early_stopping_patience=%d early_stopping_min_delta=%.6f' % (int(batch_size), int(early_stopping_patience), float(early_stopping_min_delta)))

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
        print('model initialized resolved_device=%s amp_enabled=%s elapsed=%.2fs' % (model.resolved_device, str(bool(model.amp_enabled)), time.perf_counter() - model_init_start))

    data_prep_start = time.perf_counter()
    requested_decision_only = bool(decision_only)
    effective_decision_only = requested_decision_only
    decision_only_fallback_used = False
    using_preencoded_cache = bool(cache_path)

    preencoded = None
    dropped_forced = 0
    records = []

    if using_preencoded_cache:
        cache_load_start = time.perf_counter()
        with np.load(cache_path) as cache:
            preencoded = {key: cache[key] for key in cache.files}
        if verbose:
            print('loaded preencoded cache path=%s records=%d elapsed=%.2fs' % (cache_path, int(len(preencoded['family_target'])), time.perf_counter() - cache_load_start))
    else:
        records, dropped_forced = _collect_training_records(
            dataset_path,
            max_records=max_records,
            decision_only=effective_decision_only,
        )

    if (not using_preencoded_cache) and effective_decision_only and not records:
        decision_only_fallback_used = True
        effective_decision_only = False
        if verbose:
            print('decision-only filtering produced zero records; falling back to full-data training mode')
        records, _ = _collect_training_records(
            dataset_path,
            max_records=max_records,
            decision_only=False,
        )

    if verbose:
        if using_preencoded_cache:
            prepared_count = len(preencoded['family_target']) if preencoded is not None else 0
        else:
            prepared_count = len(records) if isinstance(records, list) else 'stream'
        print('data preparation done records=%s dropped_forced=%d elapsed=%.2fs' % (str(prepared_count), int(dropped_forced), time.perf_counter() - data_prep_start))

    split_start = time.perf_counter()
    split_seed = int(model.seed)
    train_records_list = []
    validation_records_list = []
    train_preencoded_indices = None
    validation_preencoded_indices = None
    if using_preencoded_cache:
        if preencoded is None:
            raise ValueError('Cache loading failed: preencoded payload is missing.')
        cache_total = int(len(preencoded['family_target']))
        if max_records is not None:
            cache_total = min(cache_total, int(max_records))
        train_preencoded_indices, validation_preencoded_indices = _split_index_array(cache_total, train_ratio=0.8, seed=split_seed)
        train_count = int(len(train_preencoded_indices))
        val_count = int(len(validation_preencoded_indices))
    else:
        train_records_list, validation_records_list = _split_records(records, train_ratio=0.8, seed=split_seed)
        train_count = len(train_records_list)
        val_count = len(validation_records_list)
    if verbose:
        print('dataset split done train_records=%d validation_records=%d split=80/20 elapsed=%.2fs' % (train_count, val_count, time.perf_counter() - split_start))

    if train_count <= 0:
        raise ValueError('No training records available after split. Check dataset size and filters.')

    fit_start = time.perf_counter()
    epoch_total = max(1, int(epochs))
    patience_value = max(0, int(early_stopping_patience))
    min_delta_value = max(0.0, float(early_stopping_min_delta))

    stats = _empty_training_stats()
    best_val_masked_cross_entropy = None
    best_epoch = 0
    best_state_dict = None
    epochs_without_improvement = 0
    stopped_early = False
    epochs_trained = 0
    validation_decision_only = False
    validation_decision_count = 0

    for epoch_idx in range(epoch_total):
        if verbose:
            print('early-stopping loop epoch %d/%d training_records=%d validation_records=%d' % (epoch_idx + 1, epoch_total, train_count, val_count))

        if using_preencoded_cache:
            if train_preencoded_indices is None:
                raise ValueError('Training pre-encoded split is missing.')
            epoch_stats = model.fit_preencoded(
                preencoded,
                epochs=1,
                batch_size=batch_size,
                shuffle=True,
                verbose=False,
                show_progress=bool(verbose),
                sample_indices=train_preencoded_indices,
                policy_weight=policy_weight,
                value_weight=value_weight,
                belief_weight=belief_weight,
                forced_policy_weight=forced_policy_weight,
                aux_value_weight=0.15,
                efficiency_weight=0.0 if ablate_efficiency else 0.1,
                belief_consistency_weight=0.0 if ablate_belief else 0.1,
            )
        else:
            epoch_stats = model.fit(
                train_records_list,
                epochs=1,
                max_records=None,
                shuffle=True,
                verbose=False,
                show_progress=bool(verbose),
                policy_weight=policy_weight,
                value_weight=value_weight,
                belief_weight=belief_weight,
                forced_policy_weight=forced_policy_weight,
                aux_value_weight=0.15,
                efficiency_weight=0.0 if ablate_efficiency else 0.1,
                belief_consistency_weight=0.0 if ablate_belief else 0.1,
                batch_size=batch_size,
            )
        stats = _merge_training_stats(stats, epoch_stats)
        epochs_trained += 1

        if using_preencoded_cache:
            if validation_preencoded_indices is None or len(validation_preencoded_indices) <= 0:
                continue
            cache_payload = preencoded
            assert cache_payload is not None
            decision_mask = np.asarray(cache_payload['decision_mask'][validation_preencoded_indices], dtype=bool)
            decision_indices = validation_preencoded_indices[decision_mask]
            if len(decision_indices) > 0:
                validation_decision_only = True
                validation_decision_count = int(len(decision_indices))
                val_masked_cross_entropy, val_evaluated = model.evaluate_preencoded_loss_for_indices(
                    cache_payload,
                    sample_indices=decision_indices,
                )
            else:
                validation_decision_only = False
                validation_decision_count = int(len(validation_preencoded_indices))
                val_masked_cross_entropy, val_evaluated = model.evaluate_preencoded_loss_for_indices(
                    cache_payload,
                    sample_indices=validation_preencoded_indices,
                )
        else:
            if not validation_records_list:
                continue
            decision_records = [record for record in validation_records_list if _is_decision_record(record)]
            if decision_records:
                validation_decision_only = True
                validation_decision_count = len(decision_records)
                val_masked_cross_entropy, val_evaluated = _evaluate_masked_cross_entropy(
                    model,
                    decision_records,
                    decision_only=True,
                )
            else:
                validation_decision_only = False
                validation_decision_count = len(validation_records_list)
                val_masked_cross_entropy, val_evaluated = _evaluate_masked_cross_entropy(model, validation_records_list)
        if val_masked_cross_entropy is None:
            if verbose:
                print('validation skipped for epoch %d/%d (no evaluable records)' % (epoch_idx + 1, epoch_total))
            continue

        if verbose:
            print('validation epoch %d/%d masked_cross_entropy=%.6f evaluated=%d' % (epoch_idx + 1, epoch_total, val_masked_cross_entropy, val_evaluated))

        improved = (
            best_val_masked_cross_entropy is None
            or val_masked_cross_entropy < (best_val_masked_cross_entropy - min_delta_value)
        )
        if improved:
            best_val_masked_cross_entropy = val_masked_cross_entropy
            best_epoch = epoch_idx + 1
            best_state_dict = model.state_dict()
            epochs_without_improvement = 0
            if verbose:
                print('new best validation metric at epoch %d/%d' % (epoch_idx + 1, epoch_total))
        else:
            epochs_without_improvement += 1
            if verbose:
                print('no validation improvement for %d epoch(s)' % epochs_without_improvement)
            if patience_value > 0 and epochs_without_improvement >= patience_value:
                stopped_early = True
                if verbose:
                    print('early stopping triggered at epoch %d/%d (patience=%d)' % (epoch_idx + 1, epoch_total, patience_value))
                break

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    if verbose:
        print('model.fit completed elapsed=%.2fs samples=%d epochs_trained=%d' % (time.perf_counter() - fit_start, int(stats.get('samples', 0)), int(epochs_trained)))

    if using_preencoded_cache:
        selected_temperature = 1.0
        if verbose:
            print('skipping calibration sweep for preencoded cache training; selected_temperature=1.00')
    else:
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
            'decision_only': bool(effective_decision_only),
            'decision_only_requested': bool(requested_decision_only),
            'decision_only_fallback_used': bool(decision_only_fallback_used),
            'train_record_count': int(train_count),
            'validation_record_count': int(val_count),
            'train_split_ratio': 0.8,
            'batch_size': int(batch_size),
            'policy_weight': float(policy_weight),
            'value_weight': float(value_weight),
            'belief_weight': float(belief_weight),
            'forced_policy_weight': float(forced_policy_weight),
            'promotion_metric': 'top1_masked_accuracy',
            'validation_decision_only': bool(validation_decision_only),
            'validation_decision_count': int(validation_decision_count),
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
            'mixed_precision_enabled': bool(model.amp_enabled),
            'torch_compile_enabled': bool(model.compile_enabled),
            'using_preencoded_cache': bool(using_preencoded_cache),
            'cache_path': cache_path,
            'early_stopping': {
                'enabled': True,
                'patience': int(patience_value),
                'min_delta': float(min_delta_value),
                'stopped_early': bool(stopped_early),
                'epochs_trained': int(epochs_trained),
                'best_epoch': int(best_epoch),
                'best_validation_masked_cross_entropy': None if best_val_masked_cross_entropy is None else float(best_val_masked_cross_entropy),
            },
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
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
        batch_size=args.batch_size,
        cache_path=args.cache,
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


def _cmd_preencode_cnn(args):
    summary = preencode_cnn(
        dataset_path=args.dataset,
        cache_out_path=args.output,
        max_records=args.max_records,
        decision_only=args.decision_only,
        device=args.device,
        verbose=args.verbose,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


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
    train_cnn_parser.add_argument('--early-stopping-patience', type=int, default=2, help='Stop if validation loss does not improve for N epochs')
    train_cnn_parser.add_argument('--early-stopping-min-delta', type=float, default=0.0, help='Minimum validation loss improvement required to reset patience')
    train_cnn_parser.add_argument('--batch-size', type=int, default=256, help='Supervised training batch size')
    train_cnn_parser.add_argument('--cache', default=None, help='Optional pre-encoded dataset cache (.npz)')
    train_cnn_parser.set_defaults(func=_cmd_train_cnn)

    preencode_parser = sub.add_parser('preencode-cnn', help='Pre-encode dataset tensors to cache for high-throughput training')
    preencode_parser.add_argument('--dataset', required=True, help='Input JSONL trajectory dataset path')
    preencode_parser.add_argument('--output', required=True, help='Output .npz cache path')
    preencode_parser.add_argument('--max-records', type=int, default=None, help='Optional cap for pre-encoding')
    preencode_parser.add_argument('--decision-only', action='store_true', help='Encode only decision states (legal_actions > 1)')
    preencode_parser.add_argument('--device', default='cpu', help='Device to use for feature encoding (usually cpu)')
    preencode_parser.add_argument('--verbose', action='store_true', help='Print pre-encoding progress')
    preencode_parser.set_defaults(func=_cmd_preencode_cnn)

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
