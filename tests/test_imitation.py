import json
import os
import sys
import tempfile
import time
import unittest
ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'src'))

import numpy as np

from dataset import JsonlTrajectoryWriter, TrajectoryRecord
from features import FeatureExtractor
from imitation import (
    apply_training_signals,
    default_cache_path,
    ensure_cache,
    evaluate_cnn,
    merge_caches,
    parse_dataset_spec,
    preencode_cnn,
    split_by_game,
    train_cnn,
)
from model import CnnPolicyValueModel
from state import GameState

_HANDS = [
    ['W1', 'W1', 'W2', 'W3', 'B4', 'B5', 'B6', 'T7', 'T8', 'T9', 'J1', 'J1', 'F1', 'F2'],
    ['W4', 'W5', 'W6', 'B1', 'B1', 'B2', 'T2', 'T3', 'T4', 'J2', 'J2', 'F3', 'F4', 'W9'],
    ['B7', 'B8', 'B9', 'T1', 'T1', 'T5', 'W2', 'W3', 'W4', 'J3', 'J3', 'F1', 'F4', 'B2'],
]


def _make_records(count=12, game_count=2):
    extractor = FeatureExtractor()
    records = []
    for index in range(count):
        state = GameState()
        state.my_id = index % 4
        state.hand = list(_HANDS[index % len(_HANDS)])
        state.opponent_discards = {pid: [] for pid in range(4) if pid != state.my_id}
        state.opponent_packs = {pid: [] for pid in range(4) if pid != state.my_id}
        state.last_request_type = 2
        state.last_tile = state.hand[-1]
        state.last_actor = state.my_id
        legal_actions = state.enumerate_legal_actions()
        records.append(
            TrajectoryRecord(
                game_id='game-%d' % (index % game_count),
                turn_index=index,
                player_id=state.my_id,
                request_type=2,
                request_action=None,
                action=legal_actions[0],
                legal_actions=legal_actions,
                reward=0.1 if index % 2 == 0 else 0.0,
                done=False,
                features=extractor.extract(state),
                metadata={'steps_from_end': count - 1 - index},
            )
        )
    return records


def _write_dataset(path, records):
    with JsonlTrajectoryWriter(path) as writer:
        for record in records:
            writer.write(record)


class TestPreencode(unittest.TestCase):

    def test_preencode_and_cache_shapes(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset_path = os.path.join(tmp, 'data.jsonl')
            _write_dataset(dataset_path, _make_records(8))
            cache_path = os.path.join(tmp, 'cache.npz')
            summary = preencode_cnn(dataset_path, cache_path)
            self.assertEqual(summary['record_count'], 8)
            payload = ensure_cache(dataset_path, cache_path=cache_path)
            self.assertEqual(len(payload['target_index']), 8)
            self.assertEqual(payload['tile_planes_q'].dtype.name, 'uint8')

    def test_preencode_rejects_old_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset_path = os.path.join(tmp, 'data.jsonl')
            record = _make_records(1)[0]
            record.features = dict(record.features)
            record.features['schema_version'] = 3
            _write_dataset(dataset_path, [record])
            with self.assertRaises(ValueError):
                preencode_cnn(dataset_path, os.path.join(tmp, 'cache.npz'))

    def test_ensure_cache_rebuilds_when_stale(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset_path = os.path.join(tmp, 'data.jsonl')
            _write_dataset(dataset_path, _make_records(4))
            cache_path = default_cache_path(dataset_path)
            ensure_cache(dataset_path)
            first_mtime = os.path.getmtime(cache_path)
            time.sleep(0.05)
            _write_dataset(dataset_path, _make_records(6))
            os.utime(dataset_path, (time.time() + 5, time.time() + 5))
            payload = ensure_cache(dataset_path)
            self.assertEqual(len(payload['target_index']), 6)
            self.assertGreaterEqual(os.path.getmtime(cache_path), first_mtime)


class TestTrainingSignals(unittest.TestCase):

    def test_parse_dataset_spec(self):
        self.assertEqual(parse_dataset_spec('data/a.jsonl'), ('data/a.jsonl', 1.0))
        self.assertEqual(parse_dataset_spec('data/a.jsonl:2.5'), ('data/a.jsonl', 2.5))

    def test_apply_training_signals_decay_and_weights(self):
        preencoded = {
            'reward': np.array([0.5, 0.5, 0.0], dtype=np.float32),
            'steps_from_end': np.array([0, 2, 0], dtype=np.int16),
        }
        result = apply_training_signals(preencoded, credit_gamma=0.9, outcome_scale=1.0, draw_weight=0.5)
        self.assertAlmostEqual(result['return_target'][0], 0.5, places=6)
        self.assertAlmostEqual(result['return_target'][1], 0.5 * 0.81, places=6)
        # Later winning decisions weigh more than early ones; draws are down-weighted.
        self.assertGreater(result['sample_weight'][0], result['sample_weight'][1])
        self.assertAlmostEqual(result['sample_weight'][2], 0.5, places=6)

    def test_split_by_game_keeps_games_intact(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset_path = os.path.join(tmp, 'data.jsonl')
            _write_dataset(dataset_path, _make_records(24, game_count=6))
            payload = ensure_cache(dataset_path)
            train_idx, val_idx = split_by_game(payload, train_ratio=0.75, seed=3)
            self.assertGreater(len(train_idx), 0)
            self.assertGreater(len(val_idx), 0)
            train_games = set(payload['game_key'][train_idx].tolist())
            val_games = set(payload['game_key'][val_idx].tolist())
            self.assertEqual(train_games & val_games, set())

    def test_merge_caches_mixes_sources_with_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            path_a = os.path.join(tmp, 'a.jsonl')
            path_b = os.path.join(tmp, 'b.jsonl')
            _write_dataset(path_a, _make_records(4))
            _write_dataset(path_b, _make_records(6))
            merged = merge_caches(
                [ensure_cache(path_a), ensure_cache(path_b)], [1.0, 3.0]
            )
            self.assertEqual(len(merged['target_index']), 10)
            self.assertAlmostEqual(float(merged['source_weight'][0]), 1.0)
            self.assertAlmostEqual(float(merged['source_weight'][-1]), 3.0)


class TestTrainAndEvaluate(unittest.TestCase):

    def test_train_eval_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset_path = os.path.join(tmp, 'data.jsonl')
            _write_dataset(dataset_path, _make_records(12))
            model_path = os.path.join(tmp, 'model.h5')
            result = train_cnn(
                dataset_path,
                model_path,
                epochs=1,
                channels=8,
                blocks=1,
                hidden_size=32,
                batch_size=4,
                device='cpu',
            )
            self.assertTrue(os.path.exists(model_path))
            self.assertEqual(result['metadata']['record_count'], 12)
            self.assertTrue(json.dumps(result))

            loaded = CnnPolicyValueModel.load(model_path)
            self.assertEqual(loaded.channels, 8)

            metrics = evaluate_cnn(dataset_path, model_path, top_ks=(1, 3))
            self.assertEqual(metrics['evaluated'], 12)
            self.assertIn('masked_cross_entropy', metrics)


if __name__ == '__main__':
    unittest.main()
