import json
import os
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'src'))

from action_codec import ActionCodec
from dataset import JsonlTrajectoryWriter, TrajectoryRecord
from imitation import (
    _resolve_model_out_path,
    choose_action_from_model,
    evaluate_cnn,
    load_policy_model,
    train_cnn,
)
from policy import NeuralPolicy
from state import GameState


def _feature_bundle(marker):
    hand = [0] * 34
    seen = [0] * 34
    discard = [0] * 34
    pack = [0] * 34
    opponent_one = [0] * 34
    opponent_two = [0] * 34
    opponent_three = [0] * 34

    hand[marker % 34] = 1
    seen[(marker + 1) % 34] = 1
    discard[(marker + 2) % 34] = 1
    pack[(marker + 3) % 34] = 1
    opponent_one[(marker + 4) % 34] = 1
    opponent_two[(marker + 5) % 34] = 1
    opponent_three[(marker + 6) % 34] = 1

    return {
        'hand_counts': hand,
        'seen_counts': seen,
        'self_discard_counts': discard,
        'pack_counts': pack,
        'opponent_discard_counts': [opponent_one, opponent_two, opponent_three],
        'meta': [0, 0, 0, 0, 2, 0, 1, 0],
        'request_type': 2,
        'seat': 0,
        'target_player': True,
    }


class TestImitationNeuralOnly(unittest.TestCase):
    def test_resolve_model_out_path_places_relative_paths_under_src(self):
        resolved = _resolve_model_out_path('model.h5')
        self.assertEqual(resolved, os.path.join(ROOT, 'src', 'model.h5'))

        nested = _resolve_model_out_path(os.path.join('models', 'model.h5'))
        self.assertEqual(nested, os.path.join(ROOT, 'src', 'models', 'model.h5'))

        already_under_src = _resolve_model_out_path(os.path.join('src', 'model.h5'))
        self.assertEqual(already_under_src, os.path.join(ROOT, 'src', 'model.h5'))

    def test_resolve_model_out_path_keeps_absolute_path(self):
        absolute = os.path.abspath(os.path.join(ROOT, 'tmp_model.h5'))
        self.assertEqual(_resolve_model_out_path(absolute), absolute)

    def _write_dataset(self, path):
        with JsonlTrajectoryWriter(path) as writer:
            writer.write(
                TrajectoryRecord(
                    game_id='g1',
                    turn_index=0,
                    player_id=0,
                    request_type=2,
                    request_action='DRAW',
                    action='PLAY W1',
                    legal_actions=['PASS', 'PLAY W1', 'PLAY W2'],
                    features=_feature_bundle(0),
                )
            )
            writer.write(
                TrajectoryRecord(
                    game_id='g2',
                    turn_index=0,
                    player_id=0,
                    request_type=2,
                    request_action='DRAW',
                    action='PLAY W2',
                    legal_actions=['PASS', 'PLAY W1', 'PLAY W2'],
                    features=_feature_bundle(1),
                )
            )
            writer.write(
                TrajectoryRecord(
                    game_id='g3',
                    turn_index=0,
                    player_id=0,
                    request_type=3,
                    request_action='PLAY',
                    action='PASS',
                    legal_actions=['PASS', 'HU'],
                    features=_feature_bundle(2),
                )
            )

    def _write_dataset_with_forced(self, path):
        with JsonlTrajectoryWriter(path) as writer:
            writer.write(
                TrajectoryRecord(
                    game_id='f1',
                    turn_index=0,
                    player_id=0,
                    request_type=2,
                    request_action='DRAW',
                    action='PLAY W1',
                    legal_actions=['PLAY W1'],
                    features=_feature_bundle(4),
                )
            )
            writer.write(
                TrajectoryRecord(
                    game_id='d1',
                    turn_index=1,
                    player_id=0,
                    request_type=2,
                    request_action='DRAW',
                    action='PLAY W2',
                    legal_actions=['PASS', 'PLAY W2'],
                    features=_feature_bundle(5),
                )
            )

    def _write_forced_only_dataset(self, path):
        with JsonlTrajectoryWriter(path) as writer:
            writer.write(
                TrajectoryRecord(
                    game_id='f1',
                    turn_index=0,
                    player_id=0,
                    request_type=2,
                    request_action='DRAW',
                    action='PLAY W1',
                    legal_actions=['PLAY W1'],
                    features=_feature_bundle(7),
                )
            )
            writer.write(
                TrajectoryRecord(
                    game_id='f2',
                    turn_index=1,
                    player_id=0,
                    request_type=2,
                    request_action='DRAW',
                    action='PLAY W2',
                    legal_actions=['PLAY W2'],
                    features=_feature_bundle(8),
                )
            )

    def test_train_eval_and_load_cnn(self):
        codec = ActionCodec()
        with tempfile.TemporaryDirectory() as tmp:
            dataset_path = os.path.join(tmp, 'train.jsonl')
            model_path = os.path.join(tmp, 'model.h5')
            self._write_dataset(dataset_path)

            train_result = train_cnn(
                dataset_path=dataset_path,
                model_out_path=model_path,
                epochs=5,
                learning_rate=0.05,
                hidden_size=16,
            )
            self.assertEqual(train_result['model_type'], 'cnn_policy_value_v1')
            self.assertTrue(os.path.exists(model_path))

            metrics = evaluate_cnn(dataset_path, model_path, top_ks=[1, 3])
            self.assertEqual(metrics['total_evaluated'], 3)
            self.assertGreaterEqual(metrics['topk_accuracy']['1'], 1.0 / 3.0)
            self.assertGreaterEqual(metrics['topk_accuracy']['3'], 1.0)
            self.assertGreaterEqual(metrics['nll'], 0.0)
            self.assertGreaterEqual(metrics['brier'], 0.0)
            self.assertGreater(metrics['calibration_temperature'], 0.0)

            model = load_policy_model(model_path)
            action = choose_action_from_model(
                model=model,
                features=_feature_bundle(0),
                legal_actions=['PASS', 'PLAY W1'],
                codec=codec,
            )
            self.assertIn(action, ['PASS', 'PLAY W1'])

    def test_load_policy_model_rejects_legacy_checkpoint_family(self):
        with tempfile.TemporaryDirectory() as tmp:
            legacy = os.path.join(tmp, 'legacy.json')
            with open(legacy, 'w', encoding='utf-8') as handle:
                json.dump({'model_type': 'frequency_lookup_v1'}, handle)

            with self.assertRaises(ValueError):
                load_policy_model(legacy)

    def test_neural_policy_uses_cnn_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset_path = os.path.join(tmp, 'train.jsonl')
            model_path = os.path.join(tmp, 'model.h5')
            self._write_dataset(dataset_path)
            train_cnn(dataset_path, model_path, epochs=3, learning_rate=0.05, hidden_size=16)

            state = GameState()
            state.my_id = 0
            state.hand = ['W1', 'W2', 'W3']
            state.last_request_type = 2
            state.last_request_action = 'DRAW'
            state.last_tile = 'W4'
            state.last_actor = 0

            policy = NeuralPolicy(state=state, model_path=model_path)
            self.assertIsNotNone(policy.model)
            action = policy.choose_action()
            self.assertTrue(state.is_legal_action(action))

    def test_train_cnn_decision_only_filters_forced_states(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset_path = os.path.join(tmp, 'forced_and_decision.jsonl')
            model_path = os.path.join(tmp, 'model.h5')
            self._write_dataset_with_forced(dataset_path)

            train_result = train_cnn(
                dataset_path=dataset_path,
                model_out_path=model_path,
                epochs=1,
                learning_rate=0.05,
                hidden_size=16,
                decision_only=True,
            )

            stats = train_result['training_stats']
            self.assertEqual(stats['samples'], 1)
            self.assertEqual(stats['decision_samples'], 1)
            self.assertEqual(stats['forced_samples'], 0)
            self.assertEqual(train_result['metadata']['dropped_forced_records'], 1)

    def test_train_cnn_persists_loss_weights_in_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset_path = os.path.join(tmp, 'train.jsonl')
            model_path = os.path.join(tmp, 'model.h5')
            self._write_dataset(dataset_path)

            train_result = train_cnn(
                dataset_path=dataset_path,
                model_out_path=model_path,
                epochs=1,
                policy_weight=1.25,
                value_weight=0.75,
                belief_weight=0.15,
                forced_policy_weight=0.0,
            )

            metadata = train_result['metadata']
            self.assertEqual(metadata['policy_weight'], 1.25)
            self.assertEqual(metadata['value_weight'], 0.75)
            self.assertEqual(metadata['belief_weight'], 0.15)
            self.assertEqual(metadata['forced_policy_weight'], 0.0)

    def test_train_cnn_records_ablation_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset_path = os.path.join(tmp, 'train.jsonl')
            model_path = os.path.join(tmp, 'model.h5')
            self._write_dataset(dataset_path)

            train_result = train_cnn(
                dataset_path=dataset_path,
                model_out_path=model_path,
                epochs=1,
                ablate_encoder=True,
                ablate_features=True,
                ablate_belief=True,
                ablate_efficiency=True,
                ablate_search=True,
            )

            ablation = train_result['metadata']['ablation']
            self.assertTrue(ablation['encoder'])
            self.assertTrue(ablation['features'])
            self.assertTrue(ablation['belief'])
            self.assertTrue(ablation['efficiency'])
            self.assertTrue(ablation['search'])

    def test_train_cnn_records_split_and_early_stopping_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset_path = os.path.join(tmp, 'train.jsonl')
            model_path = os.path.join(tmp, 'model.h5')
            self._write_dataset(dataset_path)

            train_result = train_cnn(
                dataset_path=dataset_path,
                model_out_path=model_path,
                epochs=3,
                early_stopping_patience=1,
            )

            metadata = train_result['metadata']
            self.assertEqual(metadata['train_split_ratio'], 0.8)
            self.assertGreaterEqual(metadata['train_record_count'], 1)
            self.assertGreaterEqual(metadata['validation_record_count'], 1)

            early_stopping = metadata['early_stopping']
            self.assertTrue(early_stopping['enabled'])
            self.assertEqual(early_stopping['patience'], 1)
            self.assertGreaterEqual(early_stopping['epochs_trained'], 1)

    def test_train_cnn_decision_only_falls_back_when_no_decision_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset_path = os.path.join(tmp, 'forced_only.jsonl')
            model_path = os.path.join(tmp, 'model.h5')
            self._write_forced_only_dataset(dataset_path)

            train_result = train_cnn(
                dataset_path=dataset_path,
                model_out_path=model_path,
                epochs=1,
                decision_only=True,
            )

            metadata = train_result['metadata']
            self.assertTrue(metadata['decision_only_requested'])
            self.assertFalse(metadata['decision_only'])
            self.assertTrue(metadata['decision_only_fallback_used'])
            self.assertGreater(train_result['training_stats']['samples'], 0)


if __name__ == '__main__':
    unittest.main()
