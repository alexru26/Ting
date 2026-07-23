import os
import sys
import tempfile
import unittest
ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'src'))

import numpy as np
import torch

from features import FeatureExtractor
from model import CnnPolicyValueModel, encode_action, FAMILY_LABELS, NONE_TOKEN
from state import GameState


def _tiny_model():
    return CnnPolicyValueModel(channels=8, blocks=1, hidden_size=32, seed=3)


def _sample_state():
    state = GameState()
    state.my_id = 0
    state.hand = ['W1', 'W1', 'W2', 'W3', 'B4', 'B5', 'B6', 'T7', 'T8', 'T9', 'J1', 'J1', 'F1', 'F2']
    state.opponent_discards = {1: [], 2: [], 3: []}
    state.opponent_packs = {1: [], 2: [], 3: []}
    state.last_request_type = 2
    state.last_tile = 'F2'
    state.last_actor = 0
    return state


def _sample_features_and_actions():
    state = _sample_state()
    return FeatureExtractor().extract(state), state.enumerate_legal_actions()


def _build_preencoded(features, legal_actions, target_action, count=16, reward=0.25):
    model = _tiny_model()
    planes, meta = model.encode_features_batch([features] * count)
    family, arg1, arg2, mask = model.encode_legal_actions([list(legal_actions)] * count)
    return {
        'tile_planes_q': np.round(planes * 4.0).astype(np.uint8),
        'meta': meta,
        'legal_family': family.astype(np.uint8),
        'legal_arg1': arg1.astype(np.uint8),
        'legal_arg2': arg2.astype(np.uint8),
        'legal_len': mask.sum(axis=1).astype(np.int16),
        'target_index': np.full((count,), legal_actions.index(target_action), dtype=np.int16),
        'reward': np.full((count,), float(reward), dtype=np.float32),
    }


class TestActionEncoding(unittest.TestCase):

    def test_encode_action_families(self):
        for label in FAMILY_LABELS:
            family, _arg1, _arg2 = encode_action(label if label != 'CHI' else 'CHI W2 W4')
            self.assertEqual(family, FAMILY_LABELS.index(label))

    def test_encode_action_arguments(self):
        family, arg1, arg2 = encode_action('CHI W2 B9')
        self.assertEqual(FAMILY_LABELS[family], 'CHI')
        self.assertNotEqual(arg1, arg2)
        _family, none1, none2 = encode_action('PASS')
        self.assertEqual(none1, none2)

    def test_encode_action_rejects_unknown_family(self):
        with self.assertRaises(ValueError):
            encode_action('JUMP W1')


class TestInference(unittest.TestCase):

    def test_choose_action_is_legal_and_deterministic(self):
        features, legal_actions = _sample_features_and_actions()
        model = _tiny_model()
        first = model.choose_action_from_features(features, legal_actions)
        second = model.choose_action_from_features(features, legal_actions)
        self.assertIn(first, legal_actions)
        self.assertEqual(first, second)

    def test_policy_info_probabilities_sum_to_one(self):
        features, legal_actions = _sample_features_and_actions()
        model = _tiny_model()
        info = model.policy_info_from_features(features, legal_actions)
        self.assertAlmostEqual(sum(info['probabilities']), 1.0, places=4)
        self.assertEqual(len(info['actions']), len(legal_actions))

    def test_rejects_wrong_schema_features(self):
        model = _tiny_model()
        with self.assertRaises(ValueError):
            model.policy_info_from_features({'schema_version': 3}, ['PASS'])

    def test_rejects_empty_legal_actions(self):
        features, _legal = _sample_features_and_actions()
        model = _tiny_model()
        with self.assertRaises(ValueError):
            model.policy_info_from_features(features, [])


class TestTraining(unittest.TestCase):

    def test_supervised_training_learns_target(self):
        features, legal_actions = _sample_features_and_actions()
        target = legal_actions[2]
        preencoded = _build_preencoded(features, legal_actions, target, count=8)
        model = _tiny_model()
        for _ in range(30):
            model.fit_preencoded(preencoded, epochs=1, batch_size=8, shuffle=False)
        chosen = model.choose_action_from_features(features, legal_actions)
        self.assertEqual(chosen, target)

    def test_evaluate_preencoded_reports_metrics(self):
        features, legal_actions = _sample_features_and_actions()
        preencoded = _build_preencoded(features, legal_actions, legal_actions[0], count=8)
        model = _tiny_model()
        metrics = model.evaluate_preencoded(preencoded, top_ks=(1, 3))
        self.assertEqual(metrics['evaluated'], 8)
        self.assertEqual(metrics['decision_evaluated'], 8)
        self.assertIn('1', metrics['topk_accuracy'])
        self.assertGreaterEqual(metrics['masked_cross_entropy'], 0.0)

    def test_ppo_update_keeps_parameters_finite(self):
        features, legal_actions = _sample_features_and_actions()
        model = _tiny_model()
        info = model.policy_info_from_features(features, legal_actions)
        transitions = [
            {
                'features': features,
                'legal_actions': legal_actions,
                'action': legal_actions[idx % len(legal_actions)],
                'old_log_prob': info['log_probabilities'][idx % len(legal_actions)],
                'advantage': 0.5 if idx % 2 == 0 else -0.5,
                'return_target': 0.1,
            }
            for idx in range(6)
        ]
        stats = model.ppo_update(transitions, epochs=2, minibatch_size=3)
        self.assertGreater(stats['updates'], 0)
        for tensor in model.model.state_dict().values():
            self.assertTrue(bool(torch.isfinite(tensor).all()))


class TestPersistence(unittest.TestCase):

    def test_save_load_roundtrip(self):
        features, legal_actions = _sample_features_and_actions()
        model = _tiny_model()
        expected = model.choose_action_from_features(features, legal_actions)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'model.h5')
            model.save(path)
            loaded = CnnPolicyValueModel.load(path)
        self.assertEqual(loaded.channels, model.channels)
        self.assertEqual(loaded.choose_action_from_features(features, legal_actions), expected)

    def test_load_missing_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            CnnPolicyValueModel.load('/nonexistent/model.h5')

    def test_load_rejects_wrong_model_type(self):
        import h5py
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'legacy.h5')
            with h5py.File(path, 'w') as handle:
                handle.attrs['model_type'] = 'cnn_policy_value_v1'
            with self.assertRaises(ValueError):
                CnnPolicyValueModel.load(path)

    def test_load_state_dict_rejects_shape_mismatch(self):
        model = _tiny_model()
        state = model.state_dict()
        key = next(iter(state))
        state[key] = np.zeros((1, 1), dtype=np.float32)
        other = _tiny_model()
        with self.assertRaises(ValueError):
            other.load_state_dict(state)


if __name__ == '__main__':
    unittest.main()
