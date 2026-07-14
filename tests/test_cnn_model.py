import os
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'src'))

from action_codec import ActionCodec
from dataset import JsonlTrajectoryWriter, TrajectoryRecord
from imitation import choose_action_from_model, evaluate_cnn, load_policy_model, train_cnn
from ml_packages import package_profile
from model import CnnPolicyValueModel


def _feature_bundle(marker):
    hand = [0] * 34
    seen = [0] * 34
    discard = [0] * 34
    pack = [0] * 34
    opponent_one = [0] * 34
    opponent_two = [0] * 34
    opponent_three = [0] * 34

    hand[marker] = 1
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
        'meta': [marker, 0, 0, 0, 2, 0, 1, 0],
        'request_type': 2,
        'seat': 0,
        'target_player': True,
    }


class TestCnnModel(unittest.TestCase):
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

    def test_train_eval_and_load_cnn(self):
        codec = ActionCodec()
        with tempfile.TemporaryDirectory() as tmp:
            dataset_path = os.path.join(tmp, 'train.jsonl')
            model_path = os.path.join(tmp, 'model.json')
            self._write_dataset(dataset_path)

            train_result = train_cnn(
                dataset_path=dataset_path,
                model_out_path=model_path,
                epochs=10,
                learning_rate=0.05,
                hidden_size=16,
            )
            self.assertEqual(train_result['model_type'], 'cnn_policy_value_v1')
            self.assertEqual(train_result['backend'], 'torch')
            self.assertEqual(train_result['package_profile']['preferred_numeric_backend'], 'torch')
            self.assertTrue(os.path.exists(model_path))

            metrics = evaluate_cnn(dataset_path, model_path, top_ks=[1, 3])
            self.assertEqual(metrics['total_evaluated'], 3)
            self.assertGreaterEqual(metrics['topk_accuracy']['1'], 2.0 / 3.0)
            self.assertGreaterEqual(metrics['masked_cross_entropy'], 0.0)
            self.assertGreaterEqual(metrics['nll'], 0.0)
            self.assertGreaterEqual(metrics['value_mse'], 0.0)
            self.assertGreaterEqual(metrics['ece'], 0.0)
            self.assertGreaterEqual(metrics['brier'], 0.0)

            model = load_policy_model(model_path)
            action = choose_action_from_model(
                model=model,
                features=_feature_bundle(0),
                legal_actions=['PASS', 'PLAY W1'],
                codec=codec,
            )
            self.assertIn(action, ['PASS', 'PLAY W1'])

    def test_belief_outputs_and_ablation_controls(self):
        codec = ActionCodec()
        model = CnnPolicyValueModel(
            action_space_size=codec.size,
            hidden_size=16,
            learning_rate=0.05,
        )
        features = _feature_bundle(3)

        info = model.policy_info_from_features(features, ['PASS', 'PLAY W1'])
        self.assertIn('belief_probs', info)
        self.assertIn('belief_entropy', info)
        self.assertEqual(len(info['belief_probs']), 34)
        self.assertGreaterEqual(info['belief_entropy'], 0.0)

        shaped = model.train_step(features, ['PASS', 'PLAY W1'], 'PLAY W1', 1.0)
        self.assertIn('belief_loss', shaped)
        self.assertGreaterEqual(shaped['belief_loss'], 0.0)
        self.assertIn('aux_value_loss', shaped)
        self.assertGreaterEqual(shaped['aux_value_loss'], 0.0)
        self.assertIn('belief_consistency_loss', shaped)
        self.assertGreaterEqual(shaped['belief_consistency_loss'], 0.0)
        self.assertIn('weighted_total_loss', shaped)
        self.assertGreaterEqual(shaped['weighted_total_loss'], 0.0)

        base_action = model.choose_action_from_features(features, ['PASS', 'PLAY W1'], belief_weight=0.0)
        belief_action = model.choose_action_from_features(features, ['PASS', 'PLAY W1'], belief_weight=3.0)
        self.assertIn(base_action, ['PASS', 'PLAY W1'])
        self.assertIn(belief_action, ['PASS', 'PLAY W1'])

    def test_conditioned_decode_and_legal_masking(self):
        codec = ActionCodec()
        model = CnnPolicyValueModel(
            action_space_size=codec.size,
            hidden_size=16,
            learning_rate=0.05,
        )
        features = _feature_bundle(6)
        legal_actions = ['PASS', 'PLAY W1']

        chosen = model.decode_conditioned_action_from_features(features, legal_actions=legal_actions, deterministic=True)
        self.assertIn(chosen, legal_actions)

        distribution = model.action_distribution_from_features(features, legal_actions)
        self.assertEqual(set(distribution.keys()), set(legal_actions))
        self.assertAlmostEqual(sum(distribution.values()), 1.0, places=5)

    def test_backward_compatible_load_without_belief_keys(self):
        codec = ActionCodec()
        model = CnnPolicyValueModel(
            action_space_size=codec.size,
            hidden_size=16,
            learning_rate=0.05,
        )
        stripped = {key: value for key, value in model.state_dict().items() if 'belief' not in key}
        restored = CnnPolicyValueModel(
            action_space_size=codec.size,
            hidden_size=16,
            learning_rate=0.05,
        )
        restored.load_state_dict(stripped)
        info = restored.policy_info_from_features(_feature_bundle(4), ['PASS', 'PLAY W1'])
        self.assertIn('belief_probs', info)


if __name__ == '__main__':
    unittest.main()