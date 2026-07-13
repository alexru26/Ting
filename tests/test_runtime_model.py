import os
import sys
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'src'))

import imitation
import runtime_model


class _TypeErrorModel:
    def choose_action_from_features(self, features, legal_actions):
        return legal_actions[0] if legal_actions else None


class _BadModel:
    def choose_action_from_features(self, features, legal_actions, belief_weight=0.0):
        raise RuntimeError('boom')


class TestRuntimeModel(unittest.TestCase):

    def test_choose_action_from_model_falls_back_when_signature_omits_belief_weight(self):
        model = _TypeErrorModel()
        action = runtime_model.choose_action_from_model(model, {}, ['PASS'], belief_weight=0.5)
        self.assertEqual(action, 'PASS')

    def test_choose_action_from_model_returns_none_on_runtime_error(self):
        model = _BadModel()
        self.assertIsNone(runtime_model.choose_action_from_model(model, {}, ['PASS']))

    def test_imitation_load_policy_model_delegates_to_runtime_model(self):
        sentinel = object()
        with mock.patch('runtime_model.load_policy_model', return_value=sentinel) as load_mock:
            loaded = imitation.load_policy_model('checkpoint-path')
        self.assertIs(loaded, sentinel)
        load_mock.assert_called_once_with('checkpoint-path')


if __name__ == '__main__':
    unittest.main()
