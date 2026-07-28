import os
import sys
import tempfile
import unittest
ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'src'))

from model import CnnPolicyValueModel
from policy import NeuralPolicy, create_policy, load_model
from state import GameState


class _FakeModel:
    def __init__(self, action=None):
        self.action = action
        self.calls = 0

    def choose_action_from_features(self, _features, legal_actions):
        self.calls += 1
        return self.action if self.action is not None else legal_actions[0]


def _draw_state():
    state = GameState()
    state.my_id = 0
    state.hand = ['W1', 'W1', 'W2', 'W3', 'B4', 'B5', 'B6', 'T7', 'T8', 'T9', 'J1', 'J1', 'F1', 'F2']
    state.opponent_discards = {1: [], 2: [], 3: []}
    state.opponent_packs = {1: [], 2: [], 3: []}
    state.last_request_type = 2
    state.last_tile = 'F2'
    state.last_actor = 0
    return state


def _forced_state():
    state = GameState()
    state.my_id = 0
    state.hand = ['W1', 'W2', 'W3']
    state.last_request_type = 3
    state.last_request_action = 'DRAW'
    state.last_actor = 2
    return state


class TestNeuralPolicy(unittest.TestCase):

    def test_forced_turn_skips_model(self):
        fake = _FakeModel()
        policy = NeuralPolicy(_forced_state(), model=fake)
        self.assertEqual(policy.choose_action(), 'PASS')
        self.assertEqual(fake.calls, 0)

    def test_model_decision_is_returned(self):
        fake = _FakeModel()
        policy = NeuralPolicy(_draw_state(), model=fake)
        action = policy.choose_action()
        self.assertEqual(fake.calls, 1)
        self.assertIn(action, _draw_state().enumerate_legal_actions())

    def test_illegal_model_output_raises(self):
        fake = _FakeModel(action='PLAY B9')
        policy = NeuralPolicy(_draw_state(), model=fake)
        with self.assertRaises(ValueError):
            policy.choose_action()

    def test_missing_checkpoint_raises(self):
        with self.assertRaises(FileNotFoundError):
            load_model('/nonexistent/model.h5')

    def test_create_policy_uses_real_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'model.h5')
            CnnPolicyValueModel(channels=8, blocks=1, hidden_size=32).save(path)
            policy = create_policy(_draw_state(), model_path=path)
            action = policy.choose_action()
        self.assertIn(action, _draw_state().enumerate_legal_actions())

    def test_default_model_path_prefers_botzone_data_dir(self):
        from policy import default_model_path
        with tempfile.TemporaryDirectory() as tmp:
            os.makedirs(os.path.join(tmp, 'data'))
            data_path = os.path.join(tmp, 'data', 'model.h5')
            CnnPolicyValueModel(channels=8, blocks=1, hidden_size=32).save(data_path)
            old_cwd = os.getcwd()
            old_env = os.environ.pop('TING_POLICY_MODEL_PATH', None)
            try:
                os.chdir(tmp)
                self.assertEqual(default_model_path(), os.path.join('data', 'model.h5'))
            finally:
                os.chdir(old_cwd)
                if old_env is not None:
                    os.environ['TING_POLICY_MODEL_PATH'] = old_env


if __name__ == '__main__':
    unittest.main()
