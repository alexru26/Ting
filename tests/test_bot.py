import io
import json
import os
import sys
import tempfile
import unittest
ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'src'))

from bot import MahjongBot
from model import CnnPolicyValueModel
from state import GameState

_DEAL = '1 0 0 0 0 W1 W2 W3 B4 B5 B6 T7 T8 T9 J1 J1 F1 F2'


class TestMahjongBot(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.model_path = os.path.join(cls.tmp.name, 'model.h5')
        CnnPolicyValueModel(channels=8, blocks=1, hidden_size=32).save(cls.model_path)
        os.environ['TING_POLICY_MODEL_PATH'] = cls.model_path

    @classmethod
    def tearDownClass(cls):
        os.environ.pop('TING_POLICY_MODEL_PATH', None)
        cls.tmp.cleanup()

    def test_init_and_deal_return_pass(self):
        bot = MahjongBot()
        self.assertEqual(bot.handle_input({'requests': ['0 0 0'], 'responses': []}), 'PASS')
        self.assertEqual(
            bot.handle_input({'requests': ['0 0 0', _DEAL], 'responses': ['PASS']}), 'PASS'
        )

    def test_draw_turn_returns_legal_action(self):
        bot = MahjongBot()
        payload = {'requests': ['0 0 0', _DEAL, '2 W4'], 'responses': ['PASS', 'PASS']}
        action = bot.handle_input(payload)
        state = GameState.from_history(payload['requests'], payload['responses'])
        self.assertIn(action, state.enumerate_legal_actions())
        self.assertNotEqual(action, 'PASS')

    def test_opponent_draw_returns_pass(self):
        bot = MahjongBot()
        payload = {'requests': ['0 0 0', _DEAL, '3 1 DRAW'], 'responses': ['PASS', 'PASS']}
        self.assertEqual(bot.handle_input(payload), 'PASS')

    def test_discard_response_is_legal(self):
        bot = MahjongBot()
        payload = {
            'requests': ['0 0 0', _DEAL, '3 3 DRAW', '3 3 PLAY W1'],
            'responses': ['PASS', 'PASS', 'PASS'],
        }
        action = bot.handle_input(payload)
        state = GameState.from_history(payload['requests'], payload['responses'])
        self.assertIn(action, state.enumerate_legal_actions())

    def test_empty_requests_raise(self):
        bot = MahjongBot()
        with self.assertRaises(ValueError):
            bot.handle_input({'requests': [], 'responses': []})

    def test_malformed_request_raises(self):
        bot = MahjongBot()
        with self.assertRaises(ValueError):
            bot.handle_input({'requests': ['garbage'], 'responses': []})

    def test_run_serves_full_history_then_incremental_lines(self):
        first = json.dumps(
            {'requests': ['0 0 0', _DEAL, '2 W4'], 'responses': ['PASS', 'PASS']}
        )
        second = '3 0 PLAY F1'
        stdin = io.StringIO(first + '\n' + second + '\n')
        stdout = io.StringIO()
        old_stdin, old_stdout = sys.stdin, sys.stdout
        try:
            sys.stdin, sys.stdout = stdin, stdout
            MahjongBot().run()
        finally:
            sys.stdin, sys.stdout = old_stdin, old_stdout
        lines = [line for line in stdout.getvalue().splitlines() if line]
        self.assertEqual(len(lines), 4)
        self.assertEqual(lines[1], '>>>BOTZONE_REQUEST_KEEP_RUNNING<<<')
        self.assertEqual(lines[3], '>>>BOTZONE_REQUEST_KEEP_RUNNING<<<')
        first_reply = json.loads(lines[0])
        self.assertTrue(first_reply['response'].startswith('PLAY '))
        self.assertEqual(json.loads(lines[2]), {'response': 'PASS'})


if __name__ == '__main__':
    unittest.main()
