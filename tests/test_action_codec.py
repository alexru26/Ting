import os
import sys
import unittest
ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'src'))
from action_codec import ActionCodec

class TestActionCodec(unittest.TestCase):

    def test_round_trip(self):
        codec = ActionCodec()
        actions = ['PASS', 'HU', 'GANG', 'PLAY W1', 'GANG B9', 'BUGANG T3', 'PENG F1', 'CHI W3 J1']
        for action in actions:
            action_id = codec.encode(action)
            decoded = codec.decode(action_id)
            self.assertEqual(decoded, action)

    def test_stable_ids_for_base_actions(self):
        codec = ActionCodec()
        self.assertEqual(codec.encode('PASS'), 0)
        self.assertEqual(codec.encode('HU'), 1)
        self.assertEqual(codec.decode(2), 'GANG')
if __name__ == '__main__':
    unittest.main()
