import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'src'))

from scoring import calculate_fan, fan_calculator_available


class TestScoringRequirements(unittest.TestCase):

    def test_fan_calculator_is_required(self):
        self.assertTrue(fan_calculator_available())

    def test_calculate_fan_works_for_valid_hand(self):
        hand = (
            'W1', 'W1',
            'W2', 'W2',
            'W3', 'W3',
            'B1', 'B1',
            'B2', 'B2',
            'B3', 'B3',
            'T1',
        )
        fan = calculate_fan(
            (),
            hand,
            'T1',
            0,
            False,
            False,
            False,
            False,
            0,
            0,
        )
        self.assertIsInstance(fan, int)
        self.assertGreaterEqual(fan, 0)


if __name__ == '__main__':
    unittest.main()
