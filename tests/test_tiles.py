import os
import sys
import unittest
ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'src'))

from tiles import is_numbered, tile_value


class TestTilesValidation(unittest.TestCase):

    def test_tile_value_accepts_valid_numbered_tiles(self):
        self.assertEqual(tile_value('W1'), 1)
        self.assertEqual(tile_value('B9'), 9)
        self.assertEqual(tile_value('T5'), 5)

    def test_tile_value_accepts_valid_honor_tiles(self):
        self.assertEqual(tile_value('F4'), 4)
        self.assertEqual(tile_value('J3'), 3)

    def test_tile_value_rejects_out_of_range_numbered_tiles(self):
        with self.assertRaises(ValueError):
            tile_value('W0')
        with self.assertRaises(ValueError):
            tile_value('B10')

    def test_tile_value_rejects_out_of_range_honor_tiles(self):
        with self.assertRaises(ValueError):
            tile_value('F0')
        with self.assertRaises(ValueError):
            tile_value('F5')
        with self.assertRaises(ValueError):
            tile_value('J0')
        with self.assertRaises(ValueError):
            tile_value('J4')

    def test_tile_value_rejects_invalid_suits(self):
        with self.assertRaises(ValueError):
            tile_value('X1')

    def test_is_numbered(self):
        self.assertTrue(is_numbered('W3'))
        self.assertFalse(is_numbered('J1'))

if __name__ == '__main__':
    unittest.main()
