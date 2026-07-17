"""Tests for local_game.py CLI output behavior."""
import io
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'src'))
from local_game import run_games, render_tui_board

class TestLocalGameOutput(unittest.TestCase):

    def test_single_game_prints_final_state(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            run_games(n=1, seed=42)
        out = buf.getvalue()
        self.assertIn('Final game state:', out)
        for pid in range(4):
            self.assertIn(f'Player {pid}', out)
            self.assertIn('hand (', out)
            self.assertIn('discards (', out)

    def test_show_turns_prints_trace(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            run_games(n=1, seed=42, show_turns=True)
        out = buf.getvalue()
        self.assertIn('Turn-by-turn state trace:', out)
        self.assertIn('phase=after_deal', out)

    def test_render_tui_board_contains_sections(self):
        state = {'phase': 'turn_end_p0', 'wall_remaining': 70, 'players': [{'pid': 0, 'hand': ['W1', 'W2', 'W3'], 'flowers': ['H1'], 'packs': [('PENG', 'W9', 2)], 'discards': ['B1', 'B2']}, {'pid': 1, 'hand': ['B1'], 'flowers': [], 'packs': [], 'discards': []}, {'pid': 2, 'hand': ['T1'], 'flowers': [], 'packs': [], 'discards': []}, {'pid': 3, 'hand': ['J1'], 'flowers': [], 'packs': [], 'discards': []}]}
        buf = io.StringIO()
        with redirect_stdout(buf):
            render_tui_board(state, game_index=1, total_games=1, clear_screen=False)
        out = buf.getvalue()
        self.assertIn('Local Mahjong TUI', out)
        self.assertIn('open_calls', out)
        self.assertIn('flowers', out)
        self.assertIn('discards', out)

    def test_tui_mode_prints_board(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            run_games(n=1, seed=42, tui=True, tui_delay=0, no_clear=True)
        out = buf.getvalue()
        self.assertIn('Local Mahjong TUI', out)
        self.assertIn('Final result:', out)

    def test_dataset_export_shows_progress(self):
        path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jsonl') as handle:
                path = handle.name
            buf = io.StringIO()
            with redirect_stdout(buf):
                run_games(n=2, seed=42, export_dataset_path=path)
            out = buf.getvalue()
            self.assertIn('Generating dataset at', out)
            self.assertIn('2/2 games completed', out)
        finally:
            if path and os.path.exists(path):
                os.remove(path)
if __name__ == '__main__':
    unittest.main()
