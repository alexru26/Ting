import ast
import os
import sys
import unittest
ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'src'))
from state import GameState

def _python_files():
    targets = []
    for folder in ('src', 'tests'):
        base = os.path.join(ROOT, folder)
        for dirpath, _dirnames, filenames in os.walk(base):
            for name in filenames:
                if name.endswith('.py'):
                    targets.append(os.path.join(dirpath, name))
    return sorted(targets)

class TestNoTypeAnnotations(unittest.TestCase):

    def test_no_type_annotations_in_python_files(self):
        offenders = []
        for path in _python_files():
            with open(path, 'r', encoding='utf-8') as handle:
                source = handle.read()
            tree = ast.parse(source, filename=path)
            for node in ast.walk(tree):
                if isinstance(node, ast.AnnAssign):
                    offenders.append(path)
                    break
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if node.returns is not None:
                        offenders.append(path)
                        break
                    args = []
                    args.extend(node.args.args)
                    args.extend(node.args.posonlyargs)
                    args.extend(node.args.kwonlyargs)
                    if node.args.vararg is not None:
                        args.append(node.args.vararg)
                    if node.args.kwarg is not None:
                        args.append(node.args.kwarg)
                    if any((arg.annotation is not None for arg in args)):
                        offenders.append(path)
                        break
            if 'from __future__ import ' + 'annotations' in source:
                offenders.append(path)
        self.assertEqual(sorted(set(offenders)), [])

class TestReadmeDatasetDocs(unittest.TestCase):

    def test_readme_contains_dataset_export_docs(self):
        readme_path = os.path.join(ROOT, 'README.md')
        with open(readme_path, 'r', encoding='utf-8') as handle:
            text = handle.read().lower()
        self.assertIn('dataset', text)
        self.assertIn('--export-dataset', text)
        self.assertIn('jsonl', text)


class TestGameStateParsingRobustness(unittest.TestCase):

    def test_parse_from_history_ignores_invalid_first_request_type(self):
        state = GameState()
        state.parse_from_history(['2 W1'], [])
        self.assertEqual(state.last_request_type, -1)
        self.assertEqual(state.hand, [])

    def test_parse_from_history_ignores_malformed_init_and_deal(self):
        state = GameState()
        state.parse_from_history(['0 0', '1 0'], [])
        self.assertEqual(state.last_request_type, -1)
        self.assertEqual(state.hand, [])

    def test_apply_type3_chi_ignores_invalid_mid_tile(self):
        state = GameState()
        state.my_id = 0
        state.hand = ['W1', 'W2', 'W4']
        state._last_discard = 'W3'
        state._last_discard_player = 3
        state._apply_type3(['3', '1', 'CHI', 'W0', 'W4'], None)
        self.assertEqual(state.opponent_packs.get(1, []), [])
        self.assertEqual(state._last_discard, 'W3')

    def test_set_current_request_ignores_malformed_payload(self):
        state = GameState()
        state._set_current_request('3')
        self.assertEqual(state.last_request_type, -1)

if __name__ == '__main__':
    unittest.main()
