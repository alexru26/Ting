import ast
import os
import unittest
ROOT = os.path.dirname(os.path.dirname(__file__))

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
if __name__ == '__main__':
    unittest.main()
