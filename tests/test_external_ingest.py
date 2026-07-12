import json
import os
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'src'))

from external_ingest import build_opponent_registry, ingest_games_directory


class TestExternalIngest(unittest.TestCase):
    def _write_match(self, path):
        payload = {
            'match_id': 'm-1',
            'players': [
                {'seat': 0, 'name': 'A', 'target': True},
                {'seat': 1, 'name': 'B', 'target': True},
            ],
            'log': [
                {
                    'output': {
                        'command': 'request',
                        'content': {
                            '0': '2 0 W1',
                            '1': '3 0 PLAY W2',
                        },
                        'display': {'action': 'DRAW'},
                    }
                },
                {
                    '0': {'response': 'PLAY W1', 'verdict': 'OK'},
                    '1': {'response': 'PASS', 'verdict': 'OK'},
                },
            ],
        }
        with open(path, 'w', encoding='utf-8') as handle:
            json.dump(payload, handle)

    def test_ingest_games_directory_outputs_jsonl(self):
        with tempfile.TemporaryDirectory() as tmp:
            games_dir = os.path.join(tmp, 'games')
            os.makedirs(games_dir)
            self._write_match(os.path.join(games_dir, 'm1.json'))

            out_path = os.path.join(tmp, 'external_dataset.jsonl')
            stats = ingest_games_directory(games_dir, out_path)

            self.assertEqual(stats['files_seen'], 1)
            self.assertEqual(stats['files_ingested'], 1)
            self.assertGreaterEqual(stats['records_written'], 2)

            with open(out_path, 'r', encoding='utf-8') as handle:
                rows = [json.loads(line) for line in handle if line.strip()]

            self.assertGreaterEqual(len(rows), 2)
            self.assertEqual(rows[0]['game_id'], 'm-1')
            self.assertIn('source_dataset', rows[0]['metadata'])


class TestOpponentRegistry(unittest.TestCase):
    def test_build_opponent_registry_lists_model_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            models_dir = os.path.join(tmp, 'models')
            os.makedirs(models_dir)
            for name in ['imit_a.pkl', 'imit_b.pkl', 'ignore.txt']:
                with open(os.path.join(models_dir, name), 'w', encoding='utf-8') as handle:
                    handle.write('x')

            out_path = os.path.join(tmp, 'opponents.json')
            registry = build_opponent_registry(models_dir, out_path)

            self.assertEqual(registry['count'], 2)
            ids = [item['id'] for item in registry['opponents']]
            self.assertEqual(ids, ['imit_a', 'imit_b'])

            with open(out_path, 'r', encoding='utf-8') as handle:
                persisted = json.load(handle)
            self.assertEqual(persisted['count'], 2)


if __name__ == '__main__':
    unittest.main()