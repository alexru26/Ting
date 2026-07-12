import json
import os
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'src'))

from dataset import (
    JsonlTrajectoryReader,
    JsonlTrajectoryWriter,
    TrajectoryRecord,
    create_mixed_split_manifest,
    ingest_external_jsonl,
    normalize_external_record,
)


class TestExternalDatasetIngest(unittest.TestCase):
    def test_normalize_external_record_maps_fields(self):
        payload = {
            'match_id': 'm-1',
            'turn': 3,
            'seat': 2,
            'request_type': 3,
            'request_action': 'PLAY',
            'chosen_action': 'PASS',
            'legalMoves': ['PASS', 'HU'],
            'terminal': 'true',
            'score_delta': 1.25,
            'state_features': {'meta': [1, 2, 3]},
        }
        record = normalize_external_record(payload, source_name='ijcai2026')

        self.assertIsNotNone(record)
        self.assertEqual(record.game_id, 'm-1')
        self.assertEqual(record.turn_index, 3)
        self.assertEqual(record.player_id, 2)
        self.assertEqual(record.action, 'PASS')
        self.assertEqual(record.legal_actions, ['PASS', 'HU'])
        self.assertTrue(record.done)
        self.assertAlmostEqual(record.reward, 1.25)
        self.assertEqual(record.features, {'meta': [1, 2, 3]})
        self.assertEqual(record.metadata.get('source_dataset'), 'ijcai2026')

    def test_ingest_external_jsonl_writes_trajectory_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw_path = os.path.join(tmp, 'external_raw.jsonl')
            out_path = os.path.join(tmp, 'external_norm.jsonl')

            rows = [
                {
                    'game_id': 'ext-1',
                    'turn_index': 0,
                    'player_id': 0,
                    'request_type': 2,
                    'request_action': 'DRAW',
                    'action': 'PLAY W1',
                    'legal_actions': ['PASS', 'PLAY W1'],
                    'features': {'meta': [0]},
                },
                {
                    'game_id': 'ext-2',
                    'turn_index': 1,
                    'player_id': 1,
                    'request_type': 3,
                    'request_action': 'PLAY',
                    'response': 'PASS',
                    'legalMoves': ['PASS', 'HU'],
                    'features': {'meta': [1]},
                },
            ]

            with open(raw_path, 'w', encoding='utf-8') as handle:
                for row in rows:
                    handle.write(json.dumps(row) + '\n')

            stats = ingest_external_jsonl(raw_path, out_path, source_name='ijcai2026')
            self.assertEqual(stats['written'], 2)
            self.assertEqual(stats['dropped'], 0)

            records = list(JsonlTrajectoryReader(out_path))
            self.assertEqual(len(records), 2)
            self.assertEqual(records[0].metadata.get('source_dataset'), 'ijcai2026')
            self.assertEqual(records[1].action, 'PASS')


class TestMixedSplitManifest(unittest.TestCase):
    def _write_dataset(self, path, game_ids):
        with JsonlTrajectoryWriter(path) as writer:
            idx = 0
            for gid in game_ids:
                writer.write(
                    TrajectoryRecord(
                        game_id=gid,
                        turn_index=idx,
                        player_id=0,
                        request_type=2,
                        request_action='DRAW',
                        action='PLAY W1',
                        legal_actions=['PASS', 'PLAY W1'],
                        features={'meta': [idx]},
                    )
                )
                idx += 1

    def test_create_mixed_split_manifest_is_reproducible(self):
        with tempfile.TemporaryDirectory() as tmp:
            local_path = os.path.join(tmp, 'local.jsonl')
            external_path = os.path.join(tmp, 'external.jsonl')
            manifest_a = os.path.join(tmp, 'manifest_a.json')
            manifest_b = os.path.join(tmp, 'manifest_b.json')

            self._write_dataset(local_path, ['l1', 'l2', 'l3', 'l4'])
            self._write_dataset(external_path, ['e1', 'e2', 'e3', 'e4'])

            out_a = create_mixed_split_manifest(local_path, external_path, manifest_a, seed=13)
            out_b = create_mixed_split_manifest(local_path, external_path, manifest_b, seed=13)

            self.assertEqual(out_a['splits'], out_b['splits'])

            all_local = []
            all_external = []
            for split_name in ['train', 'val', 'test']:
                all_local.extend(out_a['splits'][split_name]['local_game_ids'])
                all_external.extend(out_a['splits'][split_name]['external_game_ids'])

            self.assertEqual(sorted(all_local), ['l1', 'l2', 'l3', 'l4'])
            self.assertEqual(sorted(all_external), ['e1', 'e2', 'e3', 'e4'])


if __name__ == '__main__':
    unittest.main()