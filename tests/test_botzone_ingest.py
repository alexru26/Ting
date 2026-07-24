import os
import sys
import tempfile
import unittest
ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'src'))

from botzone_ingest import ingest_botzone_log, parse_rounds, _parse_event_line
from dataset import JsonlTrajectoryReader

_SAMPLE = os.path.join(ROOT, 'data', 'sample.txt')


class TestLogParsing(unittest.TestCase):

    def test_parse_event_line_with_ignores(self):
        event = _parse_event_line('Player 1 Hu B7 Ignore Player 0 Peng B7 Ignore Player 3 Chi B8'.split())
        self.assertEqual(event['player'], 1)
        self.assertEqual(event['verb'], 'Hu')
        self.assertEqual(event['tile'], 'B7')
        self.assertEqual(len(event['ignored']), 2)
        self.assertEqual(event['ignored'][0], {'player': 0, 'verb': 'Peng', 'tile': 'B7'})
        self.assertEqual(event['ignored'][1], {'player': 3, 'verb': 'Chi', 'tile': 'B8'})

    def test_parse_gang_event_without_tile(self):
        event = _parse_event_line('Player 2 Gang T5'.split())
        self.assertEqual(event['verb'], 'Gang')
        self.assertEqual(event['tile'], 'T5')

    @unittest.skipUnless(os.path.exists(_SAMPLE), 'sample.txt not available')
    def test_parse_rounds_from_sample(self):
        rounds = list(parse_rounds(_SAMPLE))
        self.assertEqual(len(rounds), 16)
        first = rounds[0]
        self.assertEqual(len(first['deals']), 4)
        self.assertTrue(all(len(deal) == 13 for deal in first['deals']))
        self.assertEqual(len(first['scores']), 4)
        self.assertGreater(len(first['events']), 10)


class TestIngestion(unittest.TestCase):

    @unittest.skipUnless(os.path.exists(_SAMPLE), 'sample.txt not available')
    def test_ingest_sample_rounds(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = os.path.join(tmp, 'traj.jsonl')
            stats = ingest_botzone_log(_SAMPLE, output, max_rounds=3, workers=1)
            self.assertEqual(stats['rounds'], 3)
            self.assertEqual(stats['label_not_legal'], 0)
            self.assertEqual(stats['round_failures'], 0)
            records = list(JsonlTrajectoryReader(output))
            self.assertEqual(len(records), stats['records'])
            games = set()
            for record in records:
                self.assertIn(record.action, record.legal_actions)
                self.assertEqual(record.features['schema_version'], 5)
                self.assertIn('steps_from_end', record.metadata)
                self.assertEqual(record.metadata['source'], 'botzone')
                games.add(record.game_id)
            self.assertEqual(len(games), 3)
            # Every player's final record is flagged done with 0 steps left.
            for game_id in games:
                rows = [r for r in records if r.game_id == game_id]
                for pid in {r.player_id for r in rows}:
                    player_rows = [r for r in rows if r.player_id == pid]
                    self.assertTrue(player_rows[-1].done)
                    self.assertEqual(player_rows[-1].metadata['steps_from_end'], 0)


if __name__ == '__main__':
    unittest.main()
