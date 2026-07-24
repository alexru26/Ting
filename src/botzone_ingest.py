"""Botzone match-log ingestion.

Replays the plain-text round records in `data/data.txt` / `data/sample.txt`
(format documented in `data/README.md`) and emits trajectory JSONL in the
exact schema produced by local self-play, so both sources mix freely in the
supervised pipeline.

Per decision point the loader reconstructs the acting player's `GameState`
(the information they actually had), enumerates legal actions, and labels:

- draw decisions with the logged PLAY / GANG (concealed) / BUGANG / HU;
- claim decisions after a discard for every player with a real choice:
  the claimant's action, an ignored player's declared HU/GANG, or PASS;
- rob-the-kong decisions after BUGANG.

Ignored PENG/CHI declarations are skipped (the log omits the follow-up
discard, so no complete action label exists). Records whose label is not in
our legal-action set (rare judge/calculator disagreements) are counted and
skipped unless --strict.

Rewards are the final round scores scaled by 1/REWARD_SCALE, with
`steps_from_end` stored per record for fan-backward credit decay at train
time.
"""

import argparse
import json
import multiprocessing
import os
import time

from dataset import JsonlTrajectoryWriter, TrajectoryRecord
from features import FeatureExtractor
from local_game import REWARD_SCALE
from state import GameState
from tiles import ALL_TILES

_CLAIM_VERBS = ('Chi', 'Peng', 'Gang', 'Hu')


def parse_rounds(path):
    """Yield round dicts: {match_id, quan, deals, events, fan, scores}."""
    current = None
    with open(path, 'r', encoding='utf-8') as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split()
            if parts[0] == 'Match':
                if current is not None:
                    yield current
                current = {
                    'match_id': parts[1] if len(parts) > 1 else 'unknown',
                    'quan': 0,
                    'deals': [None] * 4,
                    'events': [],
                    'fan': 0,
                    'scores': [0, 0, 0, 0],
                }
            elif current is None:
                continue
            elif parts[0] == 'Wind':
                current['quan'] = int(parts[1])
            elif parts[0] == 'Player' and len(parts) >= 3 and parts[2] == 'Deal':
                current['deals'][int(parts[1])] = parts[3:16]
            elif parts[0] == 'Player':
                current['events'].append(_parse_event_line(parts))
            elif parts[0] == 'Fan':
                current['fan'] = int(parts[1])
            elif parts[0] == 'Huang':
                current['fan'] = 0
            elif parts[0] == 'Score':
                current['scores'] = [int(value) for value in parts[1:5]]
    if current is not None:
        yield current


def _parse_event_line(parts):
    """Parse `Player N Verb [tile] [Ignore Player M Verb tile]...`."""
    event = {
        'player': int(parts[1]),
        'verb': parts[2],
        'tile': parts[3] if len(parts) > 3 and parts[3] != 'Ignore' else None,
        'ignored': [],
    }
    idx = 3 if event['tile'] is None else 4
    while idx < len(parts):
        if parts[idx] == 'Ignore' and idx + 3 < len(parts) + 1:
            ignored_player = int(parts[idx + 2])
            ignored_verb = parts[idx + 3] if idx + 3 < len(parts) else ''
            ignored_tile = parts[idx + 4] if idx + 4 < len(parts) else None
            event['ignored'].append(
                {'player': ignored_player, 'verb': ignored_verb.capitalize(), 'tile': ignored_tile}
            )
            idx += 5
        else:
            idx += 1
    return event


class _ReplayPlayer:
    def __init__(self, pid, deal):
        self.pid = pid
        self.hand = list(deal)
        self.packs = []
        self.discards = []


class _RoundReplayer:
    """Replays one logged round and emits trajectory records."""

    def __init__(self, round_data, extractor, source_name, stats):
        self.round = round_data
        self.extractor = extractor
        self.source_name = source_name
        self.stats = stats
        self.players = [_ReplayPlayer(pid, round_data['deals'][pid]) for pid in range(4)]
        self.quan = int(round_data['quan']) % 4
        self.last_discard = None
        self.last_discard_player = None
        self.records = []

    def game_state_for(self, pid, request_type, last_tile, last_actor, last_action):
        me = self.players[pid]
        gs = GameState()
        gs.my_id = pid
        gs.quan = self.quan
        gs.hand = me.hand[:]
        gs.packs = me.packs[:]
        gs.discards = me.discards[:]
        gs.flowers = 0
        for p in self.players:
            if p.pid == pid:
                continue
            packs_visible = []
            for ptype, ptile, poffer in p.packs:
                if ptype == 'GANG' and poffer == p.pid:
                    # Another player's concealed kong stays hidden.
                    packs_visible.append(('GANG', None, poffer))
                else:
                    packs_visible.append((ptype, ptile, poffer))
            gs.opponent_discards[p.pid] = p.discards[:]
            gs.opponent_packs[p.pid] = packs_visible
        for p in self.players:
            for tile in p.discards:
                gs.seen_tiles[tile] += 1
            for ptype, ptile, poffer in p.packs:
                concealed = ptype == 'GANG' and poffer == p.pid
                if concealed and p.pid != pid:
                    continue
                if ptype == 'PENG':
                    gs.seen_tiles[ptile] += 3
                elif ptype == 'GANG':
                    gs.seen_tiles[ptile] += 4
                elif ptype == 'CHI':
                    mid_val = int(ptile[1:])
                    for value in (mid_val - 1, mid_val, mid_val + 1):
                        gs.seen_tiles['%s%d' % (ptile[0], value)] += 1
        gs.last_request_type = request_type
        gs.last_tile = last_tile
        gs.last_actor = last_actor
        gs.last_request_action = last_action
        gs._last_discard = self.last_discard
        gs._last_discard_player = self.last_discard_player
        return gs

    def emit(self, pid, gs, action, force_decision=False):
        legal_actions = gs.enumerate_legal_actions()
        if len(legal_actions) <= 1 and not force_decision:
            return
        if action not in legal_actions:
            self.stats['label_not_legal'] += 1
            if self.stats.get('strict'):
                raise ValueError(
                    'Label %r not legal in match %s (legal: %r)'
                    % (action, self.round['match_id'], legal_actions)
                )
            return
        self.records.append(
            TrajectoryRecord(
                game_id='%s' % self.round['match_id'],
                turn_index=len(self.records),
                player_id=pid,
                request_type=gs.last_request_type,
                request_action=gs.last_request_action,
                action=action,
                legal_actions=legal_actions,
                reward=0.0,
                done=False,
                features=self.extractor.extract(gs),
                metadata={'source': self.source_name},
            )
        )

    def replay(self):
        events = self.round['events']
        index = 0
        while index < len(events):
            event = events[index]
            verb = event['verb']
            pid = event['player']
            player = self.players[pid]

            if verb == 'Draw':
                drawn = event['tile']
                player.hand.append(drawn)
                index = self._handle_draw_decision(events, index, pid, drawn)
            elif verb == 'Play':
                # A bare Play (not consumed by a draw/claim decision) should
                # not occur, but apply it defensively.
                self._apply_play(pid, event['tile'])
                index += 1
                index = self._handle_claims(events, index, pid, event['tile'])
            elif verb in ('Chi', 'Peng', 'Gang', 'Hu'):
                # Claims are consumed inside _handle_claims; reaching one here
                # means the log had no preceding discard - skip it.
                self.stats['orphan_claims'] += 1
                index += 1
            elif verb == 'BuGang':
                # Handled inside draw decisions; defensive skip.
                self.stats['orphan_claims'] += 1
                index += 1
            else:
                index += 1

        self._backfill_rewards()
        return self.records

    def _handle_draw_decision(self, events, index, pid, drawn):
        """Label and apply the actor's post-draw action. Returns next index."""
        player = self.players[pid]
        next_event = events[index + 1] if index + 1 < len(events) else None
        gs = self.game_state_for(pid, 2, drawn, pid, 'DRAW')

        if next_event is None or next_event['player'] != pid:
            return index + 1

        verb = next_event['verb']
        if verb == 'Play':
            self.emit(pid, gs, 'PLAY %s' % next_event['tile'], force_decision=True)
            self._apply_play(pid, next_event['tile'])
            return self._handle_claims(events, index + 2, pid, next_event['tile'])
        if verb == 'AnGang':
            tile = next_event['tile']
            self.emit(pid, gs, 'GANG %s' % tile, force_decision=True)
            for _ in range(4):
                player.hand.remove(tile)
            player.packs.append(('GANG', tile, pid))
            return index + 2
        if verb == 'BuGang':
            tile = next_event['tile']
            self.emit(pid, gs, 'BUGANG %s' % tile, force_decision=True)
            player.hand.remove(tile)
            for i, (ptype, ptile, poffer) in enumerate(player.packs):
                if ptype == 'PENG' and ptile == tile:
                    player.packs[i] = ('GANG', tile, poffer)
                    break
            return self._handle_rob_kong(events, index + 2, pid, tile)
        if verb == 'Hu':
            self.emit(pid, gs, 'HU', force_decision=True)
            return len(events)
        return index + 1

    def _handle_claims(self, events, index, discarder, tile):
        """Emit claim decisions for a discard, apply the claim. Returns next index."""
        next_event = events[index] if index < len(events) else None

        claimant = None
        claim_verb = None
        ignored = []
        if next_event is not None and next_event['verb'] in _CLAIM_VERBS and next_event['player'] != discarder:
            claimant = next_event['player']
            claim_verb = next_event['verb']
            ignored = next_event['ignored']

        ignored_by_player = {row['player']: row for row in ignored}
        claim_label = None
        consumed = index

        if claimant is not None:
            if claim_verb == 'Hu':
                claim_label = 'HU'
                consumed = index + 1
            elif claim_verb == 'Gang':
                claim_label = 'GANG'
                consumed = index + 1
            elif claim_verb == 'Peng':
                follow = events[index + 1] if index + 1 < len(events) else None
                if follow is not None and follow['player'] == claimant and follow['verb'] == 'Play':
                    claim_label = 'PENG %s' % follow['tile']
                    consumed = index + 2
                else:
                    self.stats['claims_without_discard'] += 1
                    claim_label = None
            elif claim_verb == 'Chi':
                follow = events[index + 1] if index + 1 < len(events) else None
                if follow is not None and follow['player'] == claimant and follow['verb'] == 'Play':
                    claim_label = 'CHI %s %s' % (next_event['tile'], follow['tile'])
                    consumed = index + 2
                else:
                    self.stats['claims_without_discard'] += 1
                    claim_label = None

        # Emit decisions for every non-discarder in claim order.
        for offset in range(1, 4):
            pid = (discarder + offset) % 4
            gs = self.game_state_for(pid, 3, tile, discarder, 'PLAY')
            if pid == claimant and claim_label is not None:
                self.emit(pid, gs, claim_label)
            elif pid in ignored_by_player:
                declared = ignored_by_player[pid]['verb']
                if declared == 'Hu':
                    self.emit(pid, gs, 'HU')
                elif declared == 'Gang':
                    self.emit(pid, gs, 'GANG')
                else:
                    self.stats['skipped_ignored_partial'] += 1
            else:
                self.emit(pid, gs, 'PASS')

        # Apply the claim to the replay state.
        if claimant is not None and claim_label is not None:
            claimer = self.players[claimant]
            if claim_verb == 'Hu':
                return len(events)
            if claim_verb == 'Gang':
                for _ in range(3):
                    claimer.hand.remove(tile)
                claimer.packs.append(('GANG', tile, discarder))
                self.last_discard = None
                self.last_discard_player = None
                return consumed
            if claim_verb == 'Peng':
                for _ in range(2):
                    claimer.hand.remove(tile)
                claimer.packs.append(('PENG', tile, discarder))
                self.last_discard = tile
                self.last_discard_player = discarder
                follow_tile = claim_label.split()[1]
                self._apply_play(claimant, follow_tile)
                return self._handle_claims(events, consumed, claimant, follow_tile)
            if claim_verb == 'Chi':
                mid = next_event['tile']
                mid_val = int(mid[1:])
                seq = ['%s%d' % (mid[0], mid_val - 1), mid, '%s%d' % (mid[0], mid_val + 1)]
                for seq_tile in seq:
                    if seq_tile != tile:
                        claimer.hand.remove(seq_tile)
                offer = seq.index(tile)
                claimer.packs.append(('CHI', mid, offer))
                self.last_discard = tile
                self.last_discard_player = discarder
                follow_tile = claim_label.split()[2]
                self._apply_play(claimant, follow_tile)
                return self._handle_claims(events, consumed, claimant, follow_tile)

        return index

    def _handle_rob_kong(self, events, index, ganger, tile):
        next_event = events[index] if index < len(events) else None
        robber = None
        ignored = []
        if next_event is not None and next_event['verb'] == 'Hu' and next_event['player'] != ganger:
            robber = next_event['player']
            ignored = next_event['ignored']
        ignored_hu = {row['player'] for row in ignored if row['verb'] == 'Hu'}

        for pid in range(4):
            if pid == ganger:
                continue
            gs = self.game_state_for(pid, 3, tile, ganger, 'BUGANG')
            if pid == robber or pid in ignored_hu:
                self.emit(pid, gs, 'HU')
            else:
                self.emit(pid, gs, 'PASS')

        if robber is not None:
            return len(events)
        return index

    def _apply_play(self, pid, tile):
        player = self.players[pid]
        if tile in player.hand:
            player.hand.remove(tile)
        else:
            self.stats['play_not_in_hand'] += 1
        player.discards.append(tile)
        self.last_discard = tile
        self.last_discard_player = pid

    def _backfill_rewards(self):
        scores = self.round['scores']
        per_player_records = {}
        for record in self.records:
            per_player_records.setdefault(record.player_id, []).append(record)
        for pid, rows in per_player_records.items():
            final_reward = float(scores[pid]) / REWARD_SCALE
            for position, record in enumerate(rows):
                record.reward = final_reward
                record.metadata['steps_from_end'] = len(rows) - 1 - position
            rows[-1].done = True


def _new_stats(strict=False):
    return {
        'rounds': 0,
        'records': 0,
        'label_not_legal': 0,
        'claims_without_discard': 0,
        'skipped_ignored_partial': 0,
        'orphan_claims': 0,
        'play_not_in_hand': 0,
        'round_failures': 0,
        'strict': bool(strict),
    }


def _replay_round_records(round_data, source_name, strict):
    stats = _new_stats(strict)
    extractor = FeatureExtractor()
    replayer = _RoundReplayer(round_data, extractor, source_name, stats)
    records = replayer.replay()
    stats['records'] = len(records)
    return [record.to_dict() for record in records], stats


def _worker(args):
    round_data, source_name, strict = args
    try:
        return _replay_round_records(round_data, source_name, strict)
    except Exception:
        if strict:
            raise
        stats = _new_stats(strict)
        stats['round_failures'] = 1
        return [], stats


def ingest_botzone_log(
    input_path,
    output_path,
    source_name='botzone',
    max_rounds=None,
    workers=1,
    strict=False,
    verbose=False,
):
    start = time.perf_counter()
    totals = _new_stats(strict)
    workers = max(1, int(workers))

    def _bounded_rounds():
        for round_index, round_data in enumerate(parse_rounds(input_path)):
            if max_rounds is not None and round_index >= int(max_rounds):
                return
            if any(deal is None for deal in round_data['deals']):
                continue
            if any(tile not in ALL_TILES for deal in round_data['deals'] for tile in deal):
                continue
            yield (round_data, source_name, strict)

    with JsonlTrajectoryWriter(output_path) as writer:
        if workers == 1:
            results = map(_worker, _bounded_rounds())
            _consume_results(results, writer, totals, verbose)
        else:
            with multiprocessing.Pool(processes=workers) as pool:
                results = pool.imap(_worker, _bounded_rounds(), chunksize=8)
                _consume_results(results, writer, totals, verbose)

    totals['strict'] = bool(strict)
    totals['elapsed_seconds'] = float(time.perf_counter() - start)
    totals['input_path'] = input_path
    totals['output_path'] = output_path
    return totals


def _consume_results(results, writer, totals, verbose):
    for record_dicts, stats in results:
        totals['rounds'] += 1
        for key in ('records', 'label_not_legal', 'claims_without_discard',
                    'skipped_ignored_partial', 'orphan_claims', 'play_not_in_hand',
                    'round_failures'):
            totals[key] += stats.get(key, 0)
        for payload in record_dicts:
            writer.write(TrajectoryRecord.from_dict(payload))
        if verbose and totals['rounds'] % 200 == 0:
            print(
                '\ringested rounds=%d records=%d skipped_labels=%d'
                % (totals['rounds'], totals['records'], totals['label_not_legal']),
                end='',
                flush=True,
            )
    if verbose:
        print('')


def main():
    parser = argparse.ArgumentParser(description='Ingest Botzone match logs into trajectory JSONL')
    parser.add_argument('--input', required=True, help='Botzone log path (e.g. data/data.txt)')
    parser.add_argument('--output', required=True, help='Output trajectory JSONL path')
    parser.add_argument('--source', default='botzone', help='Source name stored in record metadata')
    parser.add_argument('--max-rounds', type=int, default=None, help='Optional round cap')
    parser.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 2), help='Parallel replay workers')
    parser.add_argument('--strict', action='store_true', help='Fail on the first inconsistent record')
    parser.add_argument('--verbose', action='store_true', help='Print progress')
    args = parser.parse_args()

    stats = ingest_botzone_log(
        input_path=args.input,
        output_path=args.output,
        source_name=args.source,
        max_rounds=args.max_rounds,
        workers=args.workers,
        strict=args.strict,
        verbose=args.verbose,
    )
    print(json.dumps(stats, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
