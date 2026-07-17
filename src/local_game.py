"""
local_game.py  Python game simulator for local testing.

Runs a complete 4-player Chinese Standard Mahjong round using the bot
policy directly (no subprocess / JSON I/O), so you can iterate quickly.

Usage
-----
    python local_game.py              # run one game, print scores
    python local_game.py --games 100  # run N games and print win statistics
"""
import argparse
import json
import os
import random
import time
import uuid
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, cast
from dataset import JsonlTrajectoryWriter, TrajectoryRecord
from features import FeatureExtractor
from scoring import calculate_fan
from state import GameState
from policy import GoalBasedPolicy, create_policy
from tiles import ALL_TILES, normalize_tiles
FLOWER_TILES = [f'H{i}' for i in range(1, 9)]

def _format_discards(discards, line_width=12):
    if not discards:
        return '-'
    lines = []
    for i in range(0, len(discards), line_width):
        lines.append(' '.join(discards[i:i + line_width]))
    return '\n'.join(lines)

def _format_packs(packs):
    if not packs:
        return '-'
    return ' | '.join([f'{ptype}:{tile}@{offer}' for ptype, tile, offer in packs])

def render_tui_board(state, game_index, total_games, clear_screen=True):
    typed_state = cast(Dict[str, Any], state)
    if clear_screen:
        print('\x1b[2J\x1b[H', end='')
    print('=' * 78)
    print(f"Local Mahjong TUI  game={game_index}/{total_games}  phase={typed_state['phase']}  wall_remaining={typed_state['wall_remaining']}")
    print('=' * 78)
    players = cast(List[Dict[str, Any]], typed_state['players'])
    for player_state in players:
        pid = player_state['pid']
        hand = ' '.join(player_state['hand']) if player_state['hand'] else '-'
        flowers = ' '.join(player_state['flowers']) if player_state['flowers'] else '-'
        packs = _format_packs(player_state['packs'])
        discards = _format_discards(player_state['discards'])
        print(f"P{pid} | hand({len(player_state['hand'])}): {hand}")
        print(f"   | flowers({len(player_state['flowers'])}): {flowers}")
        print(f'   | open_calls: {packs}')
        print(f"   | discards({len(player_state['discards'])}): {discards}")
        print('-' * 78)

def _build_wall():
    """144-tile wall: 4 copies of each of 34 tiles + 8 flower tiles."""
    wall = []
    for t in ALL_TILES:
        wall.extend([t] * 4)
    wall.extend(FLOWER_TILES)
    random.shuffle(wall)
    return wall


def _load_opponent_registry(registry_path):
    if not registry_path:
        return []
    if not os.path.exists(registry_path):
        return []

    with open(registry_path, 'r', encoding='utf-8') as handle:
        payload = json.load(handle)

    rows = payload.get('opponents', []) if isinstance(payload, dict) else []
    result = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        result.append(
            {
                'id': row.get('id') or row.get('file_name') or 'external-opponent',
                'kind': 'external',
                'path': row.get('path') or row.get('file_name'),
                'policy_mode': row.get('policy_mode', 'imitation'),
            }
        )
    return result


def _sample_opponent_rows(pool, seat_count, rng):
    seat_count = max(0, int(seat_count))
    rows = list(pool or [])
    if not rows:
        rows = [{'id': 'baseline-rule', 'kind': 'baseline', 'policy_mode': 'rule', 'path': None}]

    selected = []
    for _ in range(seat_count):
        selected.append(rng.choice(rows))
    return selected


def _build_policy_factory(opponent_rows, candidate_seat=None):
    seat_rows = {seat: row for seat, row in enumerate(opponent_rows)}

    def policy_factory(state):
        row = seat_rows.get(state.my_id)
        if row is None:
            return GoalBasedPolicy(state)
        if candidate_seat is not None and state.my_id == int(candidate_seat):
            return GoalBasedPolicy(state)
        policy_mode = str(row.get('policy_mode', 'rule')).strip().lower()
        if policy_mode == 'rule' or row.get('kind') == 'baseline':
            return GoalBasedPolicy(state)
        return create_policy(state, mode=policy_mode, model_path=row.get('path'))

    return policy_factory

class _Player:

    def __init__(self, pid):
        self.pid = pid
        self.hand = []
        self.packs = []
        self.discards = []
        self.flowers = []

    def as_game_state(self, quan, all_players, last_tile, last_actor, last_action, last_discard, last_discard_player, request_type):
        gs = GameState()
        gs.my_id = self.pid
        gs.quan = quan
        gs.hand = self.hand[:]
        gs.packs = self.packs[:]
        gs.discards = self.discards[:]
        gs.flowers = len(self.flowers)
        for p in all_players:
            if p.pid != self.pid:
                gs.opponent_discards[p.pid] = p.discards[:]
                gs.opponent_packs[p.pid] = p.packs[:]
                for tile in p.discards + [t for _, t, _ in p.packs]:
                    gs.seen_tiles[tile] += 1
        gs.last_request_type = request_type
        gs.last_tile = last_tile
        gs.last_actor = last_actor
        gs.last_request_action = last_action
        gs._last_discard = last_discard
        gs._last_discard_player = last_discard_player
        return gs

    @property
    def n_melded(self):
        return len(self.packs)

    def fan_calc_packs(self, my_id):
        result = []
        for ptype, ptile, poffer in self.packs:
            if ptype in ('PENG', 'GANG'):
                rel = (poffer - my_id + 4) % 4
                result.append((ptype, ptile, rel))
            else:
                result.append(('CHI', ptile, poffer))
        return tuple(result)

class Game:
    """
    Simulates one round of Chinese Standard Mahjong (no wall-last tiebreak).
    """
    MIN_FAN = 8

    def __init__(self, quan=0, seed=None, dataset_writer=None, game_id=None, policy_factory=None):
        self.quan = quan
        if seed is not None:
            random.seed(seed)
        self.wall = _build_wall()
        self.players = [_Player(i) for i in range(4)]
        self.scores = [0, 0, 0, 0]
        self._wall_ptr = 0
        self._last_discard = None
        self._last_discard_player = None
        self._done = False
        self._winner = None
        self._win_fan = 0
        self._is_self_drawn = False
        self.turn_logs = []
        self.dataset_writer = dataset_writer
        self.feature_extractor = FeatureExtractor()
        self.game_id = game_id or str(uuid.uuid4())
        self._decision_index = 0
        self.policy_factory = policy_factory

    def run(self):
        """Play the full round and return the result dict."""
        self._deal()
        self._record_state('after_deal')
        turn = 0
        while not self._done and self._wall_ptr < len(self.wall):
            pid = turn % 4
            self._take_turn(pid)
            if not self._done:
                turn += 1
        return {'winner': self._winner, 'fan': self._win_fan, 'self_drawn': self._is_self_drawn, 'scores': self.scores, 'turn_logs': self.turn_logs}

    def _record_state(self, phase):
        self.turn_logs.append({'phase': phase, 'wall_remaining': len(self.wall) - self._wall_ptr, 'players': [{'pid': p.pid, 'hand': normalize_tiles(p.hand), 'discards': p.discards[:], 'packs': p.packs[:], 'flowers': p.flowers[:]} for p in self.players]})

    def _choose_policy_action(self, pid, gs):
        if self.policy_factory is None:
            policy = GoalBasedPolicy(gs)
        else:
            policy = self.policy_factory(gs)
        action = policy.choose_action()
        decision_info = getattr(policy, 'decision_info', None)
        self._emit_trajectory(pid, gs, action, decision_info=decision_info)
        return action

    def _emit_trajectory(self, pid, gs, action, decision_info=None):
        if self.dataset_writer is None:
            return
        metadata = {}
        if decision_info is not None:
            metadata['decision_info'] = decision_info
        record = TrajectoryRecord(game_id=self.game_id, turn_index=self._decision_index, player_id=pid, request_type=gs.last_request_type, request_action=gs.last_request_action, action=action, legal_actions=gs.enumerate_legal_actions(), reward=0.0, done=False, features=self.feature_extractor.extract(gs), metadata=metadata)
        self.dataset_writer.write(record)
        self._decision_index += 1

    def _deal(self):
        for p in self.players:
            while len(p.hand) < 13:
                tile = self._draw_tile()
                if tile.startswith('H'):
                    p.flowers.append(tile)
                else:
                    p.hand.append(tile)

    def _draw_tile(self):
        if self._wall_ptr >= len(self.wall):
            return ''
        tile = self.wall[self._wall_ptr]
        self._wall_ptr += 1
        return tile

    def _take_turn(self, pid):
        """Draw a tile for `pid` then ask them to act."""
        self._record_state(f'turn_start_p{pid}')
        tile = self._draw_tile()
        if not tile:
            return
        p = self.players[pid]
        while tile.startswith('H'):
            p.flowers.append(tile)
            tile = self._draw_tile()
            if not tile:
                return
        p.hand.append(tile)
        gs = p.as_game_state(self.quan, self.players, last_tile=tile, last_actor=pid, last_action='DRAW', last_discard=self._last_discard, last_discard_player=self._last_discard_player, request_type=2)
        action = self._choose_policy_action(pid, gs)
        self._apply_draw_response(pid, tile, action)
        self._record_state(f'turn_end_p{pid}')

    def _apply_draw_response(self, pid, drawn, action):
        p = self.players[pid]
        parts = action.split()
        verb = parts[0]
        if verb == 'HU':
            if self._check_win(pid, drawn, is_self_drawn=True):
                self._resolve_win(pid, drawn, is_self_drawn=True)
        elif verb == 'PLAY':
            discard = parts[1]
            if discard in p.hand:
                p.hand.remove(discard)
            p.discards.append(discard)
            self._last_discard = discard
            self._last_discard_player = pid
            self._notify_discard(pid, discard)
        elif verb == 'GANG':
            gang_tile = parts[1]
            cnt = p.hand.count(gang_tile)
            if cnt >= 4:
                for _ in range(4):
                    p.hand.remove(gang_tile)
                p.packs.append(('GANG', gang_tile, pid))
                self._supplement_draw(pid)
        elif verb == 'BUGANG':
            bg_tile = parts[1]
            if bg_tile in p.hand:
                p.hand.remove(bg_tile)
            for i, (ptype, ptile, poffer) in enumerate(p.packs):
                if ptype == 'PENG' and ptile == bg_tile:
                    p.packs[i] = ('GANG', bg_tile, poffer)
                    break
            if not self._check_rob_gang(pid, bg_tile):
                self._supplement_draw(pid)
        else:
            if drawn in p.hand:
                p.hand.remove(drawn)
            p.discards.append(drawn)
            self._last_discard = drawn
            self._last_discard_player = pid
            self._notify_discard(pid, drawn)

    def _supplement_draw(self, pid):
        """Draw a supplement tile after a kong."""
        tile = self._draw_tile()
        if not tile:
            return
        p = self.players[pid]
        while tile.startswith('H'):
            p.flowers.append(tile)
            tile = self._draw_tile()
            if not tile:
                return
        p.hand.append(tile)
        gs = p.as_game_state(self.quan, self.players, last_tile=tile, last_actor=pid, last_action='DRAW', last_discard=self._last_discard, last_discard_player=self._last_discard_player, request_type=2)
        action = self._choose_policy_action(pid, gs)
        self._apply_draw_response(pid, tile, action)

    def _notify_discard(self, discard_player, tile):
        """Let opponents respond to the discard (HU > GANG > PENG > CHI)."""
        if self._done:
            return
        for offset in range(1, 4):
            pid = (discard_player + offset) % 4
            if self._claim_hu(pid, tile, discard_player):
                return
        for offset in range(1, 4):
            pid = (discard_player + offset) % 4
            p = self.players[pid]
            if p.hand.count(tile) >= 3:
                gs = p.as_game_state(self.quan, self.players, last_tile=tile, last_actor=discard_player, last_action='PLAY', last_discard=tile, last_discard_player=discard_player, request_type=3)
                resp = self._choose_policy_action(pid, gs)
                if resp == 'GANG':
                    for _ in range(3):
                        p.hand.remove(tile)
                    p.packs.append(('GANG', tile, discard_player))
                    self._last_discard = None
                    self._supplement_draw(pid)
                    return
        for offset in range(1, 4):
            pid = (discard_player + offset) % 4
            p = self.players[pid]
            if p.hand.count(tile) >= 2:
                gs = p.as_game_state(self.quan, self.players, last_tile=tile, last_actor=discard_player, last_action='PLAY', last_discard=tile, last_discard_player=discard_player, request_type=3)
                resp = self._choose_policy_action(pid, gs)
                if resp.startswith('PENG'):
                    parts = resp.split()
                    disc = parts[1] if len(parts) > 1 else None
                    if disc:
                        for _ in range(2):
                            p.hand.remove(tile)
                        p.packs.append(('PENG', tile, discard_player))
                        self._last_discard = tile
                        self._last_discard_player = discard_player
                        if disc in p.hand:
                            p.hand.remove(disc)
                        p.discards.append(disc)
                        self._last_discard = disc
                        self._last_discard_player = pid
                        self._notify_discard(pid, disc)
                        return
        chi_pid = (discard_player + 1) % 4
        p = self.players[chi_pid]
        gs = p.as_game_state(self.quan, self.players, last_tile=tile, last_actor=discard_player, last_action='PLAY', last_discard=tile, last_discard_player=discard_player, request_type=3)
        resp = self._choose_policy_action(chi_pid, gs)
        if resp.startswith('CHI'):
            parts = resp.split()
            mid = parts[1] if len(parts) > 1 else None
            disc = parts[2] if len(parts) > 2 else None
            if mid and disc:
                mid_val = int(mid[1:])
                suit = mid[0]
                seq = [f'{suit}{mid_val - 1}', mid, f'{suit}{mid_val + 1}']
                from_hand = [t for t in seq if t != tile]
                if all((t in p.hand for t in from_hand)):
                    offer = seq.index(tile)
                    for t in from_hand:
                        p.hand.remove(t)
                    p.packs.append(('CHI', mid, offer))
                    self._last_discard = tile
                    self._last_discard_player = discard_player
                    if disc in p.hand:
                        p.hand.remove(disc)
                    p.discards.append(disc)
                    self._last_discard = disc
                    self._last_discard_player = chi_pid
                    self._notify_discard(chi_pid, disc)
                    return

    def _claim_hu(self, pid, tile, from_player):
        p = self.players[pid]
        gs = p.as_game_state(self.quan, self.players, last_tile=tile, last_actor=from_player, last_action='PLAY', last_discard=tile, last_discard_player=from_player, request_type=3)
        resp = self._choose_policy_action(pid, gs)
        if resp == 'HU':
            if self._check_win(pid, tile, is_self_drawn=False, from_player=from_player):
                self._resolve_win(pid, tile, is_self_drawn=False, from_player=from_player)
                return True
        return False

    def _check_rob_gang(self, gang_player, tile):
        """Check if any opponent can rob the supplement kong."""
        for pid in range(4):
            if pid == gang_player:
                continue
            p = self.players[pid]
            gs = p.as_game_state(self.quan, self.players, last_tile=tile, last_actor=gang_player, last_action='BUGANG', last_discard=self._last_discard, last_discard_player=self._last_discard_player, request_type=3)
            resp = self._choose_policy_action(pid, gs)
            if resp == 'HU':
                if self._check_win(pid, tile, is_self_drawn=False, from_player=gang_player, is_about_kong=True):
                    self._resolve_win(pid, tile, is_self_drawn=False, from_player=gang_player)
                    return True
        return False

    def _check_win(self, pid, win_tile, is_self_drawn, from_player=None, is_about_kong=False):
        p = self.players[pid]
        hand = p.hand[:]
        if is_self_drawn and win_tile in hand:
            hand.remove(win_tile)
        fc_packs = p.fan_calc_packs(pid)
        fan = calculate_fan(fc_packs, tuple(hand), win_tile, len(p.flowers), is_self_drawn, False, is_about_kong, False, pid % 4, self.quan)
        return fan >= self.MIN_FAN

    def _resolve_win(self, pid, win_tile, is_self_drawn, from_player=None):
        self._done = True
        self._winner = pid
        self._is_self_drawn = is_self_drawn
        p = self.players[pid]
        hand = p.hand[:]
        if is_self_drawn and win_tile in hand:
            hand.remove(win_tile)
        fc_packs = p.fan_calc_packs(pid)
        fan = calculate_fan(fc_packs, tuple(hand), win_tile, len(p.flowers), is_self_drawn, False, False, False, pid % 4, self.quan)
        self._win_fan = fan
        base = 8 + fan
        if is_self_drawn:
            for other in range(4):
                if other != pid:
                    self.scores[other] -= base
                    self.scores[pid] += base
        else:
            payer = from_player if from_player is not None else (pid + 3) % 4
            self.scores[payer] -= 8 + fan
            self.scores[pid] += (8 + fan) * 3 - (8 + fan) * 2
            self.scores[pid] = 3 * base - 2 * base
            for i in range(4):
                self.scores[i] = 0
            self.scores[pid] = 3 * base
            for other in range(4):
                if other == payer:
                    self.scores[other] -= 8 + fan
                elif other != pid:
                    self.scores[other] -= 8

def run_games(n=1, quan=0, seed=None, show_turns=False, tui=False, tui_delay=0.05, no_clear=False, export_dataset_path=None, opponent_registry_path=None, random_opponents=False):
    wins = defaultdict(int)
    total_fan = defaultdict(int)
    draws = 0
    writer = None
    opponent_pool = _load_opponent_registry(opponent_registry_path)
    if export_dataset_path:
        writer = JsonlTrajectoryWriter(export_dataset_path)
    try:
        for i in range(n):
            game_seed = None if seed is None else seed + i
            policy_factory = None
            if opponent_pool and random_opponents:
                rng = random.Random(game_seed if game_seed is not None else i)
                opponent_rows = _sample_opponent_rows(opponent_pool, 4, rng)
                policy_factory = _build_policy_factory(opponent_rows)

            g = Game(quan=quan, seed=game_seed, dataset_writer=writer, game_id=f'game-{i}', policy_factory=policy_factory)
            result = g.run()
            if result['winner'] is None:
                draws += 1
            else:
                w = result['winner']
                wins[w] += 1
                total_fan[w] += result['fan']
            if n == 1:
                print(f"Winner: player {result['winner']}  fan={result['fan']}  self_drawn={result['self_drawn']}")
                print(f"Scores: {result['scores']}")
                if tui:
                    for state in result['turn_logs']:
                        render_tui_board(state, game_index=i + 1, total_games=n, clear_screen=not no_clear)
                        if tui_delay > 0:
                            time.sleep(tui_delay)
                    print('\nFinal result:')
                    print(f"  winner={result['winner']} fan={result['fan']} self_drawn={result['self_drawn']} scores={result['scores']}")
                if show_turns:
                    print('\nTurn-by-turn state trace:')
                    for state in result['turn_logs']:
                        print(f"\n  phase={state['phase']}  wall_remaining={state['wall_remaining']}")
                        for player_state in state['players']:
                            print(f"    P{player_state['pid']} hand={player_state['hand']} discards={player_state['discards']}")
                print('\nFinal game state:')
                final_state = result['turn_logs'][-1]
                print(f"  phase={final_state['phase']}  wall_remaining={final_state['wall_remaining']}")
                for player_state in final_state['players']:
                    print(f"  Player {player_state['pid']}")
                    print(f"    hand ({len(player_state['hand'])}): {player_state['hand']}")
                    print(f"    discards ({len(player_state['discards'])}): {player_state['discards']}")
                    print(f"    packs: {player_state['packs']}")
                    print(f"    flowers ({len(player_state['flowers'])}): {player_state['flowers']}")
    finally:
        if writer is not None:
            writer.close()
    if n > 1:
        print(f"\n{'=' * 40}")
        print(f'Games: {n}  Draws: {draws}')
        for pid in range(4):
            rate = wins[pid] / n * 100
            avg_fan = total_fan[pid] / wins[pid] if wins[pid] else 0
            print(f'  Player {pid}: {wins[pid]} wins ({rate:.1f}%)  avg_fan={avg_fan:.1f}')
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Local Mahjong game simulator')
    parser.add_argument('--games', type=int, default=1, help='Number of games to simulate')
    parser.add_argument('--quan', type=int, default=0, help='Prevalent wind (0=East)')
    parser.add_argument('--seed', type=int, default=None, help='Random seed')
    parser.add_argument('--show-turns', action='store_true', help='When --games=1, print turn-by-turn hands/discards for all players')
    parser.add_argument('--tui', action='store_true', help='When --games=1, animate a simple terminal board view')
    parser.add_argument('--tui-delay', type=float, default=0.05, help='Seconds between TUI frames (default: 0.05)')
    parser.add_argument('--no-clear', action='store_true', help='Do not clear screen between TUI frames')
    parser.add_argument('--export-dataset', type=str, default=None, help='Optional path to write trajectory dataset JSONL.')
    parser.add_argument('--opponent-registry', type=str, default=None, help='Path to opponents_registry.json to sample opponents from')
    parser.add_argument('--random-opponents', action='store_true', help='Randomly sample all four seats from the opponent registry for each game')
    args = parser.parse_args()
    run_games(n=args.games, quan=args.quan, seed=args.seed, show_turns=args.show_turns, tui=args.tui, tui_delay=args.tui_delay, no_clear=args.no_clear, export_dataset_path=args.export_dataset, opponent_registry_path=args.opponent_registry, random_opponents=args.random_opponents)
