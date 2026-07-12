import glob
import json
import os

from dataset import JsonlTrajectoryWriter, TrajectoryRecord


def _safe_int(value, default_value):
    try:
        return int(value)
    except Exception:
        return default_value


def _parse_request(raw_request):
    text = str(raw_request or '').strip()
    if not text:
        return -1, None
    parts = text.split()
    request_type = _safe_int(parts[0], -1)
    request_action = ' '.join(parts[1:]) if len(parts) > 1 else None
    return request_type, request_action


def _parse_response_action(response):
    if response is None:
        return None
    return str(response).strip() or None


def _extract_players(match_payload):
    players = {}
    for row in match_payload.get('players', []):
        try:
            seat = _safe_int(row.get('seat'), -1)
        except Exception:
            seat = -1
        if seat < 0:
            continue
        players[seat] = {
            'name': row.get('name') or ('seat-%d' % seat),
            'target': bool(row.get('target', False)),
            'uid': row.get('uid'),
        }
    return players


def _iter_match_records(match_payload, source_name):
    game_id = str(match_payload.get('match_id') or match_payload.get('_id') or 'unknown-game')
    players = _extract_players(match_payload)
    logs = match_payload.get('log', [])

    pending_requests = {}
    turn_index = 0

    for event in logs:
        if not isinstance(event, dict):
            continue

        output = event.get('output')
        if isinstance(output, dict):
            content = output.get('content', {})
            display = output.get('display', {}) if isinstance(output.get('display'), dict) else {}
            if isinstance(content, dict):
                for seat_key, request_text in content.items():
                    seat = _safe_int(seat_key, -1)
                    if seat < 0:
                        continue
                    request_type, request_action = _parse_request(request_text)
                    pending_requests[seat] = {
                        'turn_index': turn_index,
                        'request_type': request_type,
                        'request_action': request_action,
                        'event_action': display.get('action'),
                    }
                    turn_index += 1
            continue

        has_player_rows = False
        for seat_key, row in event.items():
            seat = _safe_int(seat_key, -1)
            if seat < 0 or not isinstance(row, dict):
                continue
            has_player_rows = True
            action = _parse_response_action(row.get('response'))
            if not action:
                continue

            pending = pending_requests.get(seat, {})
            player_row = players.get(seat, {})
            metadata = {
                'source_dataset': source_name,
                'source_format': 'botzone_match_json_v1',
                'seat': seat,
                'player_name': player_row.get('name'),
                'player_uid': player_row.get('uid'),
                'target_player': bool(player_row.get('target', False)),
                'verdict': row.get('verdict'),
                'time_ms': row.get('time'),
                'memory_mb': row.get('memory'),
            }

            features = {
                'event_action': pending.get('event_action'),
                'raw_request': pending.get('request_action'),
                'target_player': bool(player_row.get('target', False)),
            }

            yield TrajectoryRecord(
                game_id=game_id,
                turn_index=_safe_int(pending.get('turn_index', turn_index), turn_index),
                player_id=seat,
                request_type=_safe_int(pending.get('request_type', -1), -1),
                request_action=pending.get('request_action'),
                action=action,
                legal_actions=[action],
                reward=0.0,
                done=False,
                features=features,
                metadata=metadata,
            )
            turn_index += 1

        if has_player_rows:
            pending_requests = {}


def ingest_games_directory(games_dir, output_path, source_name='external-games'):
    parent = os.path.dirname(output_path)
    if parent and not os.path.exists(parent):
        os.makedirs(parent)

    files = sorted(glob.glob(os.path.join(games_dir, '*.json')))
    stats = {
        'games_dir': games_dir,
        'output_path': output_path,
        'source_name': source_name,
        'files_seen': len(files),
        'files_ingested': 0,
        'files_failed': 0,
        'records_written': 0,
    }

    with JsonlTrajectoryWriter(output_path) as writer:
        for path in files:
            try:
                with open(path, 'r', encoding='utf-8') as handle:
                    payload = json.load(handle)
                written_here = 0
                for record in _iter_match_records(payload, source_name):
                    writer.write(record)
                    written_here += 1
                stats['records_written'] += written_here
                stats['files_ingested'] += 1
            except Exception:
                stats['files_failed'] += 1

    return stats


def build_opponent_registry(models_dir, output_path):
    parent = os.path.dirname(output_path)
    if parent and not os.path.exists(parent):
        os.makedirs(parent)

    files = sorted(glob.glob(os.path.join(models_dir, '*.pkl')))
    opponents = []
    for path in files:
        name = os.path.basename(path)
        model_id, _ = os.path.splitext(name)
        opponents.append(
            {
                'id': model_id,
                'file_name': name,
                'path': path,
                'policy_mode': 'imitation',
            }
        )

    registry = {
        'models_dir': models_dir,
        'count': len(opponents),
        'opponents': opponents,
    }

    with open(output_path, 'w', encoding='utf-8') as handle:
        json.dump(registry, handle, ensure_ascii=True, sort_keys=True, indent=2)

    return registry