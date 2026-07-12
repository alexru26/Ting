import json
import os
import random


class TrajectoryRecord:
    def __init__(
        self,
        game_id,
        turn_index,
        player_id,
        request_type,
        request_action,
        action,
        legal_actions,
        reward=0.0,
        done=False,
        features=None,
        metadata=None,
    ):
        self.game_id = game_id
        self.turn_index = turn_index
        self.player_id = player_id
        self.request_type = request_type
        self.request_action = request_action
        self.action = action
        self.legal_actions = list(legal_actions)
        self.reward = float(reward)
        self.done = bool(done)
        self.features = dict(features or {})
        self.metadata = dict(metadata or {})

    def to_dict(self):
        return {
            'game_id': self.game_id,
            'turn_index': self.turn_index,
            'player_id': self.player_id,
            'request_type': self.request_type,
            'request_action': self.request_action,
            'action': self.action,
            'legal_actions': self.legal_actions,
            'reward': self.reward,
            'done': self.done,
            'features': self.features,
            'metadata': self.metadata,
        }

    @classmethod
    def from_dict(cls, payload):
        return cls(
            game_id=payload['game_id'],
            turn_index=payload['turn_index'],
            player_id=payload['player_id'],
            request_type=payload['request_type'],
            request_action=payload.get('request_action'),
            action=payload['action'],
            legal_actions=list(payload.get('legal_actions', [])),
            reward=float(payload.get('reward', 0.0)),
            done=bool(payload.get('done', False)),
            features=dict(payload.get('features', {})),
            metadata=dict(payload.get('metadata', {})),
        )


class JsonlTrajectoryWriter:
    def __init__(self, file_path):
        self.file_path = file_path
        self._handle = open(file_path, 'w', encoding='utf-8')

    def write(self, record):
        self._handle.write(json.dumps(record.to_dict(), ensure_ascii=True) + '\n')
        self._handle.flush()

    def close(self):
        if self._handle and not self._handle.closed:
            self._handle.close()

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()


class JsonlTrajectoryReader:
    def __init__(self, file_path):
        self.file_path = file_path

    def __iter__(self):
        with open(self.file_path, 'r', encoding='utf-8') as handle:
            for line in handle:
                text = line.strip()
                if not text:
                    continue
                payload = json.loads(text)
                yield TrajectoryRecord.from_dict(payload)


def _as_int(value, default_value):
    try:
        return int(value)
    except Exception:
        return default_value


def _as_float(value, default_value):
    try:
        return float(value)
    except Exception:
        return default_value


def _as_bool(value, default_value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        text = value.strip().lower()
        if text in ('1', 'true', 'yes', 'y'):
            return True
        if text in ('0', 'false', 'no', 'n'):
            return False
    return bool(value) if value is not None else default_value


def _as_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, set):
        return sorted(list(value))
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        if ',' in text:
            return [part.strip() for part in text.split(',') if part.strip()]
        return [text]
    return [value]


def normalize_external_record(payload, source_name='external'):
    if not isinstance(payload, dict):
        return None

    action = payload.get('action')
    if action is None:
        action = payload.get('chosen_action')
    if action is None:
        action = payload.get('response')
    if not action:
        return None

    legal_actions = payload.get('legal_actions')
    if legal_actions is None:
        legal_actions = payload.get('legalMoves')
    legal_actions = _as_list(legal_actions)
    if not legal_actions:
        legal_actions = [action]

    features = payload.get('features')
    if features is None:
        features = payload.get('state_features')
    if not isinstance(features, dict):
        features = {}

    metadata = payload.get('metadata')
    if not isinstance(metadata, dict):
        metadata = {}
    metadata = dict(metadata)
    metadata['source_dataset'] = source_name

    return TrajectoryRecord(
        game_id=str(payload.get('game_id') or payload.get('match_id') or payload.get('round_id') or 'unknown-game'),
        turn_index=_as_int(payload.get('turn_index', payload.get('turn', 0)), 0),
        player_id=_as_int(payload.get('player_id', payload.get('seat', -1)), -1),
        request_type=_as_int(payload.get('request_type', -1), -1),
        request_action=payload.get('request_action'),
        action=str(action),
        legal_actions=[str(item) for item in legal_actions],
        reward=_as_float(payload.get('reward', payload.get('score_delta', 0.0)), 0.0),
        done=_as_bool(payload.get('done', payload.get('terminal', False)), False),
        features=features,
        metadata=metadata,
    )


def ingest_external_jsonl(input_path, output_path, source_name='external', drop_invalid=True):
    total_lines = 0
    written = 0
    dropped = 0

    parent = os.path.dirname(output_path)
    if parent and not os.path.exists(parent):
        os.makedirs(parent)

    with open(input_path, 'r', encoding='utf-8') as handle, JsonlTrajectoryWriter(output_path) as writer:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            total_lines += 1
            payload = json.loads(text)
            record = normalize_external_record(payload, source_name=source_name)
            if record is None:
                dropped += 1
                if not drop_invalid:
                    raise ValueError('Invalid record at line %d' % total_lines)
                continue
            writer.write(record)
            written += 1

    return {
        'input_path': input_path,
        'output_path': output_path,
        'source_name': source_name,
        'total_lines': total_lines,
        'written': written,
        'dropped': dropped,
    }


def _split_ids(ids, train_ratio, val_ratio):
    total = len(ids)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    train_ids = ids[:train_end]
    val_ids = ids[train_end:val_end]
    test_ids = ids[val_end:]
    return train_ids, val_ids, test_ids


def create_mixed_split_manifest(
    local_dataset_path,
    external_dataset_path,
    out_manifest_path,
    train_ratio=0.8,
    val_ratio=0.1,
    seed=42,
):
    if train_ratio <= 0 or val_ratio < 0 or train_ratio + val_ratio >= 1:
        raise ValueError('Invalid split ratios')

    local_game_ids = sorted({record.game_id for record in JsonlTrajectoryReader(local_dataset_path)})
    external_game_ids = sorted({record.game_id for record in JsonlTrajectoryReader(external_dataset_path)})

    local_rng = random.Random(int(seed))
    external_rng = random.Random(int(seed) + 1)

    local_rng.shuffle(local_game_ids)
    external_rng.shuffle(external_game_ids)

    local_train, local_val, local_test = _split_ids(local_game_ids, train_ratio, val_ratio)
    ext_train, ext_val, ext_test = _split_ids(external_game_ids, train_ratio, val_ratio)

    manifest = {
        'seed': int(seed),
        'ratios': {
            'train': float(train_ratio),
            'val': float(val_ratio),
            'test': float(1.0 - train_ratio - val_ratio),
        },
        'sources': [
            {'name': 'local', 'dataset_path': local_dataset_path},
            {'name': 'external', 'dataset_path': external_dataset_path},
        ],
        'splits': {
            'train': {
                'local_game_ids': local_train,
                'external_game_ids': ext_train,
            },
            'val': {
                'local_game_ids': local_val,
                'external_game_ids': ext_val,
            },
            'test': {
                'local_game_ids': local_test,
                'external_game_ids': ext_test,
            },
        },
    }

    parent = os.path.dirname(out_manifest_path)
    if parent and not os.path.exists(parent):
        os.makedirs(parent)

    with open(out_manifest_path, 'w', encoding='utf-8') as handle:
        json.dump(manifest, handle, ensure_ascii=True, sort_keys=True, indent=2)

    return manifest
