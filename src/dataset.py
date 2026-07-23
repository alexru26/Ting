"""Trajectory record format and JSONL IO for locally generated games."""

import json


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
                yield TrajectoryRecord.from_dict(json.loads(text))
