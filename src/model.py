import json
import os
from typing import Any, cast

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from action_codec import ActionCodec
from ml_packages import package_profile
from tiles import ALL_TILES


FAMILY_LABELS = ['PASS', 'HU', 'GANG', 'PLAY', 'BUGANG', 'PENG', 'CHI']
NONE_TOKEN = '<NONE>'

EVENT_VOCAB = [
    None,
    'INIT',
    'DEAL',
    'DRAW',
    'PLAY',
    'PENG',
    'CHI',
    'GANG',
    'BUGANG',
    'HU',
    'PASS',
]

REQUEST_VOCAB = [
    None,
    'DRAW',
    'PLAY',
    'GANG',
    'BUGANG',
    'HU',
    'PASS',
]


TILE_COUNT = len(ALL_TILES)
CHANNEL_COUNT = 7
META_COUNT = 8 + 3 + len(EVENT_VOCAB) + len(REQUEST_VOCAB)
INPUT_SIZE = TILE_COUNT * CHANNEL_COUNT + META_COUNT


def _safe_int(value, default_value):
    try:
        return int(value)
    except Exception:
        return default_value


def _safe_float(value, default_value):
    try:
        return float(value)
    except Exception:
        return default_value


class CnnCore(nn.Module):
    def __init__(self, hidden_size, family_size, arg_size):
        super().__init__()
        self.conv1 = nn.Conv1d(CHANNEL_COUNT, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(16, 16, kernel_size=3, padding=1)
        self.meta_proj = nn.Linear(META_COUNT, 32)
        self.hidden_fuse = nn.Linear(16 + 32, hidden_size)
        self.belief_head = nn.Linear(hidden_size, TILE_COUNT)
        self.belief_proj = nn.Linear(TILE_COUNT, 16)
        self.output_fuse = nn.Linear(hidden_size + 16, hidden_size)
        self.family_head = nn.Linear(hidden_size, family_size)
        self.arg1_head = nn.Linear(hidden_size, arg_size)
        self.arg2_head = nn.Linear(hidden_size, arg_size)
        self.value_head = nn.Linear(hidden_size, 1)

    def forward(self, tile_tensor, meta_tensor):
        tile_features = F.relu(self.conv1(tile_tensor))
        tile_features = F.relu(self.conv2(tile_features))
        tile_features = F.adaptive_avg_pool1d(tile_features, 1).squeeze(-1)

        meta_features = F.relu(self.meta_proj(meta_tensor))
        hidden = F.relu(self.hidden_fuse(torch.cat([tile_features, meta_features], dim=-1)))
        belief_logits = self.belief_head(hidden)
        belief_probs = torch.softmax(belief_logits, dim=-1)
        belief_context = F.relu(self.belief_proj(belief_probs))
        hidden = F.relu(self.output_fuse(torch.cat([hidden, belief_context], dim=-1)))

        return {
            'hidden': hidden,
            'belief_logits': belief_logits,
            'belief_probs': belief_probs,
            'family_logits': self.family_head(hidden),
            'arg1_logits': self.arg1_head(hidden),
            'arg2_logits': self.arg2_head(hidden),
            'value': self.value_head(hidden),
        }


class CnnPolicyValueModel:
    def __init__(
        self,
        action_space_size,
        hidden_size=32,
        learning_rate=0.001,
        seed=7,
        state_dict=None,
        metadata=None,
    ):
        self.action_space_size = int(action_space_size)
        self.hidden_size = int(hidden_size)
        self.learning_rate = float(learning_rate)
        self.seed = int(seed)
        self.metadata = dict(metadata or {})
        self.backend = 'torch'
        self.package_profile = package_profile()
        self.codec = ActionCodec()
        self.family_vocab = list(FAMILY_LABELS)
        self.family_index = {label: idx for idx, label in enumerate(self.family_vocab)}
        self.arg_vocab = list(ALL_TILES) + [NONE_TOKEN]
        self.arg_index = {label: idx for idx, label in enumerate(self.arg_vocab)}
        self.device = torch.device('cpu')
        torch.manual_seed(self.seed)

        self.model = CnnCore(
            hidden_size=self.hidden_size,
            family_size=len(self.family_vocab),
            arg_size=len(self.arg_vocab),
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        if state_dict is not None:
            self.load_state_dict(state_dict)

    def _zero_tile_tensor(self):
        return torch.zeros((CHANNEL_COUNT, TILE_COUNT), dtype=torch.float32, device=self.device)

    def _zero_meta_tensor(self):
        return torch.zeros((META_COUNT,), dtype=torch.float32, device=self.device)

    def _channel_tensor(self, values):
        tensor = torch.zeros((TILE_COUNT,), dtype=torch.float32, device=self.device)
        if isinstance(values, list):
            limit = min(len(values), TILE_COUNT)
            if limit > 0:
                tensor[:limit] = torch.tensor(values[:limit], dtype=torch.float32, device=self.device)
        return tensor

    def _encode_features(self, features):
        if not isinstance(features, dict):
            features = {}

        tile_tensor = self._zero_tile_tensor()
        channel_keys = [
            'hand_counts',
            'seen_counts',
            'self_discard_counts',
            'pack_counts',
        ]

        for channel_index, key in enumerate(channel_keys):
            tile_tensor[channel_index] = self._channel_tensor(features.get(key))

        opponent_channels = features.get('opponent_discard_counts')
        if isinstance(opponent_channels, list):
            for offset, channel in enumerate(opponent_channels[:3], start=4):
                tile_tensor[offset] = self._channel_tensor(channel if isinstance(channel, list) else [])

        meta_values = []
        meta = features.get('meta')
        if isinstance(meta, list):
            meta_values.extend(float(value) for value in meta[:8])
        while len(meta_values) < 8:
            meta_values.append(0.0)

        request_type = _safe_int(features.get('request_type', 0), 0)
        seat = _safe_int(features.get('seat', features.get('player_id', -1)), -1)
        target_player = 1.0 if features.get('target_player') else 0.0
        event_action = features.get('event_action')
        raw_request = features.get('raw_request')

        meta_values.extend([float(request_type), float(seat), target_player])
        meta_values.extend(self._one_hot(event_action, EVENT_VOCAB))
        meta_values.extend(self._one_hot(raw_request, REQUEST_VOCAB))

        meta_tensor = torch.tensor(meta_values, dtype=torch.float32, device=self.device)
        return tile_tensor.unsqueeze(0), meta_tensor.unsqueeze(0)

    def _one_hot(self, value, vocabulary):
        vector = [0.0] * len(vocabulary)
        try:
            index = vocabulary.index(value)
        except ValueError:
            index = 0
        vector[index] = 1.0
        return vector

    def _belief_target_from_features(self, features):
        seen_counts = features.get('seen_counts') if isinstance(features, dict) else None
        if not isinstance(seen_counts, list):
            seen_counts = [0] * TILE_COUNT

        target = []
        total = 0.0
        for seen in seen_counts[:TILE_COUNT]:
            remaining = max(0.0, 4.0 - _safe_float(seen, 0.0))
            target.append(float(remaining))
            total += float(remaining)

        while len(target) < TILE_COUNT:
            target.append(0.0)

        if total <= 0.0:
            uniform = 1.0 / float(TILE_COUNT)
            return [uniform for _ in range(TILE_COUNT)]

        return [value / total for value in target]

    def _belief_entropy(self, belief_probs):
        probs = torch.clamp(belief_probs, min=1e-12)
        return float((-(probs * torch.log(probs))).sum().item())

    def _action_indices(self, action):
        parts = action.split()
        family = parts[0] if parts else 'PASS'
        arg1 = parts[1] if len(parts) > 1 else NONE_TOKEN
        arg2 = parts[2] if len(parts) > 2 else NONE_TOKEN
        return self.family_index.get(family, 0), self.arg_index.get(arg1, self.arg_index[NONE_TOKEN]), self.arg_index.get(arg2, self.arg_index[NONE_TOKEN])

    def _action_score(self, outputs, action):
        family_index, arg1_index, arg2_index = self._action_indices(action)
        return outputs['family_logits'][0, family_index] + outputs['arg1_logits'][0, arg1_index] + outputs['arg2_logits'][0, arg2_index]

    def _belief_bonus(self, outputs, action):
        if 'PLAY ' not in action and 'GANG ' not in action and 'BUGANG ' not in action:
            return 0.0
        parts = action.split()
        tile = parts[-1] if parts else NONE_TOKEN
        tile_index = self.arg_index.get(tile)
        if tile_index is None:
            return 0.0
        belief_probs = outputs.get('belief_probs')
        if belief_probs is None:
            return 0.0
        return float(belief_probs[0, tile_index].item())

    def _legal_action_scores(self, outputs, legal_actions, belief_weight=0.0):
        scores = []
        for action in legal_actions:
            score = self._action_score(outputs, action)
            if belief_weight:
                score = score + float(belief_weight) * self._belief_bonus(outputs, action)
            scores.append(score)
        return torch.stack(scores, dim=0)

    def policy_info_from_features(self, features, legal_actions, belief_weight=0.0):
        if not legal_actions:
            return {
                'actions': [],
                'scores': [],
                'probabilities': [],
                'log_probabilities': [],
                'belief_probs': [],
                'belief_entropy': 0.0,
                'entropy': 0.0,
                'value': 0.0,
            }

        self.model.eval()
        with torch.no_grad():
            tile_tensor, meta_tensor = self._encode_features(features)
            outputs = self.model(tile_tensor, meta_tensor)
            scores = self._legal_action_scores(outputs, legal_actions, belief_weight=belief_weight)
            log_probabilities = torch.log_softmax(scores, dim=0)
            probabilities = torch.softmax(scores, dim=0)
            entropy = float((-(probabilities * log_probabilities)).sum().item())
            value = float(outputs['value'][0, 0].item())
            belief_probs = outputs.get('belief_probs')
            if belief_probs is None:
                belief_list = []
                belief_entropy = 0.0
            else:
                belief_list = [float(probability.item()) for probability in belief_probs[0]]
                belief_entropy = self._belief_entropy(belief_probs)

        return {
            'actions': list(legal_actions),
            'scores': [float(score.item()) for score in scores],
            'probabilities': [float(probability.item()) for probability in probabilities],
            'log_probabilities': [float(log_probability.item()) for log_probability in log_probabilities],
            'belief_probs': belief_list,
            'belief_entropy': belief_entropy,
            'entropy': entropy,
            'value': value,
        }

    def sample_action_from_features(self, features, legal_actions, temperature=1.0, greedy=False, belief_weight=0.0):
        info = self.policy_info_from_features(features, legal_actions, belief_weight=belief_weight)
        actions = info['actions']
        if not actions:
            return None, info

        if greedy:
            index = max(range(len(actions)), key=lambda idx: (info['probabilities'][idx], actions[idx]))
        else:
            probabilities = torch.tensor(info['probabilities'], dtype=torch.float32, device=self.device)
            if temperature and float(temperature) != 1.0:
                logits = torch.log(torch.clamp(probabilities, min=1e-12)) / float(temperature)
                probabilities = torch.softmax(logits, dim=0)
            index = int(torch.multinomial(probabilities, 1).item())

        selected_action = actions[index]
        selected_log_prob = info['log_probabilities'][index]
        selected_probability = info['probabilities'][index]
        info = dict(info)
        info['selected_action'] = selected_action
        info['selected_index'] = index
        info['selected_log_prob'] = float(selected_log_prob)
        info['selected_probability'] = float(selected_probability)
        return selected_action, info

    def action_distribution_from_features(self, features, legal_actions, belief_weight=0.0):
        if not legal_actions:
            return {}

        self.model.eval()
        with torch.no_grad():
            tile_tensor, meta_tensor = self._encode_features(features)
            outputs = self.model(tile_tensor, meta_tensor)
            scores = self._legal_action_scores(outputs, legal_actions, belief_weight=belief_weight)
            probabilities = torch.softmax(scores, dim=0)

        return {action: float(probability.item()) for action, probability in zip(legal_actions, probabilities)}

    def choose_action_from_features(self, features, legal_actions, belief_weight=0.0):
        distribution = self.action_distribution_from_features(features, legal_actions, belief_weight=belief_weight)
        if not distribution:
            return None
        ranked = sorted(distribution.items(), key=lambda item: (-item[1], item[0]))
        return ranked[0][0]

    def estimate_value_from_features(self, features):
        self.model.eval()
        with torch.no_grad():
            tile_tensor, meta_tensor = self._encode_features(features)
            outputs = self.model(tile_tensor, meta_tensor)
            return float(outputs['value'][0, 0].item())

    def train_step(self, features, legal_actions, action, reward, policy_weight=1.0, value_weight=0.5, belief_weight=0.25):
        if not legal_actions:
            legal_actions = [action]
        if action not in legal_actions:
            legal_actions = list(legal_actions) + [action]
        legal_action_count = len(legal_actions)
        is_decision_state = legal_action_count > 1

        self.model.train()
        tile_tensor, meta_tensor = self._encode_features(features)
        outputs = self.model(tile_tensor, meta_tensor)

        scores = torch.stack([self._action_score(outputs, candidate) for candidate in legal_actions], dim=0).unsqueeze(0)
        target_index = torch.tensor([legal_actions.index(action)], dtype=torch.long, device=self.device)
        action_loss = F.cross_entropy(scores, target_index)

        value_target = torch.tensor([[float(_safe_float(reward, 0.0))]], dtype=torch.float32, device=self.device)
        value_loss = F.mse_loss(outputs['value'], value_target)
        belief_target = torch.tensor([self._belief_target_from_features(features)], dtype=torch.float32, device=self.device)
        belief_loss = F.kl_div(torch.log(torch.clamp(outputs['belief_probs'], min=1e-12)), belief_target, reduction='batchmean')

        loss = float(policy_weight) * action_loss + float(value_weight) * value_loss + float(belief_weight) * belief_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        with torch.no_grad():
            probabilities = torch.softmax(scores.squeeze(0), dim=0)
            action_hit = int(int(torch.argmax(probabilities).item()) == int(target_index.item()))

        return {
            'action_loss': float(action_loss.item()),
            'value_loss': float(value_loss.item()),
            'belief_loss': float(belief_loss.item()),
            'weighted_total_loss': float(loss.item()),
            'action_hit': action_hit,
            'is_decision_state': bool(is_decision_state),
            'legal_action_count': int(legal_action_count),
        }

    def ppo_train_step(self, features, legal_actions, action, advantage, return_target, old_log_prob=None, clip_range=0.2, entropy_coef=0.01, value_coef=0.5, belief_coef=0.25):
        if not legal_actions:
            legal_actions = [action]
        if action not in legal_actions:
            legal_actions = list(legal_actions) + [action]

        self.model.train()
        tile_tensor, meta_tensor = self._encode_features(features)
        outputs = self.model(tile_tensor, meta_tensor)

        scores = self._legal_action_scores(outputs, legal_actions)
        log_probabilities = torch.log_softmax(scores, dim=0)
        probabilities = torch.softmax(scores, dim=0)
        action_index = legal_actions.index(action)
        current_log_prob = log_probabilities[action_index]

        if old_log_prob is None:
            old_log_prob = float(current_log_prob.item())

        ratio = torch.exp(current_log_prob - torch.tensor(float(old_log_prob), dtype=torch.float32, device=self.device))
        clipped_ratio = torch.clamp(ratio, 1.0 - float(clip_range), 1.0 + float(clip_range))
        advantage_tensor = torch.tensor(float(advantage), dtype=torch.float32, device=self.device)
        surrogate = torch.minimum(ratio * advantage_tensor, clipped_ratio * advantage_tensor)
        policy_loss = -surrogate

        entropy = -(probabilities * log_probabilities).sum()
        return_tensor = torch.tensor([[float(return_target)]], dtype=torch.float32, device=self.device)
        value_loss = F.mse_loss(outputs['value'], return_tensor)
        belief_target = torch.tensor([self._belief_target_from_features(features)], dtype=torch.float32, device=self.device)
        belief_loss = F.kl_div(torch.log(torch.clamp(outputs['belief_probs'], min=1e-12)), belief_target, reduction='batchmean')

        loss = policy_loss + float(value_coef) * value_loss - float(entropy_coef) * entropy + float(belief_coef) * belief_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {
            'policy_loss': float(policy_loss.item()),
            'value_loss': float(value_loss.item()),
            'belief_loss': float(belief_loss.item()),
            'entropy': float(entropy.item()),
            'ratio': float(ratio.item()),
        }

    def fit(
        self,
        records,
        epochs=1,
        max_records=None,
        shuffle=False,
        verbose=False,
        policy_weight=1.0,
        value_weight=0.5,
        belief_weight=0.25,
        forced_policy_weight=0.0,
    ):
        stats = {
            'samples': 0,
            'action_loss': 0.0,
            'value_loss': 0.0,
            'action_hits': 0,
            'weighted_total_loss': 0.0,
            'decision_samples': 0,
            'decision_hits': 0,
            'forced_samples': 0,
        }
        limit = None if max_records is None else int(max_records)

        epoch_count = int(epochs)
        for epoch_index in range(epoch_count):
            epoch_samples = 0
            epoch_action_loss = 0.0
            epoch_value_loss = 0.0
            epoch_action_hits = 0
            epoch_weighted_total_loss = 0.0
            epoch_decision_samples = 0
            epoch_decision_hits = 0
            epoch_forced_samples = 0

            if shuffle:
                record_list = list(records)
                order = torch.randperm(len(record_list), generator=torch.Generator().manual_seed(self.seed + epoch_index)).tolist()
                iterable = [record_list[index] for index in order]
            else:
                iterable = records

            for record in iterable:
                if limit is not None and stats['samples'] >= limit:
                    if verbose and epoch_samples > 0:
                        epoch_accuracy = float(epoch_action_hits) / float(epoch_samples)
                        epoch_forced_rate = float(epoch_forced_samples) / float(epoch_samples)
                        if epoch_decision_samples > 0:
                            epoch_decision_accuracy = float(epoch_decision_hits) / float(epoch_decision_samples)
                            decision_segment = ' decision_accuracy=%.6f decision_samples=%d' % (
                                epoch_decision_accuracy,
                                epoch_decision_samples,
                            )
                        else:
                            decision_segment = ' decision_accuracy=n/a decision_samples=0'
                        print(
                            'epoch %d/%d samples=%d action_loss=%.6f value_loss=%.6f weighted_total_loss=%.6f action_accuracy=%.6f forced_rate=%.6f%s'
                            % (
                                epoch_index + 1,
                                epoch_count,
                                epoch_samples,
                                epoch_action_loss / float(epoch_samples),
                                epoch_value_loss / float(epoch_samples),
                                epoch_weighted_total_loss / float(epoch_samples),
                                epoch_accuracy,
                                epoch_forced_rate,
                                decision_segment,
                            )
                        )
                    return stats
                legal_actions = list(record.legal_actions or [])
                if record.action not in legal_actions:
                    legal_actions.append(record.action)
                is_decision_state = len(legal_actions) > 1
                record_policy_weight = float(policy_weight) if is_decision_state else float(forced_policy_weight)
                result = self.train_step(
                    record.features,
                    legal_actions,
                    record.action,
                    record.reward,
                    policy_weight=record_policy_weight,
                    value_weight=value_weight,
                    belief_weight=belief_weight,
                )
                stats['samples'] += 1
                stats['action_loss'] += result['action_loss']
                stats['value_loss'] += result['value_loss']
                stats['action_hits'] += result['action_hit']
                stats['weighted_total_loss'] += result.get('weighted_total_loss', 0.0)
                if result.get('is_decision_state'):
                    stats['decision_samples'] += 1
                    stats['decision_hits'] += result['action_hit']
                else:
                    stats['forced_samples'] += 1
                epoch_samples += 1
                epoch_action_loss += result['action_loss']
                epoch_value_loss += result['value_loss']
                epoch_action_hits += result['action_hit']
                epoch_weighted_total_loss += result.get('weighted_total_loss', 0.0)
                if result.get('is_decision_state'):
                    epoch_decision_samples += 1
                    epoch_decision_hits += result['action_hit']
                else:
                    epoch_forced_samples += 1

            if verbose and epoch_samples > 0:
                epoch_accuracy = float(epoch_action_hits) / float(epoch_samples)
                epoch_forced_rate = float(epoch_forced_samples) / float(epoch_samples)
                if epoch_decision_samples > 0:
                    epoch_decision_accuracy = float(epoch_decision_hits) / float(epoch_decision_samples)
                    decision_segment = ' decision_accuracy=%.6f decision_samples=%d' % (
                        epoch_decision_accuracy,
                        epoch_decision_samples,
                    )
                else:
                    decision_segment = ' decision_accuracy=n/a decision_samples=0'
                print(
                    'epoch %d/%d samples=%d action_loss=%.6f value_loss=%.6f weighted_total_loss=%.6f action_accuracy=%.6f forced_rate=%.6f%s'
                    % (
                        epoch_index + 1,
                        epoch_count,
                        epoch_samples,
                        epoch_action_loss / float(epoch_samples),
                        epoch_value_loss / float(epoch_samples),
                        epoch_weighted_total_loss / float(epoch_samples),
                        epoch_accuracy,
                        epoch_forced_rate,
                        decision_segment,
                    )
                )

        return stats

    def state_dict(self):
        return {key: tensor.detach().cpu().tolist() for key, tensor in self.model.state_dict().items()}

    def torch_state_dict(self):
        return {key: tensor.detach().cpu() for key, tensor in self.model.state_dict().items()}

    def load_state_dict(self, state_dict):
        current_state = self.model.state_dict()
        tensor_state = {}
        for key, value in state_dict.items():
            if key in current_state:
                tensor = torch.tensor(value, dtype=torch.float32, device=self.device)
                if tuple(tensor.shape) == tuple(current_state[key].shape):
                    tensor_state[key] = tensor
        current_state.update(tensor_state)
        self.model.load_state_dict(current_state, strict=False)

    def to_dict(self):
        return {
            'model_type': 'cnn_policy_value_v1',
            'action_space_size': self.action_space_size,
            'hidden_size': self.hidden_size,
            'learning_rate': self.learning_rate,
            'seed': self.seed,
            'backend': self.backend,
            'package_profile': self.package_profile,
            'state_dict': self.state_dict(),
            'metadata': self.metadata,
        }

    @classmethod
    def from_dict(cls, payload):
        if not isinstance(payload, dict):
            raise ValueError('Invalid checkpoint payload.')
        if _safe_int(payload.get('action_space_size', 0), 0) <= 0:
            raise ValueError('Invalid checkpoint payload: missing action_space_size.')
        model = cls(
            action_space_size=payload.get('action_space_size', 0),
            hidden_size=payload.get('hidden_size', 32),
            learning_rate=payload.get('learning_rate', 0.001),
            seed=payload.get('seed', 7),
            metadata=payload.get('metadata', {}),
        )
        state_payload = payload.get('state_dict') or payload.get('weights')
        if isinstance(state_payload, dict) and state_payload:
            try:
                model.load_state_dict(state_payload)
            except Exception:
                pass
        return model

    def save(self, path):
        parent = os.path.dirname(path)
        if parent and not os.path.exists(parent):
            os.makedirs(parent)
        with h5py.File(path, 'w') as handle:
            handle.attrs['model_type'] = self.to_dict()['model_type']
            handle.attrs['action_space_size'] = self.action_space_size
            handle.attrs['hidden_size'] = self.hidden_size
            handle.attrs['learning_rate'] = self.learning_rate
            handle.attrs['seed'] = self.seed
            handle.attrs['backend'] = self.backend
            handle.attrs['package_profile'] = json.dumps(self.package_profile, sort_keys=True)
            handle.attrs['metadata'] = json.dumps(self.metadata, sort_keys=True)

            state_group = handle.create_group('state_dict')
            for key, tensor in self.torch_state_dict().items():
                state_group.create_dataset(key, data=tensor.numpy())

    @classmethod
    def load(cls, path):
        with open(path, 'rb') as handle:
            header = handle.read(8)

        if header == b'\x89HDF\r\n\x1a\n':
            with h5py.File(path, 'r') as handle:
                state_dict = {}
                if 'state_dict' in handle:
                    state_group = cast(Any, handle['state_dict'])
                    for key in state_group.keys():
                        state_dict[key] = np.asarray(state_group[key][()]).tolist()
                payload = {
                    'action_space_size': int(handle.attrs.get('action_space_size', 0)),
                    'hidden_size': int(handle.attrs.get('hidden_size', 32)),
                    'learning_rate': float(handle.attrs.get('learning_rate', 0.001)),
                    'seed': int(handle.attrs.get('seed', 7)),
                    'backend': handle.attrs.get('backend', 'torch'),
                    'package_profile': json.loads(handle.attrs.get('package_profile', '{}')),
                    'state_dict': state_dict,
                    'metadata': json.loads(handle.attrs.get('metadata', '{}')),
                }
            return cls.from_dict(payload)

        with open(path, 'r', encoding='utf-8') as handle:
            payload = json.load(handle)
        return cls.from_dict(payload)
