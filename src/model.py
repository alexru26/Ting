import json
import math
import os
from contextlib import nullcontext
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
CHANNEL_COUNT = 11
TEMPORAL_FEATURES_PER_OPPONENT = 7
TEMPORAL_OPPONENT_COUNT = 3
EXTRA_META_COUNT = 1 + 1 + 3 + (TEMPORAL_FEATURES_PER_OPPONENT * TEMPORAL_OPPONENT_COUNT)
META_COUNT = 8 + 3 + len(EVENT_VOCAB) + len(REQUEST_VOCAB) + EXTRA_META_COUNT
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


def resolve_device(requested_device='cpu'):
    requested = str(requested_device or 'cpu').strip().lower()
    if not requested:
        requested = 'cpu'

    if requested == 'auto':
        if torch.cuda.is_available():
            return torch.device('cuda'), requested, 'cuda'
        return torch.device('cpu'), requested, 'cpu'

    if requested.startswith('cuda'):
        if torch.cuda.is_available():
            try:
                return torch.device(requested), requested, requested
            except Exception:
                return torch.device('cuda'), requested, 'cuda'
        return torch.device('cpu'), requested, 'cpu'

    return torch.device('cpu'), requested, 'cpu'


class CnnCore(nn.Module):
    def __init__(self, hidden_size, family_size, arg_size):
        super().__init__()
        self.conv1 = nn.Conv1d(CHANNEL_COUNT, 24, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(24, 24, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(24, 24, kernel_size=3, padding=1)
        self.meta_proj = nn.Linear(META_COUNT, 48)
        self.hidden_fuse = nn.Linear(24 + 48, hidden_size)

        self.temporal_proj = nn.Linear(TEMPORAL_FEATURES_PER_OPPONENT * TEMPORAL_OPPONENT_COUNT, 24)
        self.post_temporal = nn.Linear(hidden_size + 24, hidden_size)

        self.belief_head = nn.Linear(hidden_size, TILE_COUNT)
        self.belief_proj = nn.Linear(TILE_COUNT, 16)
        self.output_fuse = nn.Linear(hidden_size + 16, hidden_size)

        self.family_head = nn.Linear(hidden_size, family_size)
        self.family_embedding = nn.Embedding(family_size, hidden_size)

        self.arg1_head = nn.Linear(hidden_size * 2, arg_size)
        self.arg2_head = nn.Linear(hidden_size * 2, arg_size)

        self.value_hidden = nn.Linear(hidden_size, hidden_size)
        self.value_head = nn.Linear(hidden_size, 1)
        self.aux_value_head = nn.Linear(hidden_size, 1)
        self.efficiency_bonus_head = nn.Linear(hidden_size, 1)

    def forward(self, tile_tensor, meta_tensor):
        tile_features = F.relu(self.conv1(tile_tensor))
        tile_features = F.relu(self.conv2(tile_features))
        tile_features = F.relu(self.conv3(tile_features))
        tile_features = F.adaptive_avg_pool1d(tile_features, 1).squeeze(-1)

        meta_features = F.relu(self.meta_proj(meta_tensor))
        hidden = F.relu(self.hidden_fuse(torch.cat([tile_features, meta_features], dim=-1)))

        temporal_width = TEMPORAL_FEATURES_PER_OPPONENT * TEMPORAL_OPPONENT_COUNT
        temporal_slice = meta_tensor[:, -temporal_width:]
        temporal_features = F.relu(self.temporal_proj(temporal_slice))
        hidden = F.relu(self.post_temporal(torch.cat([hidden, temporal_features], dim=-1)))

        belief_logits = self.belief_head(hidden)
        belief_probs = torch.softmax(belief_logits, dim=-1)
        belief_context = F.relu(self.belief_proj(belief_probs))
        hidden = F.relu(self.output_fuse(torch.cat([hidden, belief_context], dim=-1)))

        value_hidden = F.relu(self.value_hidden(hidden))

        return {
            'hidden': hidden,
            'belief_logits': belief_logits,
            'belief_probs': belief_probs,
            'family_logits': self.family_head(hidden),
            'value': self.value_head(value_hidden),
            'aux_value': self.aux_value_head(value_hidden),
            'efficiency_bonus': self.efficiency_bonus_head(hidden),
        }

    def conditioned_arg_logits(self, hidden, family_indices):
        family_embed = self.family_embedding(family_indices)
        conditioned = torch.cat([hidden, family_embed], dim=-1)
        return self.arg1_head(conditioned), self.arg2_head(conditioned)


class CnnPolicyValueModel:
    def __init__(
        self,
        action_space_size,
        hidden_size=32,
        learning_rate=0.001,
        seed=7,
        state_dict=None,
        metadata=None,
        device='cpu',
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
        resolved_device, requested_device_name, resolved_device_name = resolve_device(device)
        self.device = resolved_device
        self.requested_device = requested_device_name
        self.resolved_device = resolved_device_name
        self.amp_enabled = bool(self.device.type == 'cuda')
        self.amp_dtype = torch.bfloat16
        try:
            self.grad_scaler = torch.amp.GradScaler('cuda', enabled=self.amp_enabled)
        except Exception:
            self.grad_scaler = torch.cuda.amp.GradScaler(enabled=self.amp_enabled)
        torch.manual_seed(self.seed)

        self.calibration_temperature = max(1e-3, _safe_float(self.metadata.get('calibration_temperature', 1.0), 1.0))

        self.model = CnnCore(
            hidden_size=self.hidden_size,
            family_size=len(self.family_vocab),
            arg_size=len(self.arg_vocab),
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        if state_dict is not None:
            self.load_state_dict(state_dict)

    def _autocast_context(self):
        if not self.amp_enabled:
            return nullcontext()
        return torch.autocast(device_type='cuda', dtype=self.amp_dtype)

    def _zero_tile_tensor(self):
        return torch.zeros((CHANNEL_COUNT, TILE_COUNT), dtype=torch.float32, device=self.device)

    def _channel_tensor(self, values):
        tensor = torch.zeros((TILE_COUNT,), dtype=torch.float32, device=self.device)
        if isinstance(values, list):
            limit = min(len(values), TILE_COUNT)
            if limit > 0:
                tensor[:limit] = torch.tensor(values[:limit], dtype=torch.float32, device=self.device)
        return tensor

    def _one_hot(self, value, vocabulary):
        vector = [0.0] * len(vocabulary)
        try:
            index = vocabulary.index(value)
        except ValueError:
            index = 0
        vector[index] = 1.0
        return vector

    def _safe_ratio(self, value, divisor):
        divisor = float(divisor)
        if divisor <= 0.0:
            return 0.0
        return max(0.0, min(1.0, float(value) / divisor))

    def _extract_temporal_meta(self, features):
        temporal = []
        temporal_payload = features.get('opponent_temporal') if isinstance(features, dict) else None
        if not isinstance(temporal_payload, list):
            temporal_payload = []

        for row in temporal_payload[:TEMPORAL_OPPONENT_COUNT]:
            if not isinstance(row, dict):
                row = {}
            suits = row.get('suit_ratios', {}) if isinstance(row.get('suit_ratios', {}), dict) else {}
            temporal.extend(
                [
                    self._safe_ratio(row.get('full_history_length', 0.0), 40.0),
                    self._safe_ratio(row.get('pack_count', 0.0), 4.0),
                    _safe_float(row.get('recent_honor_ratio', 0.0), 0.0),
                    _safe_float(row.get('honor_ratio', 0.0), 0.0),
                    _safe_float(suits.get('W', 0.0), 0.0),
                    _safe_float(suits.get('B', 0.0), 0.0),
                    _safe_float(suits.get('T', 0.0), 0.0),
                ]
            )

        while len(temporal) < TEMPORAL_FEATURES_PER_OPPONENT * TEMPORAL_OPPONENT_COUNT:
            temporal.append(0.0)
        return temporal[: TEMPORAL_FEATURES_PER_OPPONENT * TEMPORAL_OPPONENT_COUNT]

    def _encode_features(self, features):
        if not isinstance(features, dict):
            features = {}

        tile_tensor = self._zero_tile_tensor()

        tile_tensor[0] = self._channel_tensor(features.get('hand_counts_norm', features.get('hand_counts', [])))
        tile_tensor[1] = self._channel_tensor(features.get('seen_counts_norm', features.get('seen_counts', [])))
        tile_tensor[2] = self._channel_tensor(features.get('self_discard_counts_norm', features.get('self_discard_counts', [])))
        tile_tensor[3] = self._channel_tensor(features.get('pack_counts_norm', features.get('pack_counts', [])))

        opponent_channels = features.get('opponent_discard_counts_norm', features.get('opponent_discard_counts', []))
        if isinstance(opponent_channels, list):
            for offset, channel in enumerate(opponent_channels[:3], start=4):
                tile_tensor[offset] = self._channel_tensor(channel if isinstance(channel, list) else [])

        tile_tensor[7] = self._channel_tensor(features.get('hand_counts', []))
        tile_tensor[8] = self._channel_tensor(features.get('seen_counts', []))

        shanten_norm = _safe_float(features.get('hand_shanten_norm', 0.0), 0.0)
        acceptancy_norm = _safe_float(features.get('acceptancy_norm', 0.0), 0.0)
        tile_tensor[9] = torch.full((TILE_COUNT,), shanten_norm, dtype=torch.float32, device=self.device)
        tile_tensor[10] = torch.full((TILE_COUNT,), acceptancy_norm, dtype=torch.float32, device=self.device)

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

        meta_values.extend(
            [
                _safe_float(features.get('schema_version', 0.0), 0.0),
                _safe_float(features.get('hand_shanten_norm', 0.0), 0.0),
                _safe_float(features.get('acceptancy_norm', 0.0), 0.0),
                _safe_float((features.get('action_efficiency_deltas') or {}).get('PLAY', 0.0), 0.0),
                _safe_float((features.get('action_efficiency_deltas') or {}).get('GANG', 0.0), 0.0),
            ]
        )
        meta_values.extend(self._extract_temporal_meta(features))

        while len(meta_values) < META_COUNT:
            meta_values.append(0.0)

        meta_tensor = torch.tensor(meta_values[:META_COUNT], dtype=torch.float32, device=self.device)
        return tile_tensor.unsqueeze(0), meta_tensor.unsqueeze(0)

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
        return (
            self.family_index.get(family, 0),
            self.arg_index.get(arg1, self.arg_index[NONE_TOKEN]),
            self.arg_index.get(arg2, self.arg_index[NONE_TOKEN]),
        )

    def _conditioned_arg_logits(self, outputs, family_index, batch_index=0):
        family_tensor = torch.tensor([int(family_index)], dtype=torch.long, device=self.device)
        hidden = outputs['hidden'][int(batch_index): int(batch_index) + 1]
        return self.model.conditioned_arg_logits(hidden, family_tensor)

    def _action_score(self, outputs, action, batch_index=0):
        family_index, arg1_index, arg2_index = self._action_indices(action)
        arg1_logits, arg2_logits = self._conditioned_arg_logits(outputs, family_index, batch_index=batch_index)
        score = outputs['family_logits'][int(batch_index), family_index] + arg1_logits[0, arg1_index] + arg2_logits[0, arg2_index]
        return score

    def _belief_bonus(self, outputs, action, batch_index=0):
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
        return float(belief_probs[int(batch_index), tile_index].item())

    def _efficiency_prior(self, features, action):
        if not isinstance(features, dict):
            return 0.0
        deltas = features.get('action_efficiency_deltas', {})
        if not isinstance(deltas, dict):
            return 0.0
        family = action.split()[0] if action else 'PASS'
        return _safe_float(deltas.get(family, 0.0), 0.0)

    def _legal_action_scores(self, outputs, legal_actions, features=None, belief_weight=0.0, efficiency_weight=0.0, batch_index=0):
        scores = []
        efficiency_tensor = outputs.get('efficiency_bonus', torch.tensor([[0.0]], device=self.device))
        efficiency_bonus = _safe_float(efficiency_tensor[int(batch_index), 0].item(), 0.0)
        for action in legal_actions:
            score = self._action_score(outputs, action, batch_index=batch_index)
            if belief_weight:
                score = score + float(belief_weight) * self._belief_bonus(outputs, action, batch_index=batch_index)
            if efficiency_weight:
                score = score + float(efficiency_weight) * efficiency_bonus * self._efficiency_prior(features, action)
            scores.append(score)
        return torch.stack(scores, dim=0)

    def set_calibration_temperature(self, temperature):
        self.calibration_temperature = max(1e-3, _safe_float(temperature, 1.0))
        self.metadata['calibration_temperature'] = float(self.calibration_temperature)

    def _temperature_adjusted_scores(self, scores, temperature=None):
        if temperature is None:
            temperature = self.calibration_temperature
        temperature = max(1e-3, _safe_float(temperature, 1.0))
        return scores / temperature

    def policy_info_from_features(self, features, legal_actions, belief_weight=0.0, efficiency_weight=0.0, temperature=None):
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
                'aux_value': 0.0,
                'efficiency_bonus': 0.0,
            }

        self.model.eval()
        with torch.no_grad():
            tile_tensor, meta_tensor = self._encode_features(features)
            outputs = self.model(tile_tensor, meta_tensor)
            raw_scores = self._legal_action_scores(
                outputs,
                legal_actions,
                features=features,
                belief_weight=belief_weight,
                efficiency_weight=efficiency_weight,
            )
            scores = self._temperature_adjusted_scores(raw_scores, temperature=temperature)
            log_probabilities = torch.log_softmax(scores, dim=0)
            probabilities = torch.softmax(scores, dim=0)
            entropy = float((-(probabilities * log_probabilities)).sum().item())
            value = float(outputs['value'][0, 0].item())
            aux_value = float(outputs['aux_value'][0, 0].item())
            efficiency_bonus = float(outputs['efficiency_bonus'][0, 0].item())
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
            'aux_value': aux_value,
            'efficiency_bonus': efficiency_bonus,
        }

    def sample_action_from_features(self, features, legal_actions, temperature=1.0, greedy=False, belief_weight=0.0, efficiency_weight=0.0):
        info = self.policy_info_from_features(
            features,
            legal_actions,
            belief_weight=belief_weight,
            efficiency_weight=efficiency_weight,
            temperature=temperature,
        )
        actions = info['actions']
        if not actions:
            return None, info

        if greedy:
            index = max(range(len(actions)), key=lambda idx: (info['probabilities'][idx], actions[idx]))
        else:
            probabilities = torch.tensor(info['probabilities'], dtype=torch.float32, device=self.device)
            if float(temperature) != 1.0:
                logits = torch.log(torch.clamp(probabilities, min=1e-12)) / max(1e-3, float(temperature))
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

    def action_distribution_from_features(self, features, legal_actions, belief_weight=0.0, efficiency_weight=0.0, temperature=None):
        info = self.policy_info_from_features(
            features,
            legal_actions,
            belief_weight=belief_weight,
            efficiency_weight=efficiency_weight,
            temperature=temperature,
        )
        return {action: probability for action, probability in zip(info['actions'], info['probabilities'])}

    def decode_conditioned_action_from_features(self, features, legal_actions=None, deterministic=True, temperature=1.0):
        self.model.eval()
        with torch.no_grad():
            tile_tensor, meta_tensor = self._encode_features(features)
            outputs = self.model(tile_tensor, meta_tensor)
            family_logits = outputs['family_logits'][0]
            if deterministic:
                family_index = int(torch.argmax(family_logits).item())
            else:
                family_probs = torch.softmax(family_logits / max(1e-3, float(temperature)), dim=0)
                family_index = int(torch.multinomial(family_probs, 1).item())

            arg1_logits, arg2_logits = self._conditioned_arg_logits(outputs, family_index)
            arg1_index = int(torch.argmax(arg1_logits[0]).item())
            arg2_index = int(torch.argmax(arg2_logits[0]).item())

            family = self.family_vocab[family_index]
            arg1 = self.arg_vocab[arg1_index]
            arg2 = self.arg_vocab[arg2_index]

            if family in ('PASS', 'HU'):
                candidate = family
            elif family == 'GANG' and arg1 == NONE_TOKEN:
                candidate = 'GANG'
            elif family in ('PLAY', 'GANG', 'BUGANG', 'PENG'):
                candidate = '%s %s' % (family, arg1)
            elif family == 'CHI':
                candidate = '%s %s %s' % (family, arg1, arg2)
            else:
                candidate = 'PASS'

            if legal_actions:
                if candidate in legal_actions:
                    return candidate
                info = self.policy_info_from_features(features, legal_actions)
                if info['actions']:
                    best_idx = max(range(len(info['actions'])), key=lambda idx: info['probabilities'][idx])
                    return info['actions'][best_idx]
                return None
            return candidate

    def choose_action_from_features(self, features, legal_actions, belief_weight=0.0, efficiency_weight=0.0, temperature=None):
        distribution = self.action_distribution_from_features(
            features,
            legal_actions,
            belief_weight=belief_weight,
            efficiency_weight=efficiency_weight,
            temperature=temperature,
        )
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

    def _belief_consistency_penalty(self, outputs, features, batch_index=0):
        if not isinstance(features, dict):
            return torch.tensor(0.0, dtype=torch.float32, device=self.device)
        seen = features.get('seen_counts', [])
        if not isinstance(seen, list) or not seen:
            return torch.tensor(0.0, dtype=torch.float32, device=self.device)

        indices = []
        for idx, count in enumerate(seen[:TILE_COUNT]):
            if _safe_float(count, 0.0) >= 4.0:
                indices.append(idx)

        if not indices:
            return torch.tensor(0.0, dtype=torch.float32, device=self.device)

        probs = outputs['belief_probs'][int(batch_index), indices]
        return probs.mean()

    def _encode_batch_features(self, features_list):
        tile_tensors = []
        meta_tensors = []
        for features in features_list:
            tile_tensor, meta_tensor = self._encode_features(features)
            tile_tensors.append(tile_tensor.squeeze(0))
            meta_tensors.append(meta_tensor.squeeze(0))
        return torch.stack(tile_tensors, dim=0), torch.stack(meta_tensors, dim=0)

    def train_batch_step(
        self,
        batch_records,
        policy_weight=1.0,
        value_weight=0.5,
        belief_weight=0.25,
        forced_policy_weight=0.0,
        aux_value_weight=0.15,
        efficiency_weight=0.1,
        belief_consistency_weight=0.1,
    ):
        if not batch_records:
            return {
                'samples': 0,
                'action_loss': 0.0,
                'value_loss': 0.0,
                'aux_value_loss': 0.0,
                'belief_loss': 0.0,
                'belief_consistency_loss': 0.0,
                'weighted_total_loss': 0.0,
                'action_hits': 0,
                'decision_samples': 0,
                'decision_hits': 0,
                'forced_samples': 0,
                'efficiency_bonus': 0.0,
            }

        prepared = []
        feature_list = []
        for record in batch_records:
            legal_actions = list(record.legal_actions or [])
            if record.action not in legal_actions:
                legal_actions.append(record.action)
            is_decision_state = len(legal_actions) > 1
            record_policy_weight = float(policy_weight) if is_decision_state else float(forced_policy_weight)
            prepared.append((record, legal_actions, is_decision_state, record_policy_weight))
            feature_list.append(record.features)

        self.model.train()
        with self._autocast_context():
            tile_tensor, meta_tensor = self._encode_batch_features(feature_list)
            outputs = self.model(tile_tensor, meta_tensor)

            sample_losses = []
            sample_action_losses = []
            sample_value_losses = []
            sample_aux_value_losses = []
            sample_belief_losses = []
            sample_belief_consistency_losses = []
            action_hits = 0
            decision_samples = 0
            decision_hits = 0
            forced_samples = 0
            efficiency_bonus_sum = 0.0

            for idx, (record, legal_actions, is_decision_state, record_policy_weight) in enumerate(prepared):
                scores = self._legal_action_scores(
                    outputs,
                    legal_actions,
                    features=record.features,
                    belief_weight=0.0,
                    efficiency_weight=efficiency_weight,
                    batch_index=idx,
                ).unsqueeze(0)
                target_index = torch.tensor([legal_actions.index(record.action)], dtype=torch.long, device=self.device)
                action_loss = F.cross_entropy(scores, target_index)

                reward_value = float(_safe_float(record.reward, 0.0))
                value_target = torch.tensor([[reward_value]], dtype=torch.float32, device=self.device)
                value_pred = outputs['value'][idx: idx + 1]
                aux_pred = outputs['aux_value'][idx: idx + 1]
                value_loss = F.mse_loss(value_pred, value_target)
                aux_target = torch.tanh(value_target)
                aux_value_loss = F.mse_loss(aux_pred, aux_target)

                belief_target = torch.tensor([self._belief_target_from_features(record.features)], dtype=torch.float32, device=self.device)
                belief_pred = outputs['belief_probs'][idx: idx + 1]
                belief_loss = F.kl_div(torch.log(torch.clamp(belief_pred, min=1e-12)), belief_target, reduction='batchmean')
                belief_consistency_loss = self._belief_consistency_penalty(outputs, record.features, batch_index=idx)

                sample_loss = (
                    float(record_policy_weight) * action_loss
                    + float(value_weight) * value_loss
                    + float(aux_value_weight) * aux_value_loss
                    + float(belief_weight) * belief_loss
                    + float(belief_consistency_weight) * belief_consistency_loss
                )

                with torch.no_grad():
                    probabilities = torch.softmax(scores.squeeze(0), dim=0)
                    action_hit = int(int(torch.argmax(probabilities).item()) == int(target_index.item()))
                    action_hits += action_hit
                    if is_decision_state:
                        decision_samples += 1
                        decision_hits += action_hit
                    else:
                        forced_samples += 1

                sample_losses.append(sample_loss)
                sample_action_losses.append(float(action_loss.item()))
                sample_value_losses.append(float(value_loss.item()))
                sample_aux_value_losses.append(float(aux_value_loss.item()))
                sample_belief_losses.append(float(belief_loss.item()))
                sample_belief_consistency_losses.append(float(belief_consistency_loss.item()))
                efficiency_bonus_sum += float(outputs['efficiency_bonus'][idx, 0].detach().item())

            batch_loss = torch.stack(sample_losses, dim=0).mean()

        self.optimizer.zero_grad()
        if self.amp_enabled:
            self.grad_scaler.scale(batch_loss).backward()
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            batch_loss.backward()
            self.optimizer.step()

        return {
            'samples': len(batch_records),
            'action_loss': float(sum(sample_action_losses)),
            'value_loss': float(sum(sample_value_losses)),
            'aux_value_loss': float(sum(sample_aux_value_losses)),
            'belief_loss': float(sum(sample_belief_losses)),
            'belief_consistency_loss': float(sum(sample_belief_consistency_losses)),
            'weighted_total_loss': float(sum(float(loss.item()) for loss in sample_losses)),
            'action_hits': int(action_hits),
            'decision_samples': int(decision_samples),
            'decision_hits': int(decision_hits),
            'forced_samples': int(forced_samples),
            'efficiency_bonus': float(efficiency_bonus_sum),
        }

    def train_step(
        self,
        features,
        legal_actions,
        action,
        reward,
        policy_weight=1.0,
        value_weight=0.5,
        belief_weight=0.25,
        aux_value_weight=0.15,
        efficiency_weight=0.1,
        belief_consistency_weight=0.1,
    ):
        if not legal_actions:
            legal_actions = [action]
        if action not in legal_actions:
            legal_actions = list(legal_actions) + [action]
        legal_action_count = len(legal_actions)
        is_decision_state = legal_action_count > 1

        self.model.train()
        with self._autocast_context():
            tile_tensor, meta_tensor = self._encode_features(features)
            outputs = self.model(tile_tensor, meta_tensor)

            scores = self._legal_action_scores(outputs, legal_actions, features=features, belief_weight=0.0, efficiency_weight=efficiency_weight).unsqueeze(0)
            target_index = torch.tensor([legal_actions.index(action)], dtype=torch.long, device=self.device)
            action_loss = F.cross_entropy(scores, target_index)

            value_target = torch.tensor([[float(_safe_float(reward, 0.0))]], dtype=torch.float32, device=self.device)
            value_loss = F.mse_loss(outputs['value'], value_target)
            aux_target = torch.tanh(value_target)
            aux_value_loss = F.mse_loss(outputs['aux_value'], aux_target)

            belief_target = torch.tensor([self._belief_target_from_features(features)], dtype=torch.float32, device=self.device)
            belief_loss = F.kl_div(torch.log(torch.clamp(outputs['belief_probs'], min=1e-12)), belief_target, reduction='batchmean')
            belief_consistency_loss = self._belief_consistency_penalty(outputs, features)

            loss = (
                float(policy_weight) * action_loss
                + float(value_weight) * value_loss
                + float(aux_value_weight) * aux_value_loss
                + float(belief_weight) * belief_loss
                + float(belief_consistency_weight) * belief_consistency_loss
            )
        self.optimizer.zero_grad()
        if self.amp_enabled:
            self.grad_scaler.scale(loss).backward()
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            loss.backward()
            self.optimizer.step()

        with torch.no_grad():
            probabilities = torch.softmax(scores.squeeze(0), dim=0)
            action_hit = int(int(torch.argmax(probabilities).item()) == int(target_index.item()))

        return {
            'action_loss': float(action_loss.item()),
            'value_loss': float(value_loss.item()),
            'aux_value_loss': float(aux_value_loss.item()),
            'belief_loss': float(belief_loss.item()),
            'belief_consistency_loss': float(belief_consistency_loss.item()),
            'weighted_total_loss': float(loss.item()),
            'action_hit': action_hit,
            'is_decision_state': bool(is_decision_state),
            'legal_action_count': int(legal_action_count),
            'efficiency_bonus': float(outputs['efficiency_bonus'][0, 0].detach().item()),
        }

    def ppo_train_step(self, features, legal_actions, action, advantage, return_target, old_log_prob=None, clip_range=0.2, entropy_coef=0.01, value_coef=0.5, belief_coef=0.25):
        if not legal_actions:
            legal_actions = [action]
        if action not in legal_actions:
            legal_actions = list(legal_actions) + [action]

        self.model.train()
        with self._autocast_context():
            tile_tensor, meta_tensor = self._encode_features(features)
            outputs = self.model(tile_tensor, meta_tensor)

            scores = self._legal_action_scores(outputs, legal_actions, features=features)
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
        if self.amp_enabled:
            self.grad_scaler.scale(loss).backward()
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
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
        aux_value_weight=0.15,
        efficiency_weight=0.1,
        belief_consistency_weight=0.1,
        batch_size=1,
    ):
        stats = {
            'samples': 0,
            'action_loss': 0.0,
            'value_loss': 0.0,
            'aux_value_loss': 0.0,
            'belief_loss': 0.0,
            'belief_consistency_loss': 0.0,
            'efficiency_bonus': 0.0,
            'action_hits': 0,
            'weighted_total_loss': 0.0,
            'decision_samples': 0,
            'decision_hits': 0,
            'forced_samples': 0,
        }
        limit = None if max_records is None else int(max_records)

        epoch_count = int(epochs)
        for epoch_index in range(epoch_count):
            if verbose:
                print('starting epoch %d/%d' % (epoch_index + 1, epoch_count))

            epoch_start_samples = stats['samples']
            epoch_start_action_loss = stats['action_loss']
            epoch_start_value_loss = stats['value_loss']
            epoch_start_weighted_total_loss = stats['weighted_total_loss']
            epoch_start_action_hits = stats['action_hits']

            if shuffle:
                record_list = list(records)
                order = torch.randperm(len(record_list), generator=torch.Generator().manual_seed(self.seed + epoch_index)).tolist()
                iterable = [record_list[index] for index in order]
            else:
                iterable = records

            effective_batch_size = max(1, int(batch_size))
            pending_batch = []
            epoch_processed = 0

            def _flush_pending_batch(current_batch):
                if not current_batch:
                    return
                if effective_batch_size <= 1:
                    record = current_batch[0]
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
                        aux_value_weight=aux_value_weight,
                        efficiency_weight=efficiency_weight,
                        belief_consistency_weight=belief_consistency_weight,
                    )
                    stats['samples'] += 1
                    stats['action_loss'] += result['action_loss']
                    stats['value_loss'] += result['value_loss']
                    stats['aux_value_loss'] += result.get('aux_value_loss', 0.0)
                    stats['belief_loss'] += result.get('belief_loss', 0.0)
                    stats['belief_consistency_loss'] += result.get('belief_consistency_loss', 0.0)
                    stats['efficiency_bonus'] += result.get('efficiency_bonus', 0.0)
                    stats['action_hits'] += result['action_hit']
                    stats['weighted_total_loss'] += result.get('weighted_total_loss', 0.0)
                    if result.get('is_decision_state'):
                        stats['decision_samples'] += 1
                        stats['decision_hits'] += result['action_hit']
                    else:
                        stats['forced_samples'] += 1
                    return

                batch_result = self.train_batch_step(
                    current_batch,
                    policy_weight=policy_weight,
                    value_weight=value_weight,
                    belief_weight=belief_weight,
                    forced_policy_weight=forced_policy_weight,
                    aux_value_weight=aux_value_weight,
                    efficiency_weight=efficiency_weight,
                    belief_consistency_weight=belief_consistency_weight,
                )
                stats['samples'] += batch_result.get('samples', 0)
                stats['action_loss'] += batch_result.get('action_loss', 0.0)
                stats['value_loss'] += batch_result.get('value_loss', 0.0)
                stats['aux_value_loss'] += batch_result.get('aux_value_loss', 0.0)
                stats['belief_loss'] += batch_result.get('belief_loss', 0.0)
                stats['belief_consistency_loss'] += batch_result.get('belief_consistency_loss', 0.0)
                stats['efficiency_bonus'] += batch_result.get('efficiency_bonus', 0.0)
                stats['action_hits'] += batch_result.get('action_hits', 0)
                stats['weighted_total_loss'] += batch_result.get('weighted_total_loss', 0.0)
                stats['decision_samples'] += batch_result.get('decision_samples', 0)
                stats['decision_hits'] += batch_result.get('decision_hits', 0)
                stats['forced_samples'] += batch_result.get('forced_samples', 0)

            for record in iterable:
                if limit is not None and (stats['samples'] + len(pending_batch)) >= limit:
                    if verbose:
                        print('stopping early at max_records=%d' % limit)
                    break

                pending_batch.append(record)
                epoch_processed += 1
                if len(pending_batch) >= effective_batch_size:
                    _flush_pending_batch(pending_batch)
                    pending_batch = []

                if verbose and (epoch_processed == 1 or epoch_processed % 10000 == 0):
                    print('epoch %d/%d processed_records=%d total_samples=%d' % (epoch_index + 1, epoch_count, epoch_processed, stats['samples']))

            if pending_batch:
                _flush_pending_batch(pending_batch)

            if verbose and stats['samples'] > 0:
                epoch_samples = stats['samples'] - epoch_start_samples
                if epoch_samples <= 0:
                    print('epoch %d/%d had no records to train on' % (epoch_index + 1, epoch_count))
                    continue
                samples = float(epoch_samples)
                print(
                    'epoch %d/%d samples=%d action_loss=%.6f value_loss=%.6f weighted_total_loss=%.6f action_accuracy=%.6f'
                    % (
                        epoch_index + 1,
                        epoch_count,
                        epoch_samples,
                        (stats['action_loss'] - epoch_start_action_loss) / samples,
                        (stats['value_loss'] - epoch_start_value_loss) / samples,
                        (stats['weighted_total_loss'] - epoch_start_weighted_total_loss) / samples,
                        float(stats['action_hits'] - epoch_start_action_hits) / samples,
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
            'device': self.resolved_device,
        }

    @classmethod
    def from_dict(cls, payload, device='cpu'):
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
            device=device,
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
    def load(cls, path, device='cpu'):
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
            return cls.from_dict(payload, device=device)

        with open(path, 'r', encoding='utf-8') as handle:
            payload = json.load(handle)
        return cls.from_dict(payload, device=device)
