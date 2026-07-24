"""Policy-value network for the Ting Mahjong agent.

Architecture (v2):

- Input A: `PLANE_COUNT x 34` tile planes, reshaped onto a 4x9 grid
  (rows: W, B, T, honors with 2 padded cells) so 2D convolutions see both
  rank adjacency (chi shapes) and same-rank-across-suit structure.
- Input B: `META_COUNT` scalar context vector.
- Trunk: conv stem + residual blocks, flattened and fused with the meta
  embedding into a shared hidden state.
- Heads: action-family logits (7), family-conditioned argument logits
  (2 x 35, tile vocabulary + NONE), and a scalar value head.

An action `FAMILY [arg1] [arg2]` is scored as
`family_logit + arg1_logit + arg2_logit`, and the policy is a softmax over
the *legal* actions only. Training uses exactly the same masked scoring, so
the learning target matches the runtime decision rule (TODO 8: this is the
"multiple models per action" idea realised as one shared trunk with
per-family conditioned heads - a single checkpoint, which the Botzone
bundle requires, with per-family specialisation).
"""

import json
import os

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from features import FEATURE_SCHEMA_VERSION, META_COUNT, PLANE_COUNT
from tiles import ALL_TILES

FAMILY_LABELS = ['PASS', 'HU', 'GANG', 'PLAY', 'BUGANG', 'PENG', 'CHI']
NONE_TOKEN = '<NONE>'
ARG_VOCAB = list(ALL_TILES) + [NONE_TOKEN]

TILE_COUNT = len(ALL_TILES)
GRID_ROWS = 4
GRID_COLS = 9
GRID_CELLS = GRID_ROWS * GRID_COLS

MODEL_TYPE = 'ting_cnn_v3'

_FAMILY_INDEX = {label: idx for idx, label in enumerate(FAMILY_LABELS)}
_ARG_INDEX = {label: idx for idx, label in enumerate(ARG_VOCAB)}
_NONE_INDEX = _ARG_INDEX[NONE_TOKEN]


def resolve_device(requested_device='cpu'):
    requested = str(requested_device or 'cpu').strip().lower() or 'cpu'
    if requested == 'auto':
        if torch.cuda.is_available():
            return torch.device('cuda'), requested, 'cuda'
        return torch.device('cpu'), requested, 'cpu'
    if requested.startswith('cuda'):
        if torch.cuda.is_available():
            return torch.device(requested), requested, requested
        return torch.device('cpu'), requested, 'cpu'
    return torch.device('cpu'), requested, 'cpu'


def encode_action(action):
    """Map an action string to (family_index, arg1_index, arg2_index)."""
    parts = str(action).split()
    if not parts or parts[0] not in _FAMILY_INDEX:
        raise ValueError('Unknown action family: %r' % (action,))
    family = _FAMILY_INDEX[parts[0]]
    arg1 = _ARG_INDEX.get(parts[1], _NONE_INDEX) if len(parts) > 1 else _NONE_INDEX
    arg2 = _ARG_INDEX.get(parts[2], _NONE_INDEX) if len(parts) > 2 else _NONE_INDEX
    return family, arg1, arg2


def _print_progress_bar(prefix, current, total, width=32):
    total_value = max(1, int(total))
    current_value = max(0, min(int(current), total_value))
    ratio = float(current_value) / float(total_value)
    filled = int(float(width) * ratio)
    bar = ('#' * filled) + ('-' * (width - filled))
    print('\r%s [%s] %d/%d (%.1f%%)' % (prefix, bar, current_value, total_value, ratio * 100.0), end='', flush=True)
    if current_value >= total_value:
        print('')


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        return F.relu(x + out)


class CnnCore(nn.Module):
    def __init__(self, channels, blocks, hidden_size, family_embedding_size=32):
        super().__init__()
        self.stem = nn.Conv2d(PLANE_COUNT, channels, kernel_size=3, padding=1)
        self.blocks = nn.ModuleList([ResidualBlock(channels) for _ in range(blocks)])
        self.meta_proj = nn.Linear(META_COUNT, 64)
        self.trunk = nn.Linear(channels * GRID_CELLS + 64, hidden_size)

        self.family_head = nn.Linear(hidden_size, len(FAMILY_LABELS))
        self.family_embedding = nn.Embedding(len(FAMILY_LABELS), family_embedding_size)
        self.arg1_head = nn.Linear(hidden_size + family_embedding_size, len(ARG_VOCAB))
        self.arg2_head = nn.Linear(hidden_size + family_embedding_size, len(ARG_VOCAB))

        self.value_hidden = nn.Linear(hidden_size, hidden_size // 2)
        self.value_head = nn.Linear(hidden_size // 2, 1)
        # Auxiliary quality target: predicts whether this trajectory ends as a
        # positive-score outcome, separating strong actions from merely legal ones.
        self.win_head = nn.Linear(hidden_size // 2, 1)

    def forward(self, tile_planes, meta):
        batch_size = tile_planes.shape[0]
        x = F.pad(tile_planes, (0, GRID_CELLS - TILE_COUNT))
        x = x.view(batch_size, PLANE_COUNT, GRID_ROWS, GRID_COLS)
        x = F.relu(self.stem(x))
        for block in self.blocks:
            x = block(x)
        x = x.reshape(batch_size, -1)

        meta_features = F.relu(self.meta_proj(meta))
        hidden = F.relu(self.trunk(torch.cat([x, meta_features], dim=-1)))

        family_count = len(FAMILY_LABELS)
        embeddings = self.family_embedding.weight.unsqueeze(0).expand(batch_size, -1, -1)
        conditioned = torch.cat(
            [hidden.unsqueeze(1).expand(-1, family_count, -1), embeddings], dim=-1
        )

        value_hidden = F.relu(self.value_hidden(hidden))
        return {
            'hidden': hidden,
            'family_logits': self.family_head(hidden),
            'arg1_all': self.arg1_head(conditioned),
            'arg2_all': self.arg2_head(conditioned),
            'value': self.value_head(value_hidden),
            'win_logit': self.win_head(value_hidden),
        }


class CnnPolicyValueModel:
    def __init__(
        self,
        channels=32,
        blocks=3,
        hidden_size=256,
        learning_rate=0.001,
        seed=7,
        metadata=None,
        device='cpu',
    ):
        self.channels = int(channels)
        self.blocks = int(blocks)
        self.hidden_size = int(hidden_size)
        self.learning_rate = float(learning_rate)
        self.seed = int(seed)
        self.metadata = dict(metadata or {})
        self.metadata.setdefault('feature_schema_version', FEATURE_SCHEMA_VERSION)

        resolved_device, requested_name, resolved_name = resolve_device(device)
        self.device = resolved_device
        self.requested_device = requested_name
        self.resolved_device = resolved_name
        self.amp_enabled = bool(self.device.type == 'cuda')
        if self.amp_enabled:
            torch.set_float32_matmul_precision('high')

        torch.manual_seed(self.seed)
        self._sample_generator = torch.Generator(device='cpu')
        self._sample_generator.manual_seed(self.seed)

        self.model = CnnCore(
            channels=self.channels,
            blocks=self.blocks,
            hidden_size=self.hidden_size,
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def _autocast(self):
        if self.amp_enabled:
            return torch.autocast(device_type='cuda', dtype=torch.bfloat16)
        return torch.autocast(device_type='cpu', enabled=False)

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def encode_features_batch(self, features_list):
        planes = np.empty((len(features_list), PLANE_COUNT, TILE_COUNT), dtype=np.float32)
        meta = np.empty((len(features_list), META_COUNT), dtype=np.float32)
        for row, features in enumerate(features_list):
            self._validate_features(features)
            planes[row] = np.asarray(features['tile_planes'], dtype=np.float32)
            meta[row] = np.asarray(features['meta'], dtype=np.float32)
        return planes, meta

    @staticmethod
    def _validate_features(features):
        if not isinstance(features, dict):
            raise ValueError('Features must be a dict, got %r' % type(features))
        version = features.get('schema_version')
        if version != FEATURE_SCHEMA_VERSION:
            raise ValueError(
                'Feature schema mismatch: model expects v%d, got %r'
                % (FEATURE_SCHEMA_VERSION, version)
            )

    @staticmethod
    def encode_legal_actions(legal_actions_list, width=None):
        """Pad legal-action lists into (family, arg1, arg2, mask) index arrays."""
        if width is None:
            width = max(1, max(len(actions) for actions in legal_actions_list))
        batch = len(legal_actions_list)
        family = np.zeros((batch, width), dtype=np.int64)
        arg1 = np.full((batch, width), _NONE_INDEX, dtype=np.int64)
        arg2 = np.full((batch, width), _NONE_INDEX, dtype=np.int64)
        mask = np.zeros((batch, width), dtype=bool)
        for row, actions in enumerate(legal_actions_list):
            if len(actions) > width:
                raise ValueError('Legal action list longer than width %d' % width)
            for col, action in enumerate(actions):
                fam_idx, a1_idx, a2_idx = encode_action(action)
                family[row, col] = fam_idx
                arg1[row, col] = a1_idx
                arg2[row, col] = a2_idx
                mask[row, col] = True
        return family, arg1, arg2, mask

    def _to_device(self, array, dtype=None):
        tensor = torch.from_numpy(np.ascontiguousarray(array))
        if dtype is not None:
            tensor = tensor.to(dtype)
        return tensor.to(self.device)

    def _score_actions(self, outputs, family, arg1, arg2, mask):
        """Gather per-action scores (B, L) with -inf on padded slots."""
        arg_size = len(ARG_VOCAB)
        family_scores = outputs['family_logits'].gather(1, family)
        arg1_rows = outputs['arg1_all'].gather(
            1, family.unsqueeze(-1).expand(-1, -1, arg_size)
        )
        arg2_rows = outputs['arg2_all'].gather(
            1, family.unsqueeze(-1).expand(-1, -1, arg_size)
        )
        arg1_scores = arg1_rows.gather(2, arg1.unsqueeze(-1)).squeeze(-1)
        arg2_scores = arg2_rows.gather(2, arg2.unsqueeze(-1)).squeeze(-1)
        scores = family_scores + arg1_scores + arg2_scores
        return scores.masked_fill(~mask, float('-inf'))

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def policy_info_from_features(self, features, legal_actions, temperature=1.0):
        if not legal_actions:
            raise ValueError('policy_info_from_features requires at least one legal action')

        planes, meta = self.encode_features_batch([features])
        family, arg1, arg2, mask = self.encode_legal_actions([list(legal_actions)])

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(self._to_device(planes), self._to_device(meta))
            scores = self._score_actions(
                outputs,
                self._to_device(family),
                self._to_device(arg1),
                self._to_device(arg2),
                self._to_device(mask),
            )
            scores = scores / max(1e-3, float(temperature))
            log_probs = torch.log_softmax(scores, dim=1)[0]
            probs = torch.softmax(scores, dim=1)[0]
            finite = torch.isfinite(log_probs)
            entropy = float(-(probs[finite] * log_probs[finite]).sum().item())
            value = float(outputs['value'][0, 0].item())

        return {
            'actions': list(legal_actions),
            'probabilities': [float(p.item()) for p in probs],
            'log_probabilities': [float(lp.item()) for lp in log_probs],
            'entropy': entropy,
            'value': value,
        }

    def choose_action_from_features(self, features, legal_actions):
        """Deterministic greedy choice: highest probability, ties by name."""
        info = self.policy_info_from_features(features, legal_actions)
        ranked = sorted(
            zip(info['actions'], info['probabilities']),
            key=lambda item: (-item[1], item[0]),
        )
        return ranked[0][0]

    def sample_action_from_features(self, features, legal_actions, temperature=1.0, greedy=False):
        info = self.policy_info_from_features(features, legal_actions, temperature=temperature)
        actions = info['actions']
        if greedy or float(temperature) <= 0.0:
            index = max(
                range(len(actions)),
                key=lambda idx: (info['probabilities'][idx], actions[idx]),
            )
        else:
            probs = torch.tensor(info['probabilities'], dtype=torch.float32)
            index = int(torch.multinomial(probs, 1, generator=self._sample_generator).item())
        info = dict(info)
        info['selected_action'] = actions[index]
        info['selected_index'] = index
        info['selected_log_prob'] = float(info['log_probabilities'][index])
        return actions[index], info

    def estimate_value_from_features(self, features):
        planes, meta = self.encode_features_batch([features])
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(self._to_device(planes), self._to_device(meta))
            return float(outputs['value'][0, 0].item())

    # ------------------------------------------------------------------
    # Supervised training on a pre-encoded dataset
    # ------------------------------------------------------------------

    def _preencoded_batch(self, preencoded, indices):
        idx = np.asarray(indices, dtype=np.int64)
        # Planes are stored quantized (value * 4 as uint8, lossless for the
        # 0/0.25/0.5/0.75/1.0 grid) to keep caches 4x smaller than float32.
        planes_np = preencoded['tile_planes_q'][idx].astype(np.float32) * 0.25
        if planes_np.shape[1] < PLANE_COUNT:
            # Caches omit trailing oracle planes (always zero in supervised data).
            pad = np.zeros(
                (planes_np.shape[0], PLANE_COUNT - planes_np.shape[1], planes_np.shape[2]),
                dtype=np.float32,
            )
            planes_np = np.concatenate([planes_np, pad], axis=1)
        planes = self._to_device(planes_np)
        meta = self._to_device(preencoded['meta'][idx].astype(np.float32, copy=False))
        family = self._to_device(preencoded['legal_family'][idx].astype(np.int64, copy=False))
        arg1 = self._to_device(preencoded['legal_arg1'][idx].astype(np.int64, copy=False))
        arg2 = self._to_device(preencoded['legal_arg2'][idx].astype(np.int64, copy=False))
        lengths = preencoded['legal_len'][idx].astype(np.int64, copy=False)
        width = family.shape[1]
        mask = self._to_device(np.arange(width)[None, :] < lengths[:, None])
        target = self._to_device(preencoded['target_index'][idx].astype(np.int64, copy=False))

        return_key = 'return_target' if 'return_target' in preencoded else 'reward'
        returns = self._to_device(
            preencoded[return_key][idx].astype(np.float32, copy=False)
        ).unsqueeze(1)

        if 'win_flag' in preencoded:
            win_flags = preencoded['win_flag'][idx].astype(np.float32, copy=False)
        else:
            win_flags = (preencoded['reward'][idx] > 0.0).astype(np.float32)
        win = self._to_device(win_flags).unsqueeze(1)

        if 'sample_weight' in preencoded:
            weights = self._to_device(
                preencoded['sample_weight'][idx].astype(np.float32, copy=False)
            )
        else:
            weights = torch.ones(len(idx), dtype=torch.float32, device=self.device)

        return planes, meta, family, arg1, arg2, mask, target, returns, win, weights, lengths

    def train_preencoded_batch(
        self, preencoded, indices, policy_weight=1.0, value_weight=0.5, win_weight=0.2
    ):
        planes, meta, family, arg1, arg2, mask, target, returns, win, weights, lengths = (
            self._preencoded_batch(preencoded, indices)
        )
        self.model.train()
        with self._autocast():
            outputs = self.model(planes, meta)
            scores = self._score_actions(outputs, family, arg1, arg2, mask)
            per_sample = F.cross_entropy(scores, target, reduction='none')
            policy_loss = (per_sample * weights).sum() / weights.sum().clamp(min=1e-8)
            value_loss = F.mse_loss(outputs['value'], returns)
            win_loss = F.binary_cross_entropy_with_logits(outputs['win_logit'], win)
            total_loss = (
                float(policy_weight) * policy_loss
                + float(value_weight) * value_loss
                + float(win_weight) * win_loss
            )

        self.optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        self.optimizer.step()

        with torch.no_grad():
            predictions = torch.argmax(scores, dim=1)
            hits = (predictions == target).to(torch.int64)
            decision_mask = self._to_device(lengths > 1)
            decision_hits = int((hits * decision_mask.to(torch.int64)).sum().item())

        count = int(len(indices))
        return {
            'samples': count,
            'policy_loss': float(policy_loss.item()) * count,
            'value_loss': float(value_loss.item()) * count,
            'win_loss': float(win_loss.item()) * count,
            'action_hits': int(hits.sum().item()),
            'decision_samples': int(decision_mask.sum().item()),
            'decision_hits': decision_hits,
        }

    def fit_preencoded(
        self,
        preencoded,
        epochs=1,
        batch_size=512,
        shuffle=True,
        sample_indices=None,
        policy_weight=1.0,
        value_weight=0.5,
        win_weight=0.2,
        show_progress=False,
    ):
        total_all = int(len(preencoded['target_index']))
        base_indices = (
            np.arange(total_all, dtype=np.int64)
            if sample_indices is None
            else np.asarray(sample_indices, dtype=np.int64)
        )
        total = int(len(base_indices))
        stats = {
            'samples': 0,
            'policy_loss': 0.0,
            'value_loss': 0.0,
            'win_loss': 0.0,
            'action_hits': 0,
            'decision_samples': 0,
            'decision_hits': 0,
        }
        if total <= 0:
            return stats

        batch_size = max(1, int(batch_size))
        for epoch_idx in range(max(1, int(epochs))):
            indices = np.array(base_indices, copy=True)
            if shuffle:
                np.random.default_rng(self.seed + epoch_idx).shuffle(indices)
            label = 'epoch %d/%d' % (epoch_idx + 1, max(1, int(epochs)))
            seen = 0
            for start in range(0, total, batch_size):
                batch = indices[start: start + batch_size]
                result = self.train_preencoded_batch(
                    preencoded,
                    batch,
                    policy_weight=policy_weight,
                    value_weight=value_weight,
                    win_weight=win_weight,
                )
                for key in stats:
                    stats[key] += result[key]
                seen += len(batch)
                if show_progress:
                    _print_progress_bar(label, seen, total)
        return stats

    def evaluate_preencoded(self, preencoded, sample_indices=None, batch_size=2048, top_ks=(1, 3), ece_bins=10):
        total_all = int(len(preencoded['target_index']))
        indices = (
            np.arange(total_all, dtype=np.int64)
            if sample_indices is None
            else np.asarray(sample_indices, dtype=np.int64)
        )
        top_ks = sorted(set(int(k) for k in top_ks if int(k) > 0))
        metrics = {
            'evaluated': 0,
            'decision_evaluated': 0,
            'forced_evaluated': 0,
            'masked_cross_entropy': 0.0,
            'value_mse': 0.0,
            'win_accuracy': 0.0,
            'ece': 0.0,
            'topk_accuracy': {str(k): 0.0 for k in top_ks},
        }
        if len(indices) <= 0:
            return metrics

        nll_sum = 0.0
        mse_sum = 0.0
        win_hits = 0
        decision_count = 0
        hit_counts = {k: 0 for k in top_ks}
        bin_confidence = np.zeros(int(ece_bins), dtype=np.float64)
        bin_accuracy = np.zeros(int(ece_bins), dtype=np.float64)
        bin_total = np.zeros(int(ece_bins), dtype=np.int64)

        self.model.eval()
        with torch.no_grad():
            for start in range(0, len(indices), max(1, int(batch_size))):
                batch = indices[start: start + max(1, int(batch_size))]
                planes, meta, family, arg1, arg2, mask, target, returns, win, _weights, lengths = (
                    self._preencoded_batch(preencoded, batch)
                )
                outputs = self.model(planes, meta)
                scores = self._score_actions(outputs, family, arg1, arg2, mask)
                log_probs = torch.log_softmax(scores, dim=1)
                probs = torch.softmax(scores, dim=1)
                target_log_probs = log_probs.gather(1, target.unsqueeze(1)).squeeze(1)

                decision_rows = self._to_device(lengths > 1)
                nll_sum += float((-target_log_probs * decision_rows).sum().item())
                decision_count += int(decision_rows.sum().item())
                mse_sum += float(F.mse_loss(outputs['value'], returns, reduction='sum').item())
                win_pred = (outputs['win_logit'] > 0.0).to(torch.float32)
                win_hits += int((win_pred == win).sum().item())

                top1_prob, top1_index = probs.max(dim=1)
                top1_correct = (top1_index == target).to(torch.float32)
                ranked = torch.argsort(scores, dim=1, descending=True)
                for k in top_ks:
                    in_top = (ranked[:, :k] == target.unsqueeze(1)).any(dim=1)
                    hit_counts[k] += int((in_top & decision_rows).sum().item())

                # Calibration buckets over decision states only.
                decision_np = decision_rows.cpu().numpy()
                confidence_np = top1_prob.cpu().numpy()[decision_np]
                correct_np = top1_correct.cpu().numpy()[decision_np]
                bucket = np.minimum(
                    (confidence_np * int(ece_bins)).astype(np.int64), int(ece_bins) - 1
                )
                np.add.at(bin_confidence, bucket, confidence_np)
                np.add.at(bin_accuracy, bucket, correct_np)
                np.add.at(bin_total, bucket, 1)

                metrics['evaluated'] += int(len(batch))

        metrics['decision_evaluated'] = decision_count
        metrics['forced_evaluated'] = metrics['evaluated'] - decision_count
        if decision_count > 0:
            metrics['masked_cross_entropy'] = nll_sum / float(decision_count)
            for k in top_ks:
                metrics['topk_accuracy'][str(k)] = hit_counts[k] / float(decision_count)
            occupied = bin_total > 0
            gaps = np.abs(
                bin_confidence[occupied] / bin_total[occupied]
                - bin_accuracy[occupied] / bin_total[occupied]
            )
            metrics['ece'] = float((gaps * bin_total[occupied]).sum() / bin_total[occupied].sum())
        if metrics['evaluated'] > 0:
            metrics['value_mse'] = mse_sum / float(metrics['evaluated'])
            metrics['win_accuracy'] = win_hits / float(metrics['evaluated'])
        return metrics

    # ------------------------------------------------------------------
    # PPO fine-tuning
    # ------------------------------------------------------------------

    def ppo_update(
        self,
        transitions,
        clip_range=0.2,
        entropy_coef=0.01,
        value_coef=0.5,
        win_coef=0.1,
        epochs=4,
        minibatch_size=256,
    ):
        """Batched PPO update over a list of transition dicts.

        Each transition needs: features, legal_actions, action,
        old_log_prob, advantage, return_target; optional win_target.
        """
        if not transitions:
            return {'updates': 0, 'policy_loss': 0.0, 'value_loss': 0.0, 'entropy': 0.0}

        planes_np, meta_np = self.encode_features_batch(
            [t['features'] for t in transitions]
        )
        family_np, arg1_np, arg2_np, mask_np = self.encode_legal_actions(
            [list(t['legal_actions']) for t in transitions]
        )
        action_index_np = np.array(
            [list(t['legal_actions']).index(t['action']) for t in transitions],
            dtype=np.int64,
        )
        old_log_prob_np = np.array(
            [float(t['old_log_prob']) for t in transitions], dtype=np.float32
        )
        advantage_np = np.array(
            [float(t['advantage']) for t in transitions], dtype=np.float32
        )
        if len(advantage_np) > 1 and float(advantage_np.std()) > 1e-8:
            advantage_np = (advantage_np - advantage_np.mean()) / (advantage_np.std() + 1e-8)
        return_np = np.array(
            [float(t['return_target']) for t in transitions], dtype=np.float32
        )
        win_np = np.array(
            [
                float(t.get('win_target', 1.0 if float(t['return_target']) > 0.0 else 0.0))
                for t in transitions
            ],
            dtype=np.float32,
        )

        planes = self._to_device(planes_np)
        meta = self._to_device(meta_np)
        family = self._to_device(family_np)
        arg1 = self._to_device(arg1_np)
        arg2 = self._to_device(arg2_np)
        mask = self._to_device(mask_np)
        action_index = self._to_device(action_index_np)
        old_log_prob = self._to_device(old_log_prob_np)
        advantage = self._to_device(advantage_np)
        return_target = self._to_device(return_np).unsqueeze(1)
        win_target = self._to_device(win_np).unsqueeze(1)

        total = len(transitions)
        stats = {'updates': 0, 'policy_loss': 0.0, 'value_loss': 0.0, 'entropy': 0.0}
        rng = np.random.default_rng(self.seed)

        self.model.train()
        for _epoch in range(max(1, int(epochs))):
            order = np.arange(total)
            rng.shuffle(order)
            for start in range(0, total, max(1, int(minibatch_size))):
                rows = self._to_device(order[start: start + max(1, int(minibatch_size))])
                with self._autocast():
                    outputs = self.model(planes[rows], meta[rows])
                    scores = self._score_actions(
                        outputs, family[rows], arg1[rows], arg2[rows], mask[rows]
                    )
                    log_probs = torch.log_softmax(scores, dim=1)
                    probs = torch.softmax(scores, dim=1)
                    current_log_prob = log_probs.gather(
                        1, action_index[rows].unsqueeze(1)
                    ).squeeze(1)

                    ratio = torch.exp(current_log_prob - old_log_prob[rows])
                    clipped = torch.clamp(ratio, 1.0 - float(clip_range), 1.0 + float(clip_range))
                    adv = advantage[rows]
                    policy_loss = -torch.minimum(ratio * adv, clipped * adv).mean()

                    # Clamp -inf padded slots: probs are 0 there, and clamping
                    # (rather than torch.where) keeps their gradients NaN-free.
                    entropy = -(probs * log_probs.clamp(min=-30.0)).sum(dim=1).mean()
                    value_loss = F.mse_loss(outputs['value'], return_target[rows])
                    win_loss = F.binary_cross_entropy_with_logits(
                        outputs['win_logit'], win_target[rows]
                    )
                    loss = (
                        policy_loss
                        + float(value_coef) * value_loss
                        + float(win_coef) * win_loss
                        - float(entropy_coef) * entropy
                    )

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.optimizer.step()

                stats['updates'] += 1
                stats['policy_loss'] += float(policy_loss.item())
                stats['value_loss'] += float(value_loss.item())
                stats['entropy'] += float(entropy.item())

        return stats

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def state_dict(self):
        return {key: tensor.detach().cpu() for key, tensor in self.model.state_dict().items()}

    def load_state_dict(self, state_dict):
        current = self.model.state_dict()
        if set(state_dict.keys()) != set(current.keys()):
            missing = sorted(set(current) - set(state_dict))
            unexpected = sorted(set(state_dict) - set(current))
            raise ValueError(
                'Checkpoint keys mismatch. missing=%r unexpected=%r' % (missing, unexpected)
            )
        prepared = {}
        for key, value in state_dict.items():
            tensor = torch.as_tensor(np.asarray(value), dtype=torch.float32, device=self.device)
            if tuple(tensor.shape) != tuple(current[key].shape):
                raise ValueError(
                    'Checkpoint shape mismatch for %s: %r vs %r'
                    % (key, tuple(tensor.shape), tuple(current[key].shape))
                )
            prepared[key] = tensor
        self.model.load_state_dict(prepared, strict=True)

    def save(self, path):
        parent = os.path.dirname(path)
        if parent and not os.path.exists(parent):
            os.makedirs(parent)
        self.metadata['feature_schema_version'] = FEATURE_SCHEMA_VERSION
        with h5py.File(path, 'w') as handle:
            handle.attrs['model_type'] = MODEL_TYPE
            handle.attrs['channels'] = self.channels
            handle.attrs['blocks'] = self.blocks
            handle.attrs['hidden_size'] = self.hidden_size
            handle.attrs['learning_rate'] = self.learning_rate
            handle.attrs['seed'] = self.seed
            handle.attrs['plane_count'] = PLANE_COUNT
            handle.attrs['meta_count'] = META_COUNT
            handle.attrs['feature_schema_version'] = FEATURE_SCHEMA_VERSION
            handle.attrs['metadata'] = json.dumps(self.metadata, sort_keys=True)
            group = handle.create_group('state_dict')
            for key, tensor in self.state_dict().items():
                group.create_dataset(key, data=tensor.numpy())

    @classmethod
    def load(cls, path, device='cpu'):
        if not os.path.exists(path):
            raise FileNotFoundError('Model checkpoint not found: %s' % path)
        with h5py.File(path, 'r') as handle:
            model_type = handle.attrs.get('model_type')
            if model_type != MODEL_TYPE:
                raise ValueError(
                    'Unsupported checkpoint type %r (expected %r). '
                    'Retrain the model with the current pipeline.' % (model_type, MODEL_TYPE)
                )
            schema = int(handle.attrs.get('feature_schema_version', -1))
            if schema != FEATURE_SCHEMA_VERSION:
                raise ValueError(
                    'Checkpoint feature schema v%d does not match current v%d. '
                    'Retrain the model.' % (schema, FEATURE_SCHEMA_VERSION)
                )
            if int(handle.attrs.get('plane_count', -1)) != PLANE_COUNT or int(
                handle.attrs.get('meta_count', -1)
            ) != META_COUNT:
                raise ValueError('Checkpoint input contract does not match features module.')

            model = cls(
                channels=int(handle.attrs['channels']),
                blocks=int(handle.attrs['blocks']),
                hidden_size=int(handle.attrs['hidden_size']),
                learning_rate=float(handle.attrs.get('learning_rate', 0.001)),
                seed=int(handle.attrs.get('seed', 7)),
                metadata=json.loads(handle.attrs.get('metadata', '{}')),
                device=device,
            )
            state = {key: np.asarray(handle['state_dict'][key][()]) for key in handle['state_dict']}
        model.load_state_dict(state)
        return model
