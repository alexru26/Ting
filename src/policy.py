"""Runtime policy: the neural model is the only decision source.

There is no rule-based fallback and no silent recovery: a missing or
incompatible checkpoint raises immediately, and any inference failure
propagates to the caller. Forced turns (exactly one legal action) are
answered without touching the model, so every model call is a real
decision.

Environment overrides:
- TING_POLICY_MODEL_PATH: checkpoint path (default: src/model.h5).
- TING_LOG_DECISIONS=1: log each decision and its source to stderr.
"""

import logging
import os

from features import FeatureExtractor
from model import CnnPolicyValueModel

_LOGGER = logging.getLogger(__name__)

_MODEL_CACHE = {}


def default_model_path():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model.h5')


def load_model(model_path=None, device='cpu'):
    """Load (and cache) the policy checkpoint. Raises if unavailable."""
    path = model_path or os.getenv('TING_POLICY_MODEL_PATH') or default_model_path()
    cache_key = (os.path.abspath(path), str(device))
    if cache_key not in _MODEL_CACHE:
        _MODEL_CACHE[cache_key] = CnnPolicyValueModel.load(path, device=device)
    return _MODEL_CACHE[cache_key]


class NeuralPolicy:
    def __init__(self, state, model_path=None, model=None):
        self.state = state
        self.model = model if model is not None else load_model(model_path)
        self._log_decisions = os.getenv('TING_LOG_DECISIONS', '').strip() in ('1', 'true', 'True')

    def choose_action(self):
        legal_actions = self.state.enumerate_legal_actions()
        if not legal_actions:
            raise ValueError(
                'No legal actions for request type %r action %r'
                % (self.state.last_request_type, self.state.last_request_action)
            )

        if len(legal_actions) == 1:
            self._log('forced', legal_actions[0], legal_actions)
            return legal_actions[0]

        features = FeatureExtractor().extract(self.state)
        action = self.model.choose_action_from_features(features, legal_actions)
        if action not in legal_actions:
            raise ValueError('Model produced illegal action %r' % (action,))
        self._log('model', action, legal_actions)
        return action

    def _log(self, source, action, legal_actions):
        if self._log_decisions:
            _LOGGER.warning(
                'decision source=%s action=%s legal_count=%d', source, action, len(legal_actions)
            )


def create_policy(state, model_path=None, model=None):
    return NeuralPolicy(state, model_path=model_path, model=model)
