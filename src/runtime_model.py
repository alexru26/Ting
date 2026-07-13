import json
import logging

from model import CnnPolicyValueModel


_LOGGER = logging.getLogger(__name__)


def load_policy_model(path):
    with open(path, 'rb') as handle:
        header = handle.read(8)

    if header == b'\x89HDF\r\n\x1a\n':
        return CnnPolicyValueModel.load(path)

    with open(path, 'r', encoding='utf-8') as handle:
        payload = json.load(handle)

    if not isinstance(payload, dict):
        raise ValueError('Invalid checkpoint payload.')
    return CnnPolicyValueModel.from_dict(payload)


def choose_action_from_model(model, features, legal_actions, codec=None, belief_weight=0.0):
    if hasattr(model, 'choose_action_from_features'):
        try:
            return model.choose_action_from_features(features, legal_actions, belief_weight=belief_weight)
        except TypeError:
            try:
                return model.choose_action_from_features(features, legal_actions)
            except Exception as exc:
                _LOGGER.debug('Model action selection failed after fallback signature: %s', exc)
                return None
        except Exception as exc:
            _LOGGER.debug('Model action selection failed: %s', exc)
            return None

    return None
