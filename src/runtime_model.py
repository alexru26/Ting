import json

from model import CnnPolicyValueModel


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
        except Exception:
            try:
                return model.choose_action_from_features(features, legal_actions)
            except Exception:
                return None

    return None
