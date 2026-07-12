import os
import importlib.util

try:
    import numpy as _numpy
except Exception:
    _numpy = None


try:
    import torch as _torch
except Exception:
    _torch = None


try:
    import scipy as _scipy
except Exception:
    _scipy = None


try:
    import h5py as _h5py
except Exception:
    _h5py = None


def has_numpy():
    return _numpy is not None


def has_torch():
    return _torch is not None


def has_scipy():
    return _scipy is not None


def has_h5py():
    return _h5py is not None


def has_tensorflow():
    return importlib.util.find_spec('tensorflow') is not None


def preferred_numeric_backend():
    if has_torch():
        return 'torch'
    if has_numpy():
        return 'numpy'
    return 'python'


def package_profile():
    return {
        'numpy': has_numpy(),
        'torch': has_torch(),
        'scipy': has_scipy(),
        'h5py': has_h5py(),
        'tensorflow': has_tensorflow(),
        'preferred_numeric_backend': preferred_numeric_backend(),
        'recommended_stack': [
            'torch',
            'numpy',
            'scipy',
            'h5py',
        ],
    }