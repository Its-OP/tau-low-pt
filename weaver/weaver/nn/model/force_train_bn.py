"""In-place wrapper that pins BatchNorm submodules to training mode.

Some setups must run BatchNorm with batch statistics even at inference.
The canonical example in this repo is a frozen Stage-1 pre-filter inside
a cascade whose BatchNorm running statistics are stale (the pre-filter
training loop ran validation in ``.train()`` mode, corrupting
``running_mean`` / ``running_var``); using them at eval time drops R@600
from 0.90 to 0.70 and caps downstream cascade performance.

Without this helper, callers either (a) manually iterate all submodules
on every forward to toggle BN ``.training`` flags, or (b) call
``model.train()`` which puts *everything* — including Dropout — into
training mode as an unwanted side effect.

``force_train_bn(module)`` patches ``module.train`` in place so that any
BatchNorm submodule inside it always ends up with ``training=True``,
while the rest of the module (Dropout, LayerNorm, etc.) still obeys the
requested mode. Returns ``module`` for fluent chaining.
"""
from __future__ import annotations

from typing import TypeVar

import torch.nn as nn

ModuleT = TypeVar('ModuleT', bound=nn.Module)

_BATCH_NORM_TYPES = (
    nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm,
)
_FORCE_TRAIN_BN_SENTINEL = '_force_train_bn_patched'


def force_train_bn(module: ModuleT) -> ModuleT:
    """Patch ``module.train`` so BN submodules stay in training mode.

    Idempotent: applying the helper twice to the same module is a no-op.
    """
    if getattr(module, _FORCE_TRAIN_BN_SENTINEL, False):
        return module

    original_train = module.train

    def patched_train(mode: bool = True):
        original_train(mode)
        for submodule in module.modules():
            if isinstance(submodule, _BATCH_NORM_TYPES):
                submodule.training = True
        return module

    module.train = patched_train
    setattr(module, _FORCE_TRAIN_BN_SENTINEL, True)
    # Re-run the patched train() so BN training flags reflect the
    # requirement immediately, regardless of the mode the module was
    # in when this helper was called.
    module.train(module.training)
    return module
