"""Disable BatchNorm running statistics in-place.

Some cascaded setups must run BatchNorm with batch statistics even at
inference. The canonical case is a frozen Stage-1 pre-filter inside a
cascade: its ``running_mean`` / ``running_var`` were corrupted because
the pre-filter training loop ran validation in ``.train()`` mode, and
using the stored running statistics at cascade inference drops R@600
from 0.90 to 0.70.

``disable_bn_running_stats(module)`` sweeps every BatchNorm submodule
and flips it to stateless mode::

    batch_norm.track_running_stats = False
    batch_norm.running_mean = None
    batch_norm.running_var = None
    batch_norm.num_batches_tracked = None

With no running buffers, ``BatchNorm.forward`` takes the same code path
in both ``.train()`` and ``.eval()`` — it computes per-batch mean /
variance from the input. The cascade's outer ``.train()`` / ``.eval()``
state no longer affects normalization, so Dropout can still follow the
outer mode for proper validation behaviour.

Compared to the previous ``force_train_bn`` monkey-patch:

* no override of ``module.train``,
* no hidden sentinel attribute,
* no running-stat drift on every forward (buffers do not exist),
* a standard PyTorch mechanism (``track_running_stats=False``).

Call this AFTER loading any pre-trained weights. Loading a checkpoint
whose state dict still has ``running_mean`` / ``running_var`` tensors
must happen first, because after the transform those buffers are
``None`` and ``load_state_dict`` with ``strict=True`` would fail.
"""
from __future__ import annotations

from typing import TypeVar

import torch.nn as nn

ModuleT = TypeVar('ModuleT', bound=nn.Module)

_BATCH_NORM_TYPES = (
    nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm,
)


def disable_bn_running_stats(module: ModuleT) -> ModuleT:
    """Set ``track_running_stats=False`` and null buffers on every BN.

    Idempotent: BN submodules already in stateless mode are left alone.
    Returns the same module for fluent chaining.
    """
    for submodule in module.modules():
        if not isinstance(submodule, _BATCH_NORM_TYPES):
            continue
        submodule.track_running_stats = False
        # Nulling these buffers routes ``BatchNorm.forward`` through the
        # batch-statistics branch in eval mode (see the
        # ``bn_training = (running_mean is None) and (running_var is None)``
        # check in ``nn.modules.batchnorm._BatchNorm.forward``).
        submodule.running_mean = None
        submodule.running_var = None
        submodule.num_batches_tracked = None
    return module
