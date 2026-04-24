"""Tests for ``force_train_bn`` — the in-place BN-mode pin used by the
cascade to keep Stage-1 BatchNorm on batch statistics while letting
Dropout obey the surrounding train/eval regime.
"""
import torch
import torch.nn as nn

from weaver.nn.model.force_train_bn import force_train_bn


def _make_model_with_bn_and_dropout():
    return nn.Sequential(
        nn.Conv1d(8, 16, kernel_size=1),
        nn.BatchNorm1d(16),
        nn.Dropout(p=0.5),
        nn.Conv1d(16, 16, kernel_size=1),
        nn.BatchNorm1d(16),
    )


def _collect_bn(module):
    return [
        m for m in module.modules()
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d))
    ]


def _collect_dropout(module):
    return [m for m in module.modules() if isinstance(m, nn.Dropout)]


class TestForceTrainBN:
    def test_returns_same_module(self):
        model = _make_model_with_bn_and_dropout()
        assert force_train_bn(model) is model

    def test_eval_keeps_bn_in_train_mode(self):
        model = _make_model_with_bn_and_dropout()
        force_train_bn(model)
        model.eval()
        for batch_norm in _collect_bn(model):
            assert batch_norm.training is True

    def test_eval_puts_dropout_in_eval_mode(self):
        model = _make_model_with_bn_and_dropout()
        force_train_bn(model)
        model.eval()
        for dropout in _collect_dropout(model):
            assert dropout.training is False

    def test_train_still_trains_dropout(self):
        model = _make_model_with_bn_and_dropout()
        force_train_bn(model)
        model.train()
        for dropout in _collect_dropout(model):
            assert dropout.training is True
        for batch_norm in _collect_bn(model):
            assert batch_norm.training is True

    def test_is_idempotent(self):
        model = _make_model_with_bn_and_dropout()
        force_train_bn(model)
        patched = model.train
        force_train_bn(model)
        assert model.train is patched

    def test_works_through_parent_recursion(self):
        # When the wrapped module is a child of a larger container,
        # parent.train()/parent.eval() propagates to children via
        # nn.Module's own recursion — the patch must intercept that.
        child = _make_model_with_bn_and_dropout()
        force_train_bn(child)
        parent = nn.Module()
        parent.child = child

        parent.eval()
        for batch_norm in _collect_bn(child):
            assert batch_norm.training is True
        for dropout in _collect_dropout(child):
            assert dropout.training is False

        parent.train()
        for batch_norm in _collect_bn(child):
            assert batch_norm.training is True

    def test_batch_norm_still_updates_running_stats_in_eval(self):
        # With the patch, BN behaves as if training=True — running stats
        # update during forward. Confirms the patch takes effect end-to-end.
        model = _make_model_with_bn_and_dropout()
        force_train_bn(model)
        model.eval()

        batch_norm_layers = _collect_bn(model)
        running_mean_before = batch_norm_layers[0].running_mean.clone()

        inputs = torch.randn(4, 8, 32)
        model(inputs)

        running_mean_after = batch_norm_layers[0].running_mean
        assert not torch.equal(running_mean_before, running_mean_after)
