"""Tests for ``disable_bn_running_stats``.

The helper removes BatchNorm's reliance on running statistics by setting
``track_running_stats=False`` and nulling the per-layer buffers. With no
running stats, ``BatchNorm.forward`` falls through to per-batch mean/var
in both ``.train()`` and ``.eval()`` mode — the same invariant the
cascade needs for a frozen Stage-1 pre-filter whose running stats are
stale.
"""
import torch
import torch.nn as nn

from weaver.nn.model.stateless_bn import disable_bn_running_stats


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
        if isinstance(m, nn.modules.batchnorm._BatchNorm)
    ]


class TestDisableBNRunningStats:
    def test_sets_track_running_stats_false(self):
        model = _make_model_with_bn_and_dropout()
        disable_bn_running_stats(model)
        for batch_norm in _collect_bn(model):
            assert batch_norm.track_running_stats is False

    def test_nulls_running_buffers(self):
        model = _make_model_with_bn_and_dropout()
        disable_bn_running_stats(model)
        for batch_norm in _collect_bn(model):
            assert batch_norm.running_mean is None
            assert batch_norm.running_var is None
            assert batch_norm.num_batches_tracked is None

    def test_returns_same_module(self):
        model = _make_model_with_bn_and_dropout()
        assert disable_bn_running_stats(model) is model

    def test_eval_mode_uses_batch_statistics(self):
        # Input chosen so running-stat behaviour (mean 0, var 1) and
        # batch-stat behaviour diverge visibly.
        inputs = torch.randn(4, 3, 16) * 5.0 + 10.0

        # Bare BN in eval: uses running_mean=0 / running_var=1, so output
        # ≈ input (no normalization effect).
        bare_bn = nn.BatchNorm1d(3)
        bare_bn.eval()
        bare_output = bare_bn(inputs)

        # BN with running stats disabled, in eval: falls through to batch
        # stats, output is mean-zero / unit-variance per-channel.
        stateless_bn = nn.BatchNorm1d(3)
        disable_bn_running_stats(stateless_bn)
        stateless_bn.eval()
        stateless_output = stateless_bn(inputs)

        assert not torch.allclose(bare_output, stateless_output, atol=1e-4)

        # Output should match a BN explicitly forward'd in train mode.
        train_bn = nn.BatchNorm1d(3)
        train_bn.train()
        with torch.no_grad():
            train_output = train_bn(inputs)
        assert torch.allclose(train_output, stateless_output, atol=1e-5)

    def test_train_mode_still_uses_batch_statistics(self):
        # In train mode, BN always uses batch stats, regardless of the
        # tracking flag. Confirm the helper doesn't break that path.
        torch.manual_seed(0)
        inputs = torch.randn(4, 3, 16) * 3.0 + 2.0

        model_a = nn.BatchNorm1d(3)
        model_a.train()
        output_a = model_a(inputs)

        model_b = nn.BatchNorm1d(3)
        disable_bn_running_stats(model_b)
        model_b.train()
        output_b = model_b(inputs)

        assert torch.allclose(output_a, output_b, atol=1e-5)

    def test_dropout_and_other_layers_untouched(self):
        model = _make_model_with_bn_and_dropout()
        disable_bn_running_stats(model)
        # Dropout still has its p; Conv1d still has its weight.
        dropouts = [m for m in model.modules() if isinstance(m, nn.Dropout)]
        assert len(dropouts) == 1
        assert dropouts[0].p == 0.5
        convs = [m for m in model.modules() if isinstance(m, nn.Conv1d)]
        assert len(convs) == 2

    def test_forward_does_not_update_anything(self):
        # With track_running_stats=False, forward should not mutate any
        # internal state (no running_mean / num_batches_tracked writes).
        model = nn.BatchNorm1d(4)
        disable_bn_running_stats(model)
        model.train()
        _ = model(torch.randn(2, 4, 8))
        assert model.running_mean is None
        assert model.running_var is None
        assert model.num_batches_tracked is None

    def test_idempotent(self):
        model = _make_model_with_bn_and_dropout()
        disable_bn_running_stats(model)
        disable_bn_running_stats(model)
        for batch_norm in _collect_bn(model):
            assert batch_norm.track_running_stats is False
            assert batch_norm.running_mean is None

    def test_nested_module(self):
        child = _make_model_with_bn_and_dropout()
        parent = nn.ModuleDict({'child': child})
        disable_bn_running_stats(parent)
        for batch_norm in _collect_bn(parent):
            assert batch_norm.track_running_stats is False
            assert batch_norm.running_mean is None

    def test_load_state_dict_after_transform(self):
        # Stage-1 checkpoint has running_mean/var populated. Loading it
        # into a module whose BN has been stateless-ified must either
        # succeed (via strict=False) or the caller must load first then
        # disable. Verify the "load first, then disable" ordering works
        # and preserves the loaded weights / biases.
        source = nn.Sequential(nn.Conv1d(4, 4, 1), nn.BatchNorm1d(4))
        source[1].weight.data.fill_(2.5)
        source[1].bias.data.fill_(-1.7)

        target = nn.Sequential(nn.Conv1d(4, 4, 1), nn.BatchNorm1d(4))
        target.load_state_dict(source.state_dict())
        disable_bn_running_stats(target)

        assert torch.allclose(target[1].weight, source[1].weight)
        assert torch.allclose(target[1].bias, source[1].bias)
        assert target[1].track_running_stats is False
        assert target[1].running_mean is None

    def test_state_dict_omits_running_stats_after_transform(self):
        # Once running stats are disabled, saving + loading the module
        # should round-trip cleanly without stale running_mean entries
        # drifting through the state_dict.
        model = nn.Sequential(nn.BatchNorm1d(4))
        disable_bn_running_stats(model)

        state = model.state_dict()
        # running_* keys are either absent or None; num_batches_tracked
        # likewise. weight / bias remain.
        for key, value in state.items():
            if key.endswith('running_mean') or key.endswith('running_var') or key.endswith('num_batches_tracked'):
                assert value is None

        fresh = nn.Sequential(nn.BatchNorm1d(4))
        disable_bn_running_stats(fresh)
        fresh.load_state_dict(state)
        assert fresh[0].running_mean is None
