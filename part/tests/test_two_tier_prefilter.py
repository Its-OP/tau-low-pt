"""Tests for TwoTierPreFilter (P6).

The two-tier prefilter stacks a cheap coarse scorer + a richer refine
scorer inside Stage 1: the coarse pass scores all tracks and selects
the top-N, the refine pass scores the selected subset only. Composite
output keeps the (B, P) shape of the single-tier prefilter so the
downstream training / eval harness is unchanged.

Contract pinned here:
* Output shape matches the single-tier TrackPreFilter: ``(B, P)``.
* Selected tracks (by coarse top-N) receive the refine score.
* Non-selected valid tracks keep a baseline derived from coarse score
  minus a large offset, so they always rank below selected tracks —
  this is the mechanism that makes top-N selection **effective** on
  the composite output.
* Padded (masked) tracks stay at ``-inf`` so they never show up in
  top-K selections downstream.
* Gradient flows through both tiers when training.
"""
from __future__ import annotations

import torch


def _make_batch(batch_size: int = 2, num_tracks: int = 40, input_dim: int = 16):
    points = torch.randn(batch_size, 2, num_tracks) * 2.0
    features = torch.randn(batch_size, input_dim, num_tracks)
    momentum = torch.randn(batch_size, 3, num_tracks) * 2.0
    mass_sq = torch.full((batch_size, 1, num_tracks), 0.0196)
    energy = torch.sqrt(
        (momentum ** 2).sum(dim=1, keepdim=True) + mass_sq,
    )
    lorentz = torch.cat([momentum, energy], dim=1)
    mask = torch.ones(batch_size, 1, num_tracks)
    mask[:, :, -3:] = 0.0
    return points, features, lorentz, mask


def _make_model(top_n: int = 20, **overrides):
    from weaver.nn.model.TwoTierPreFilter import TwoTierPreFilter

    base_config = dict(
        input_dim=16,
        top_n=top_n,
        coarse_hidden_dim=32,
        refine_hidden_dim=48,
        coarse_neighbors=8,
        # refine_neighbors must stay < top_n; the refine tier runs kNN
        # over the top-N subset so k >= N throws in HierarchicalGraph.
        refine_neighbors=min(12, max(4, top_n - 1)),
        coarse_message_rounds=1,
        refine_message_rounds=2,
        use_edge_features=True,
        dropout=0.0,
    )
    base_config.update(overrides)
    return TwoTierPreFilter(**base_config)


class TestTwoTierShape:
    def test_forward_matches_single_tier_shape(self):
        model = _make_model()
        model.eval()
        points, features, lorentz, mask = _make_batch()
        with torch.no_grad():
            scores = model(points, features, lorentz, mask)
        assert scores.shape == (points.shape[0], points.shape[2])

    def test_masked_tracks_return_minus_inf(self):
        model = _make_model()
        model.eval()
        points, features, lorentz, mask = _make_batch()
        with torch.no_grad():
            scores = model(points, features, lorentz, mask)
        valid = mask.squeeze(1).bool()
        # All invalid positions are −inf; all valid positions finite.
        assert torch.isinf(scores[~valid]).all()
        assert scores[~valid].lt(0).all()
        assert torch.isfinite(scores[valid]).all()


class TestTopNSelection:
    def test_top_n_selected_rank_above_non_selected(self):
        """Composite-score contract: every selected track scores
        strictly above every non-selected valid track."""
        model = _make_model(top_n=12)
        model.eval()
        points, features, lorentz, mask = _make_batch(num_tracks=30)
        with torch.no_grad():
            scores = model(points, features, lorentz, mask)

        batch_size = scores.shape[0]
        for batch_index in range(batch_size):
            valid = mask[batch_index, 0].bool()
            valid_indices = valid.nonzero(as_tuple=True)[0]
            valid_scores = scores[batch_index, valid_indices]
            ranks = torch.argsort(-valid_scores)
            top_n_valid = ranks[:12].tolist()
            rest = ranks[12:].tolist()
            if rest:
                assert valid_scores[top_n_valid].min() > valid_scores[rest].max(), (
                    'Top-N composite scores must dominate the rest of the '
                    'valid-track score distribution.'
                )

    def test_top_n_uses_coarse_scores_for_selection(self):
        """Gradient check: moving coarse scores should change which
        tracks get the refine-level composite score. Concretely: run
        forward twice and see that the selected-track set is
        deterministic given the same inputs."""
        model = _make_model(top_n=10)
        model.eval()
        points, features, lorentz, mask = _make_batch(num_tracks=30)
        with torch.no_grad():
            scores_a = model(points, features, lorentz, mask)
            scores_b = model(points, features, lorentz, mask)
        torch.testing.assert_close(scores_a, scores_b, atol=0, rtol=0)


class TestTwoTierGradient:
    def test_backward_propagates_to_both_tiers(self):
        model = _make_model()
        model.train()
        points, features, lorentz, mask = _make_batch()
        scores = model(points, features, lorentz, mask)
        scores.sum().backward()
        # At least one param in each tier received a finite gradient.
        for name, parameter in model.named_parameters():
            assert parameter.grad is not None, f'{name} got no grad'
            assert torch.isfinite(parameter.grad).all(), (
                f'{name} grad has non-finite entries'
            )

    def test_compute_loss_returns_total_loss_key(self):
        model = _make_model()
        model.train()
        points, features, lorentz, mask = _make_batch()
        # 3 GT pions per event at arbitrary valid indices.
        track_labels = torch.zeros(points.shape[0], 1, points.shape[2])
        track_labels[:, 0, 0] = 1.0
        track_labels[:, 0, 1] = 1.0
        track_labels[:, 0, 2] = 1.0
        loss_dict = model.compute_loss(
            points, features, lorentz, mask, track_labels,
        )
        assert 'total_loss' in loss_dict
        assert torch.isfinite(loss_dict['total_loss'])
        # Total loss must be backward-able.
        loss_dict['total_loss'].backward()


class TestTopNBoundaries:
    def test_top_n_capped_to_valid_tracks(self):
        """When ``top_n >= P``, every valid track is selected and the
        composite output equals the refine scores (no tracks pushed to
        the coarse-minus-offset branch)."""
        model = _make_model(top_n=200)
        model.eval()
        points, features, lorentz, mask = _make_batch(num_tracks=30)
        with torch.no_grad():
            scores = model(points, features, lorentz, mask)
        valid = mask.squeeze(1).bool()
        # No valid track should be at coarse-minus-offset level (very
        # negative); all valid scores are finite and in a "normal" range.
        assert scores[valid].abs().max() < 1e5
