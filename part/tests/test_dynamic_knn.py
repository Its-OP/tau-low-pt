"""Unit tests for dynamic (DGCNN/ParticleNet-style) kNN rebuild in TrackPreFilter.

The dynamic-kNN feature rebuilds the neighbor graph between message-passing
rounds by projecting the evolving per-track embedding into a learned low-
dimensional coordinate space and running L2 kNN there. Round 0 still uses
the static (eta, phi) graph when start_round >= 1; setting start_round = 0
makes every round dynamic (coord projection fires before round 0 too).

TDD: written before implementation. Tests verify:
    - No-op behavior when flag is OFF (state-dict parity, forward bit-parity)
    - Coord projection shape + gradient flow
    - Indices actually change between static and dynamic rounds
    - Padded tracks never enter any dynamic neighbor set
    - Edge features refresh when neighbor set changes
    - start_round controls which rounds are dynamic

Diagnostic contract: setting `model._record_dynamic_info = True` before a
forward pass populates `model._last_dynamic_info` as::

    {
        'neighbor_indices_per_round': list[(B, P, K) long],  # one per round
        'coords_per_round': dict[round_index -> (B, d_coord, P) float],
        'edge_max_pooled_per_round': list[(B, 4, P) float] | None,
        'dynamic_rounds_active': list[int],  # rounds where coord projection drove kNN
    }
"""
from __future__ import annotations

import pytest
import torch

from weaver.nn.model.TrackPreFilter import TrackPreFilter


# ---- Shared configuration ----

BATCH_SIZE = 2
NUM_TRACKS = 64
INPUT_DIM = 7
HIDDEN_DIM = 32
NUM_ROUNDS = 3
NUM_NEIGHBORS = 8
COORD_DIM = 4


def _make_inputs(
    batch_size: int = BATCH_SIZE,
    num_tracks: int = NUM_TRACKS,
    input_dim: int = INPUT_DIM,
    padded_suffix: int = 10,
    seed: int = 42,
):
    """Seeded physically-plausible inputs + mask marking a padded suffix."""
    generator = torch.Generator().manual_seed(seed)

    eta = torch.randn(batch_size, 1, num_tracks, generator=generator) * 1.5
    phi = (
        torch.rand(batch_size, 1, num_tracks, generator=generator) * 2 * 3.14159
        - 3.14159
    )
    points = torch.cat([eta, phi], dim=1)

    features = torch.randn(
        batch_size, input_dim, num_tracks, generator=generator,
    )

    transverse_momentum = (
        torch.rand(batch_size, 1, num_tracks, generator=generator) * 5 + 0.5
    )
    px = transverse_momentum * torch.cos(phi)
    py = transverse_momentum * torch.sin(phi)
    pz = transverse_momentum * torch.sinh(eta)
    pion_mass = 0.13957
    energy = torch.sqrt(px**2 + py**2 + pz**2 + pion_mass**2)
    lorentz_vectors = torch.cat([px, py, pz, energy], dim=1)

    mask = torch.ones(batch_size, 1, num_tracks)
    if padded_suffix > 0:
        mask[:, :, -padded_suffix:] = 0.0

    return points, features, lorentz_vectors, mask


def _baseline_kwargs() -> dict:
    return dict(
        mode='mlp',
        input_dim=INPUT_DIM,
        hidden_dim=HIDDEN_DIM,
        num_neighbors=NUM_NEIGHBORS,
        num_message_rounds=NUM_ROUNDS,
        use_edge_features=True,
        dropout=0.0,
    )


def _dynamic_kwargs(
    start_round: int = 1,
    coord_dim: int = COORD_DIM,
    refresh_edge: bool = True,
) -> dict:
    return dict(
        **_baseline_kwargs(),
        dynamic_knn=True,
        dynamic_knn_start_round=start_round,
        dynamic_knn_coord_dim=coord_dim,
        dynamic_knn_refresh_edge=refresh_edge,
    )


# ---- 1. State-dict parity (flag OFF == baseline) ----


def test_disabled_matches_baseline_state_dict():
    """dynamic_knn=False registers no new parameters."""
    baseline = TrackPreFilter(**_baseline_kwargs())
    explicit_off = TrackPreFilter(**_baseline_kwargs(), dynamic_knn=False)
    assert set(baseline.state_dict().keys()) == set(explicit_off.state_dict().keys())


# ---- 2. Forward bit-parity (flag OFF == baseline) ----


def test_disabled_matches_baseline_forward():
    """dynamic_knn=False forward output bit-matches the baseline."""
    torch.manual_seed(0)
    baseline = TrackPreFilter(**_baseline_kwargs()).eval()
    torch.manual_seed(0)
    explicit_off = TrackPreFilter(**_baseline_kwargs(), dynamic_knn=False).eval()

    points, features, lorentz_vectors, mask = _make_inputs()
    with torch.no_grad():
        scores_baseline = baseline(points, features, lorentz_vectors, mask)
        scores_off = explicit_off(points, features, lorentz_vectors, mask)
    assert torch.equal(scores_baseline, scores_off)


# ---- 3. Coord projection shape ----


def test_coord_projection_shape():
    """Recorded coords tensor has shape (B, coord_dim, P)."""
    model = TrackPreFilter(
        **_dynamic_kwargs(start_round=1, coord_dim=COORD_DIM),
    ).eval()
    model._record_dynamic_info = True

    points, features, lorentz_vectors, mask = _make_inputs()
    with torch.no_grad():
        _ = model(points, features, lorentz_vectors, mask)

    info = model._last_dynamic_info
    assert info is not None
    assert info['coords_per_round'], 'no coords recorded'
    for _round_index, coords in info['coords_per_round'].items():
        assert coords.shape == (BATCH_SIZE, COORD_DIM, NUM_TRACKS)


# ---- 4. Dynamic neighbor indices differ from static ones ----


def test_neighbor_indices_change_across_rounds():
    """Round-0 static indices differ from round-1 dynamic indices."""
    model = TrackPreFilter(
        **_dynamic_kwargs(start_round=1, coord_dim=COORD_DIM),
    ).eval()
    model._record_dynamic_info = True

    points, features, lorentz_vectors, mask = _make_inputs()
    with torch.no_grad():
        _ = model(points, features, lorentz_vectors, mask)

    per_round = model._last_dynamic_info['neighbor_indices_per_round']
    assert len(per_round) == NUM_ROUNDS
    assert per_round[0].shape == per_round[1].shape
    assert not torch.equal(per_round[0], per_round[1]), (
        'dynamic kNN produced identical indices to the static graph'
    )


# ---- 5. Padded tracks never appear in any dynamic neighbor set ----


def test_padded_tracks_never_selected():
    """No dynamic-round neighbor index points to a padded track position."""
    model = TrackPreFilter(
        **_dynamic_kwargs(start_round=1, coord_dim=COORD_DIM),
    ).eval()
    model._record_dynamic_info = True

    padded_suffix = 10
    points, features, lorentz_vectors, mask = _make_inputs(
        padded_suffix=padded_suffix,
    )
    valid_count = NUM_TRACKS - padded_suffix

    with torch.no_grad():
        _ = model(points, features, lorentz_vectors, mask)

    info = model._last_dynamic_info
    active_rounds = info['dynamic_rounds_active']
    per_round = info['neighbor_indices_per_round']
    assert active_rounds, 'no dynamic rounds recorded'
    for round_index in active_rounds:
        neighbor_indices = per_round[round_index]
        assert torch.all(neighbor_indices < valid_count), (
            f'round {round_index}: neighbor index points to padded track'
        )


# ---- 6. Gradient flows to the coord projection ----


def test_gradient_flows_to_coord_projection():
    """loss.backward() populates non-zero grad on coord_projection weights."""
    model = TrackPreFilter(
        **_dynamic_kwargs(start_round=1, coord_dim=COORD_DIM),
    ).train()

    points, features, lorentz_vectors, mask = _make_inputs()
    scores = model(points, features, lorentz_vectors, mask)
    valid_mask = mask.squeeze(1).bool()
    loss = scores[valid_mask].sum()
    loss.backward()

    coord_weight_grads = [
        param.grad for name, param in model.named_parameters()
        if 'coord_projection' in name and param.ndim >= 2
    ]
    assert coord_weight_grads, 'coord_projection weight parameter not found'
    for grad in coord_weight_grads:
        assert grad is not None
        assert torch.isfinite(grad).all()
    total = sum(g.abs().sum() for g in coord_weight_grads)
    assert total > 0, 'gradient did not reach coord_projection'


# ---- 7. Edge features refresh when indices change ----


def test_edge_features_refresh_when_enabled():
    """With refresh_edge=True, edge_max_pooled differs between static and dynamic rounds."""
    model = TrackPreFilter(
        **_dynamic_kwargs(start_round=1, coord_dim=COORD_DIM, refresh_edge=True),
    ).eval()
    model._record_dynamic_info = True

    points, features, lorentz_vectors, mask = _make_inputs()
    with torch.no_grad():
        _ = model(points, features, lorentz_vectors, mask)

    edge_per_round = model._last_dynamic_info['edge_max_pooled_per_round']
    assert edge_per_round is not None
    assert len(edge_per_round) >= 2
    assert not torch.equal(edge_per_round[0], edge_per_round[1]), (
        'Edge features identical across static + dynamic rounds despite refresh_edge=True'
    )


# ---- 8. start_round controls which rounds are dynamic ----


def test_start_round_controls_active_rounds():
    """start_round=1 keeps round 0 static; start_round=0 makes round 0 dynamic."""
    model_from_1 = TrackPreFilter(**_dynamic_kwargs(start_round=1)).eval()
    model_from_0 = TrackPreFilter(**_dynamic_kwargs(start_round=0)).eval()
    model_from_1._record_dynamic_info = True
    model_from_0._record_dynamic_info = True

    points, features, lorentz_vectors, mask = _make_inputs()
    with torch.no_grad():
        _ = model_from_1(points, features, lorentz_vectors, mask)
        _ = model_from_0(points, features, lorentz_vectors, mask)

    active_from_1 = model_from_1._last_dynamic_info['dynamic_rounds_active']
    active_from_0 = model_from_0._last_dynamic_info['dynamic_rounds_active']

    assert 0 not in active_from_1
    assert 1 in active_from_1 and 2 in active_from_1
    assert 0 in active_from_0
    assert 1 in active_from_0 and 2 in active_from_0
