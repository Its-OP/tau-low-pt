"""Tests for diagnostics.eval_cascade_pipeline.

Covers:
- Sorting + index-mapping correctness (hand-crafted tensors).
- Per-stage output shape (synthetic models).
- Loader round-trip for stages 1/2/3 (tmp_path checkpoints).
- Parquet schema and idempotent writes.
"""
from __future__ import annotations

import os
import sys

import pyarrow.parquet as pq
import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from diagnostics.eval_cascade_pipeline import (
    OUTPUT_SCHEMA,
    _evaluate_batch,
    _gather_along_tracks,
    _load_stage1,
    _load_stage2,
    _load_stage3,
    _write_parquet,
)
from weaver.nn.model.CoupleReranker import CoupleReranker
from weaver.nn.model.TrackPreFilter import TrackPreFilter


BATCH_SIZE = 2
NUM_TRACKS = 40
NUM_VALID = 30
INPUT_DIM = 16
TOP_K1 = 12
TOP_K2 = 6


def _make_inputs(batch_size=BATCH_SIZE, num_tracks=NUM_TRACKS, seed=0):
    g = torch.Generator().manual_seed(seed)
    eta = torch.randn(batch_size, 1, num_tracks, generator=g) * 1.5
    phi = torch.rand(batch_size, 1, num_tracks, generator=g) * 6.28 - 3.14
    points = torch.cat([eta, phi], dim=1)
    features = torch.randn(batch_size, INPUT_DIM, num_tracks, generator=g)
    pt = torch.rand(batch_size, 1, num_tracks, generator=g) * 5 + 0.5
    px, py = pt * torch.cos(phi), pt * torch.sin(phi)
    pz = pt * torch.sinh(eta)
    energy = torch.sqrt(px**2 + py**2 + pz**2 + 0.13957**2)
    lorentz = torch.cat([px, py, pz, energy], dim=1)
    mask = torch.ones(batch_size, 1, num_tracks)
    mask[:, :, NUM_VALID:] = 0.0
    return points, features, lorentz, mask


class _ScriptedStage1(nn.Module):
    """Stage 1 stub returning a fixed (B, P) score tensor."""

    def __init__(self, scores: torch.Tensor):
        super().__init__()
        self.scores = scores
        self._dummy = nn.Parameter(torch.zeros(1))

    def forward(self, points, features, lorentz, mask):
        return self.scores

    def select_top_k(self, scores, mask, top_k):
        # Mirror TrackPreFilter.select_top_k — argmax over masked scores.
        mask_bool = mask.squeeze(1).bool()
        masked = torch.where(
            mask_bool, scores, torch.full_like(scores, float('-inf')),
        )
        return masked.topk(top_k, dim=1).indices


class _ScriptedStage2(nn.Module):
    def __init__(self, scores: torch.Tensor):
        super().__init__()
        self.scores = scores
        self._dummy = nn.Parameter(torch.zeros(1))

    def forward(self, points, features, lorentz, mask, stage1_scores):
        # Apply -inf to padding positions to mirror CascadeReranker.
        mask_bool = mask.squeeze(1).bool()
        return torch.where(
            mask_bool, self.scores, torch.full_like(self.scores, float('-inf')),
        )


class _ScriptedStage3(nn.Module):
    def __init__(self, couple_scores: torch.Tensor):
        super().__init__()
        self.couple_scores = couple_scores
        self._dummy = nn.Parameter(torch.zeros(1))

    def forward(self, couple_features, k2_features=None):
        return self.couple_scores


# ---------------------------------------------------------------------------
# Sorting correctness
# ---------------------------------------------------------------------------

class TestSorting:
    def test_stage1_sort_order(self):
        # 1 event, 3 tracks, scores [1, 3, 2] → sorted desc → [1, 2, 0].
        scores = torch.tensor([[1.0, 3.0, 2.0]])
        points = torch.zeros(1, 2, 3)
        features = torch.zeros(1, INPUT_DIM, 3)
        lorentz = torch.zeros(1, 4, 3)
        mask = torch.ones(1, 1, 3)
        stage1 = _ScriptedStage1(scores)

        rows = _evaluate_batch(
            stage='stage1',
            stage1=stage1, stage2=None, stage3=None,
            points=points, features=features, lorentz=lorentz, mask=mask,
            top_k1=None, top_k2=None, num_couples=200, pair_flags={},
        )
        assert rows[0]['stage1_sorted_indices'] == [1, 2, 0]

    def test_stage1_excludes_padding(self):
        # 1 event, 4 tracks, last 2 are padding.
        scores = torch.tensor([[5.0, 1.0, 9.0, 9.0]])
        mask = torch.tensor([[[1.0, 1.0, 0.0, 0.0]]])
        stage1 = _ScriptedStage1(scores)
        rows = _evaluate_batch(
            stage='stage1', stage1=stage1, stage2=None, stage3=None,
            points=torch.zeros(1, 2, 4), features=torch.zeros(1, INPUT_DIM, 4),
            lorentz=torch.zeros(1, 4, 4), mask=mask,
            top_k1=None, top_k2=None, num_couples=200, pair_flags={},
        )
        assert rows[0]['stage1_sorted_indices'] == [0, 1]

    def test_stage2_mapping_to_original(self):
        # 1 event, 5 tracks, top-K1=3 picks tracks at original positions
        # [2, 0, 3] (highest stage1 scores). Stage 2 scores = [0.5, 0.9, 0.1].
        # Sorted desc: K1 positions [1, 0, 2] → original [0, 2, 3].
        s1_scores = torch.tensor([[0.7, 0.1, 0.9, 0.6, 0.2]])
        s2_scores = torch.tensor([[0.5, 0.9, 0.1]])
        mask = torch.ones(1, 1, 5)
        stage1 = _ScriptedStage1(s1_scores)
        stage2 = _ScriptedStage2(s2_scores)

        rows = _evaluate_batch(
            stage='part', stage1=stage1, stage2=stage2, stage3=None,
            points=torch.zeros(1, 2, 5), features=torch.zeros(1, INPUT_DIM, 5),
            lorentz=torch.zeros(1, 4, 5), mask=mask,
            top_k1=3, top_k2=None, num_couples=200, pair_flags={},
        )
        assert rows[0]['stage2_sorted_indices'] == [0, 2, 3]


# ---------------------------------------------------------------------------
# Per-stage output structure (real Stage 1; synthetic Stage 2 / Stage 3)
# ---------------------------------------------------------------------------

class TestPerStageOutput:
    def _stage1(self):
        return TrackPreFilter(
            mode='mlp', input_dim=INPUT_DIM, hidden_dim=32,
            num_message_rounds=1, num_neighbors=4,
        ).eval()

    def test_stage1_length_equals_n_valid(self):
        stage1 = self._stage1()
        points, features, lorentz, mask = _make_inputs()
        rows = _evaluate_batch(
            stage='stage1', stage1=stage1, stage2=None, stage3=None,
            points=points, features=features, lorentz=lorentz, mask=mask,
            top_k1=None, top_k2=None, num_couples=200, pair_flags={},
        )
        for row in rows:
            assert len(row['stage1_sorted_indices']) == NUM_VALID
            assert all(idx < NUM_VALID for idx in row['stage1_sorted_indices'])
            assert row['stage2_sorted_indices'] == []
            assert row['stage3_sorted_couples'] == []


# ---------------------------------------------------------------------------
# Couple enumeration
# ---------------------------------------------------------------------------

class TestCoupleEnumeration:
    def test_top_couple_pair_recovers_original_indices(self):
        # Hand-craft a 1-event scenario where the full pipeline stays
        # deterministic enough to assert specific couple pairs.
        # K2=3 → 3 couples in triu order: (0,1), (0,2), (1,2).
        # k2_orig=[10,20,30]; couple_scores=[0.1, 0.9, 0.5].
        # Top couple → (0,2) → original [10, 30].
        # Top 2     → [(0,2),(1,2)] → [[10,30], [20,30]].

        # Build scripted models that produce the desired top-K2 selection.
        # Simplest: rig stage 1 + stage 2 so top-K1 selects [0,1,2,...], then
        # top-K2 picks [10,20,30] in the original space.

        # Use 6-track event so top-K1=4 pulls original indices [10,20,30,X],
        # then top-K2=3 picks the first three in order.
        # Here we simplify by setting num_tracks=3 directly.
        scores_s1 = torch.tensor([[0.9, 0.5, 0.1]])
        scores_s2 = torch.tensor([[0.5, 0.5, 0.5]])
        couple_scores = torch.tensor([[0.1, 0.9, 0.5]])

        stage1 = _ScriptedStage1(scores_s1)
        stage2 = _ScriptedStage2(scores_s2)
        stage3 = _ScriptedStage3(couple_scores)

        points = torch.zeros(1, 2, 3)
        features = torch.zeros(1, INPUT_DIM, 3)
        lorentz = torch.zeros(1, 4, 3)
        # Make lorentz produce reasonable invariant masses so filter_a passes.
        # All-zero lorentz → m=0 → m <= m_tau → filter passes.
        mask = torch.ones(1, 1, 3)

        rows = _evaluate_batch(
            stage='couples', stage1=stage1, stage2=stage2, stage3=stage3,
            points=points, features=features, lorentz=lorentz, mask=mask,
            top_k1=3, top_k2=3, num_couples=200,
            pair_flags={
                'pair_kinematics_v2': False, 'pair_physics_v3': False,
                'pair_physics_signif': False,
            },
        )
        couples = rows[0]['stage3_sorted_couples']
        # original indices match positions 0/1/2 (since top-K1 sort stable on
        # descending scores [0.9,0.5,0.1] → [0,1,2]).
        # Top couple (highest score 0.9, idx 1 in triu = (0,2)) → [0, 2].
        assert couples[0] == [0, 2]
        assert couples[1] == [1, 2]
        assert couples[2] == [0, 1]

    def test_num_couples_cap(self):
        scores_s1 = torch.tensor([[0.9, 0.5, 0.1]])
        scores_s2 = torch.tensor([[0.5, 0.5, 0.5]])
        couple_scores = torch.tensor([[0.1, 0.9, 0.5]])
        stage1 = _ScriptedStage1(scores_s1)
        stage2 = _ScriptedStage2(scores_s2)
        stage3 = _ScriptedStage3(couple_scores)

        rows = _evaluate_batch(
            stage='couples', stage1=stage1, stage2=stage2, stage3=stage3,
            points=torch.zeros(1, 2, 3), features=torch.zeros(1, INPUT_DIM, 3),
            lorentz=torch.zeros(1, 4, 3), mask=torch.ones(1, 1, 3),
            top_k1=3, top_k2=3, num_couples=2,
            pair_flags={
                'pair_kinematics_v2': False, 'pair_physics_v3': False,
                'pair_physics_signif': False,
            },
        )
        assert len(rows[0]['stage3_sorted_couples']) == 2

    def test_couple_pair_distinct(self):
        scores_s1 = torch.tensor([[0.9, 0.5, 0.1]])
        scores_s2 = torch.tensor([[0.5, 0.5, 0.5]])
        couple_scores = torch.tensor([[0.1, 0.9, 0.5]])
        stage1 = _ScriptedStage1(scores_s1)
        stage2 = _ScriptedStage2(scores_s2)
        stage3 = _ScriptedStage3(couple_scores)
        rows = _evaluate_batch(
            stage='couples', stage1=stage1, stage2=stage2, stage3=stage3,
            points=torch.zeros(1, 2, 3), features=torch.zeros(1, INPUT_DIM, 3),
            lorentz=torch.zeros(1, 4, 3), mask=torch.ones(1, 1, 3),
            top_k1=3, top_k2=3, num_couples=200,
            pair_flags={
                'pair_kinematics_v2': False, 'pair_physics_v3': False,
                'pair_physics_signif': False,
            },
        )
        for couple in rows[0]['stage3_sorted_couples']:
            assert couple[0] != couple[1]


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

class _FakeDataConfig:
    def __init__(self, input_dim=INPUT_DIM):
        self.input_dicts = {'pf_features': list(range(input_dim))}


class TestLoaders:
    def test_load_stage1_roundtrip(self, tmp_path):
        original = TrackPreFilter(
            mode='mlp', input_dim=INPUT_DIM, hidden_dim=32,
            num_message_rounds=1, num_neighbors=4,
        )
        path = tmp_path / 'stage1.pt'
        torch.save({'model_state_dict': original.state_dict()}, path)

        loaded = _load_stage1(
            str(path), _FakeDataConfig(),
            num_neighbors=4, device='cpu',
        )
        # Compare a representative parameter.
        original_w = original.track_mlp[0].weight
        loaded_w = loaded.track_mlp[0].weight
        assert torch.allclose(original_w, loaded_w)

    def test_load_stage2_top_k1_from_args(self, tmp_path):
        # Build a CascadeReranker with default knobs, save with args dict.
        from weaver.nn.model.CascadeReranker import CascadeReranker
        stage2 = CascadeReranker(
            input_dim=INPUT_DIM,
            embed_dim=64, num_heads=4, num_layers=1,
            pair_input_dim=4, pair_extra_dim=6,
            pair_embed_dims=[32, 32],
            pair_embed_mode='concat', ffn_ratio=4,
            dropout=0.0, loss_mode='pairwise', rs_at_k_target=200,
        )
        path = tmp_path / 'stage2.pt'
        torch.save({
            'model_state_dict': stage2.state_dict(),
            'args': {
                'top_k1': 999,
                'stage2_embed_dim': 64,
                'stage2_num_heads': 4,
                'stage2_num_layers': 1,
                'stage2_pair_embed_dims': '32,32',
                'stage2_pair_extra_dim': 6,
                'stage2_pair_embed_mode': 'concat',
                'stage2_ffn_ratio': 4,
                'stage2_dropout': 0.0,
                'stage2_loss_mode': 'pairwise',
                'stage2_rs_at_k_target': 200,
            },
        }, path)

        _, top_k1 = _load_stage2(str(path), input_dim=INPUT_DIM, device='cpu')
        assert top_k1 == 999

    def test_load_stage3_input_dim_with_v2(self, tmp_path):
        from utils.couple_features import (
            COUPLE_FEATURE_DIM_V2,
        )
        stage3 = CoupleReranker(
            hidden_dim=64, num_residual_blocks=1, dropout=0.0,
            rest_dim=COUPLE_FEATURE_DIM_V2 - 32,
        )
        path = tmp_path / 'stage3.pt'
        torch.save({
            'couple_reranker_state_dict': stage3.state_dict(),
            'args': {
                'pair_kinematics_v2': True,
                'pair_physics_v3': False,
                'pair_physics_signif': False,
                'couple_hidden_dim': 64,
                'couple_num_residual_blocks': 1,
                'couple_dropout': 0.0,
                'top_k2': 7,
            },
        }, path)
        loaded, top_k2, flags = _load_stage3(str(path), device='cpu')
        assert top_k2 == 7
        assert flags['pair_kinematics_v2'] is True
        assert loaded.input_dim == COUPLE_FEATURE_DIM_V2

    def test_strict_false_legacy_running_stats(self, tmp_path):
        original = TrackPreFilter(
            mode='mlp', input_dim=INPUT_DIM, hidden_dim=32,
            num_message_rounds=1, num_neighbors=4,
        )
        state = dict(original.state_dict())
        # Inject a legacy running_mean key.
        state['legacy_running_mean'] = torch.zeros(32)
        path = tmp_path / 'stage1_legacy.pt'
        torch.save({'model_state_dict': state}, path)

        loaded = _load_stage1(
            str(path), _FakeDataConfig(),
            num_neighbors=4, device='cpu',
        )
        assert loaded is not None  # didn't raise


# ---------------------------------------------------------------------------
# Parquet schema
# ---------------------------------------------------------------------------

class TestParquetSchema:
    def _key(self, run=1, eid=2):
        return {
            'event_run': run, 'event_id': eid,
            'event_luminosity_block': 3, 'source_batch_id': 4,
            'source_microbatch_id': 5,
        }

    def test_stage1_schema(self, tmp_path):
        path = tmp_path / 'out.parquet'
        rows = [{
            **self._key(),
            'stage': 'stage1',
            'stage1_sorted_indices': [3, 1, 0, 2],
            'stage2_sorted_indices': [],
            'stage3_sorted_couples': [],
        }]
        _write_parquet(rows, str(path))
        table = pq.read_table(str(path))
        assert set(table.column_names) == {f.name for f in OUTPUT_SCHEMA}
        df = table.to_pandas()
        assert df.iloc[0]['stage1_sorted_indices'].tolist() == [3, 1, 0, 2]
        assert df.iloc[0]['stage2_sorted_indices'].tolist() == []
        assert df.iloc[0]['stage3_sorted_couples'].tolist() == []

    def test_couples_schema(self, tmp_path):
        path = tmp_path / 'out.parquet'
        rows = [{
            **self._key(),
            'stage': 'couples',
            'stage1_sorted_indices': [0, 1],
            'stage2_sorted_indices': [0, 1],
            'stage3_sorted_couples': [[0, 1], [1, 0]],
        }]
        _write_parquet(rows, str(path))
        table = pq.read_table(str(path))
        df = table.to_pandas()
        couples = df.iloc[0]['stage3_sorted_couples'].tolist()
        # Each couple is exactly 2 ints.
        for couple in couples:
            assert len(couple) == 2

    def test_idempotent_write(self, tmp_path):
        path = tmp_path / 'out.parquet'
        rows = [{
            **self._key(),
            'stage': 'stage1',
            'stage1_sorted_indices': [0],
            'stage2_sorted_indices': [],
            'stage3_sorted_couples': [],
        }]
        _write_parquet(rows, str(path))
        _write_parquet(rows, str(path))   # second write must not corrupt
        table = pq.read_table(str(path))
        assert table.num_rows == 1


# ---------------------------------------------------------------------------
# Gather helper
# ---------------------------------------------------------------------------

class TestGather:
    def test_gather_along_tracks_shape(self):
        tensor = torch.randn(2, 4, 6)
        idx = torch.tensor([[0, 2, 4], [1, 3, 5]])
        out = _gather_along_tracks(tensor, idx)
        assert out.shape == (2, 4, 3)

    def test_gather_along_tracks_values(self):
        tensor = torch.arange(24).reshape(2, 4, 3).float()
        idx = torch.tensor([[2, 0], [1, 2]])
        out = _gather_along_tracks(tensor, idx)
        # For batch 0, channel 0: tensor[0, 0, [2, 0]] = [2, 0]
        assert out[0, 0].tolist() == [2.0, 0.0]
        # For batch 1, channel 3: tensor[1, 3, [1, 2]] = [22, 23]
        assert out[1, 3].tolist() == [22.0, 23.0]
