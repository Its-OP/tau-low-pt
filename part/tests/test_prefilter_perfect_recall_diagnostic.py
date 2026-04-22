"""Tests for ``diagnostics.prefilter_perfect_recall_diagnostic``.

Stratifies the prefilter's Stage-1 output into pass/fail buckets by
per-event P@K perfect recall, then surfaces the features that
distinguish them. This test module pins the pure-function behaviour
used by the diagnostic: P@K primitives, per-event/per-pion feature
extraction, (η, φ) kNN GT-neighbour counting, stratification, and
Gaussian-discriminability (d') summaries. One end-of-module test also
exercises the composite-key parquet loader against a tmp-path fixture
so the IO path stays covered.
"""
from __future__ import annotations

import math

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from diagnostics.prefilter_perfect_recall_diagnostic import (
    compare_distributions,
    count_gt_in_top_k,
    count_gt_neighbors_eta_phi,
    event_perfect_recall,
    gt_rank_in_top_k,
    load_val_track_features,
    miss_cooccurrence_histogram,
    per_event_features,
    per_pion_features,
    stratify_by_p_at_k,
)


# ---------------------------------------------------------------------------
# P@K primitives
# ---------------------------------------------------------------------------


class TestEventPerfectRecall:
    def test_all_gt_in_top_k_returns_true(self):
        assert event_perfect_recall(
            gt_indices=[3, 7, 20],
            top_k_indices=[3, 5, 7, 12, 20, 40],
            K=6,
        ) is True

    def test_missing_one_gt_returns_false(self):
        assert event_perfect_recall(
            gt_indices=[3, 7, 20],
            top_k_indices=[3, 5, 7, 12, 40],  # 20 missing
            K=5,
        ) is False

    def test_k_truncation_drops_out_of_range_gt(self):
        # GT at position 6 (1-indexed = 6, 0-indexed = 5); K=5 must
        # exclude it.
        top_k = [3, 5, 7, 12, 40, 20]  # 20 is at 0-index 5
        assert event_perfect_recall([3, 7, 20], top_k, K=5) is False
        assert event_perfect_recall([3, 7, 20], top_k, K=6) is True

    def test_empty_gt_returns_true(self):
        # Vacuously — no GT to miss. Diagnostic treats this as pass.
        assert event_perfect_recall([], [1, 2, 3], K=3) is True


class TestCountGtInTopK:
    def test_all_three_in_top(self):
        assert count_gt_in_top_k([3, 7, 20], [3, 7, 20, 40, 50], K=5) == 3

    def test_partial_overlap(self):
        assert count_gt_in_top_k([3, 7, 20], [3, 99, 7, 100, 200], K=5) == 2

    def test_k_truncation(self):
        # GT 20 is at 0-index 4 → K=4 excludes it.
        assert count_gt_in_top_k([3, 7, 20], [3, 7, 9, 11, 20], K=4) == 2
        assert count_gt_in_top_k([3, 7, 20], [3, 7, 9, 11, 20], K=5) == 3

    def test_zero_matches(self):
        assert count_gt_in_top_k([1, 2, 3], [10, 20, 30], K=3) == 0


class TestGtRankInTopK:
    def test_gt_at_rank_1(self):
        assert gt_rank_in_top_k(gt_index=3, top_k_indices=[3, 7, 20]) == 1

    def test_gt_at_rank_3(self):
        assert gt_rank_in_top_k(20, [3, 7, 20, 40]) == 3

    def test_gt_absent_returns_none(self):
        assert gt_rank_in_top_k(99, [3, 7, 20]) is None

    def test_beyond_k_returns_none(self):
        # GT at 0-index 5 → rank 6. K=4 should mask it to None.
        assert gt_rank_in_top_k(20, [1, 2, 3, 4, 5, 20], K=4) is None
        assert gt_rank_in_top_k(20, [1, 2, 3, 4, 5, 20], K=6) == 6


# ---------------------------------------------------------------------------
# Per-event feature extraction
# ---------------------------------------------------------------------------


def _make_track_arrays(
    n: int,
    pt: np.ndarray | None = None,
    eta: np.ndarray | None = None,
    phi: np.ndarray | None = None,
    dxy_sig: np.ndarray | None = None,
    dz_sig: np.ndarray | None = None,
    dca_sig: np.ndarray | None = None,
    pt_error: np.ndarray | None = None,
    norm_chi2: np.ndarray | None = None,
    n_pixel_hits: np.ndarray | None = None,
    cov_phi_phi: np.ndarray | None = None,
    cov_lambda_lambda: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    def _default(x, fill):
        return x if x is not None else np.full(n, fill, dtype=np.float32)

    return {
        'pt': _default(pt, 1.0),
        'eta': _default(eta, 0.0),
        'phi': _default(phi, 0.0),
        'dxy_significance': _default(dxy_sig, 0.0),
        'dz_significance': _default(dz_sig, 0.0),
        'dca_significance': _default(dca_sig, 0.0),
        'pt_error': _default(pt_error, 0.01),
        'norm_chi2': _default(norm_chi2, 1.0),
        'n_pixel_hits': (
            n_pixel_hits.astype(np.int32)
            if n_pixel_hits is not None
            else np.full(n, 4, dtype=np.int32)
        ),
        'covariance_phi_phi': _default(cov_phi_phi, 1e-6),
        'covariance_lambda_lambda': _default(cov_lambda_lambda, 1e-6),
    }


class TestPerEventFeatures:
    def test_returns_all_expected_keys(self):
        tracks = _make_track_arrays(
            5, pt=np.array([0.5, 1.0, 1.5, 2.0, 3.0], dtype=np.float32),
        )
        feats = per_event_features(
            tracks, gt_indices=[0, 1, 2], vertex_z=0.5,
        )
        expected = {
            'n_tracks',
            'event_pt_median', 'event_pt_max', 'event_pt_std',
            'event_pt_p95',
            'mean_abs_dz_sig', 'mean_abs_dxy_sig', 'mean_chi2',
            'gt_pt_min', 'gt_pt_mean', 'gt_pt_max', 'gt_pt_sum',
            'gt_pt_spread',
            'vertex_z',
        }
        assert expected <= set(feats)

    def test_gt_pt_aggregates_are_correct(self):
        tracks = _make_track_arrays(
            4, pt=np.array([0.5, 2.0, 1.0, 3.0], dtype=np.float32),
        )
        feats = per_event_features(
            tracks, gt_indices=[0, 1, 2], vertex_z=0.0,
        )
        # GT pT = {0.5, 2.0, 1.0}
        assert feats['gt_pt_min'] == pytest.approx(0.5, abs=1e-5)
        assert feats['gt_pt_max'] == pytest.approx(2.0, abs=1e-5)
        assert feats['gt_pt_mean'] == pytest.approx((0.5 + 2.0 + 1.0) / 3, abs=1e-5)
        assert feats['gt_pt_sum'] == pytest.approx(3.5, abs=1e-5)
        assert feats['gt_pt_spread'] == pytest.approx(1.5, abs=1e-5)

    def test_n_tracks_matches_pt_length(self):
        tracks = _make_track_arrays(7)
        feats = per_event_features(
            tracks, gt_indices=[0, 1, 2], vertex_z=0.0,
        )
        assert feats['n_tracks'] == 7


class TestPerPionFeatures:
    def test_all_feature_keys_present(self):
        tracks = _make_track_arrays(
            3,
            pt=np.array([2.0, 1.0, 0.5], dtype=np.float32),
            dxy_sig=np.array([1.0, 2.0, 3.0], dtype=np.float32),
            dz_sig=np.array([0.5, 1.5, 2.5], dtype=np.float32),
            norm_chi2=np.array([1.0, 1.2, 1.4], dtype=np.float32),
        )
        feats = per_pion_features(1, tracks)
        expected = {
            'pt', 'pt_rank_in_event', 'dxy_significance',
            'dz_significance', 'dca_significance', 'pt_error',
            'norm_chi2', 'n_pixel_hits', 'covariance_phi_phi',
            'covariance_lambda_lambda',
        }
        assert expected <= set(feats)

    def test_pt_rank_is_one_for_hardest(self):
        tracks = _make_track_arrays(
            4, pt=np.array([0.5, 3.0, 1.0, 2.0], dtype=np.float32),
        )
        # index 1 has the max pT → rank 1
        feats = per_pion_features(1, tracks)
        assert feats['pt_rank_in_event'] == 1

    def test_pt_rank_is_n_tracks_for_softest(self):
        tracks = _make_track_arrays(
            4, pt=np.array([0.5, 3.0, 1.0, 2.0], dtype=np.float32),
        )
        # index 0 has the min pT → rank 4
        feats = per_pion_features(0, tracks)
        assert feats['pt_rank_in_event'] == 4


# ---------------------------------------------------------------------------
# kNN GT-neighbour counting (D4)
# ---------------------------------------------------------------------------


class TestCountGtNeighborsEtaPhi:
    def test_two_nearby_gt_plus_far_background(self):
        # 3 GT at (η, φ) close together, 50 background far away.
        gt_eta = [0.0, 0.05, 0.1]
        gt_phi = [0.0, 0.05, 0.1]
        bg_eta = list(np.linspace(1.0, 2.0, 50))
        bg_phi = list(np.linspace(1.0, 2.0, 50))
        track_eta = np.array(gt_eta + bg_eta, dtype=np.float32)
        track_phi = np.array(gt_phi + bg_phi, dtype=np.float32)
        gt_indices = [0, 1, 2]
        # kNN=16 of GT 0 → the 2 other GT are its closest neighbours.
        assert count_gt_neighbors_eta_phi(
            pion_index=0,
            gt_indices=gt_indices,
            track_eta=track_eta,
            track_phi=track_phi,
            k=16,
        ) == 2

    def test_phi_wraparound(self):
        # GT at φ = -3.0 and φ = 3.0 → angular distance ≈ 0.28 (via
        # wrap), not 6.0. They should be mutual neighbours even with a
        # busy background.
        gt_eta = [0.0, 0.0]
        gt_phi = [-3.0, 3.0]
        bg_eta = list(np.linspace(1.0, 2.0, 50))
        bg_phi = list(np.linspace(-2.0, 2.0, 50))
        track_eta = np.array(gt_eta + bg_eta, dtype=np.float32)
        track_phi = np.array(gt_phi + bg_phi, dtype=np.float32)
        gt_indices = [0, 1]
        assert count_gt_neighbors_eta_phi(
            pion_index=0,
            gt_indices=gt_indices,
            track_eta=track_eta,
            track_phi=track_phi,
            k=16,
        ) == 1

    def test_isolated_gt_returns_zero(self):
        # 1 GT alone in a sea of background.
        track_eta = np.concatenate([[0.0], np.linspace(1.5, 2.5, 100)]).astype(np.float32)
        track_phi = np.concatenate([[0.0], np.linspace(1.5, 2.5, 100)]).astype(np.float32)
        assert count_gt_neighbors_eta_phi(
            pion_index=0, gt_indices=[0],
            track_eta=track_eta, track_phi=track_phi, k=16,
        ) == 0

    def test_self_excluded_from_knn(self):
        # Single GT with itself at (0,0) and 20 backgrounds further out
        # — its own position should not inflate the GT-neighbour count.
        track_eta = np.concatenate([[0.0], np.linspace(0.5, 1.5, 20)]).astype(np.float32)
        track_phi = np.concatenate([[0.0], np.linspace(0.5, 1.5, 20)]).astype(np.float32)
        assert count_gt_neighbors_eta_phi(
            pion_index=0, gt_indices=[0],
            track_eta=track_eta, track_phi=track_phi, k=16,
        ) == 0


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


class TestStratifyByPAtK:
    def test_correct_partition(self):
        events = [
            {'gt_indices': [0, 1, 2], 'top_k_indices': [0, 1, 2, 3, 4]},  # pass
            {'gt_indices': [0, 1, 2], 'top_k_indices': [0, 1, 99, 3, 4]},  # fail
            {'gt_indices': [5, 6, 7], 'top_k_indices': [5, 6, 7, 8, 9]},  # pass
        ]
        groups = stratify_by_p_at_k(events, K=5)
        assert len(groups['pass']) == 2
        assert len(groups['fail']) == 1

    def test_k_cap_partitions_correctly(self):
        # Same events at K=3 and K=5 → potentially different partitions.
        events = [
            {'gt_indices': [0, 1, 2], 'top_k_indices': [0, 1, 99, 2, 3]},
        ]
        assert len(stratify_by_p_at_k(events, K=3)['fail']) == 1
        assert len(stratify_by_p_at_k(events, K=5)['pass']) == 1


class TestCompareDistributions:
    def test_d_prime_zero_for_identical_distributions(self):
        a = [{'x': 0.0}, {'x': 1.0}, {'x': 2.0}, {'x': 3.0}]
        b = [{'x': 0.0}, {'x': 1.0}, {'x': 2.0}, {'x': 3.0}]
        rows = compare_distributions(a, b, feature_names=['x'])
        row = rows[0]
        assert row['feature'] == 'x'
        assert abs(row['d_prime']) < 1e-5

    def test_d_prime_one_for_unit_sigma_shift(self):
        # N(0, 1) vs N(1, 1): d' = 1 / sqrt((1+1)/2) = 1.
        rng = np.random.default_rng(0)
        a = [{'x': float(v)} for v in rng.normal(0.0, 1.0, 20000)]
        b = [{'x': float(v)} for v in rng.normal(1.0, 1.0, 20000)]
        rows = compare_distributions(a, b, feature_names=['x'])
        assert abs(rows[0]['d_prime'] - 1.0) < 0.05

    def test_returns_mean_median_std_n(self):
        a = [{'x': 1.0}, {'x': 2.0}, {'x': 3.0}]
        b = [{'x': 4.0}, {'x': 5.0}]
        rows = compare_distributions(a, b, feature_names=['x'])
        row = rows[0]
        assert row['mean_a'] == pytest.approx(2.0)
        assert row['mean_b'] == pytest.approx(4.5)
        assert row['median_a'] == pytest.approx(2.0)
        assert row['median_b'] == pytest.approx(4.5)
        assert row['n_a'] == 3
        assert row['n_b'] == 2


class TestMissCooccurrenceHistogram:
    def test_sums_to_total_events(self):
        events = [
            {'gt_indices': [0, 1, 2], 'top_k_indices': [0, 1, 2, 3]},  # 3
            {'gt_indices': [0, 1, 2], 'top_k_indices': [0, 1, 3, 4]},  # 2
            {'gt_indices': [0, 1, 2], 'top_k_indices': [0, 3, 4, 5]},  # 1
            {'gt_indices': [0, 1, 2], 'top_k_indices': [3, 4, 5, 6]},  # 0
        ]
        hist = miss_cooccurrence_histogram(events, K=4)
        assert hist[3] == 1
        assert hist[2] == 1
        assert hist[1] == 1
        assert hist[0] == 1
        assert sum(hist.values()) == 4


# ---------------------------------------------------------------------------
# IO — composite-key val loader
# ---------------------------------------------------------------------------


class TestLoadValTrackFeatures:
    def _write_val_fixture(self, tmp_path, rows: list[dict]):
        """Write a minimal val-style parquet file from ``rows``."""
        columns = {
            'event_run': pa.array([r['event_run'] for r in rows], type=pa.int32()),
            'event_id': pa.array([r['event_id'] for r in rows], type=pa.int64()),
            'event_luminosity_block': pa.array(
                [r['event_luminosity_block'] for r in rows], type=pa.int32()),
            'source_batch_id': pa.array(
                [r['source_batch_id'] for r in rows], type=pa.int32()),
            'source_microbatch_id': pa.array(
                [r['source_microbatch_id'] for r in rows], type=pa.int32()),
            'event_primary_vertex_z': pa.array(
                [r['event_primary_vertex_z'] for r in rows], type=pa.float32()),
            'event_n_tracks': pa.array(
                [r['event_n_tracks'] for r in rows], type=pa.int32()),
        }
        for track_col in (
            'track_pt', 'track_eta', 'track_phi', 'track_dxy_significance',
            'track_dz_significance', 'track_dca_significance',
            'track_pt_error', 'track_norm_chi2',
            'track_covariance_phi_phi', 'track_covariance_lambda_lambda',
        ):
            columns[track_col] = pa.array(
                [r[track_col] for r in rows],
                type=pa.large_list(pa.float32()),
            )
        columns['track_n_valid_pixel_hits'] = pa.array(
            [r['track_n_valid_pixel_hits'] for r in rows],
            type=pa.large_list(pa.int32()),
        )
        columns['track_label_from_tau'] = pa.array(
            [r['track_label_from_tau'] for r in rows],
            type=pa.large_list(pa.int32()),
        )
        table = pa.table(columns)
        path = tmp_path / 'val_fixture.parquet'
        pq.write_table(table, path)
        return path

    def test_keyed_lookup_returns_correct_event(self, tmp_path):
        rows = [
            {
                'event_run': 1, 'event_id': 10,
                'event_luminosity_block': 1,
                'source_batch_id': 2, 'source_microbatch_id': 3,
                'event_primary_vertex_z': 0.5,
                'event_n_tracks': 3,
                'track_pt': [0.5, 1.0, 2.0],
                'track_eta': [0.0, 0.1, 0.2],
                'track_phi': [0.0, 0.0, 0.0],
                'track_dxy_significance': [0.1, 0.2, 0.3],
                'track_dz_significance': [0.2, 0.3, 0.4],
                'track_dca_significance': [0.0, 0.1, 0.2],
                'track_pt_error': [0.01, 0.02, 0.03],
                'track_norm_chi2': [1.0, 1.1, 1.2],
                'track_covariance_phi_phi': [1e-6, 2e-6, 3e-6],
                'track_covariance_lambda_lambda': [1e-6, 2e-6, 3e-6],
                'track_n_valid_pixel_hits': [4, 5, 6],
                'track_label_from_tau': [0, 1, 0],
            },
            {
                'event_run': 1, 'event_id': 20,
                'event_luminosity_block': 1,
                'source_batch_id': 2, 'source_microbatch_id': 4,
                'event_primary_vertex_z': 1.5,
                'event_n_tracks': 2,
                'track_pt': [0.3, 0.7],
                'track_eta': [0.5, 0.6],
                'track_phi': [0.0, 0.1],
                'track_dxy_significance': [0.0, 0.0],
                'track_dz_significance': [0.0, 0.0],
                'track_dca_significance': [0.0, 0.0],
                'track_pt_error': [0.01, 0.01],
                'track_norm_chi2': [1.0, 1.0],
                'track_covariance_phi_phi': [1e-6, 1e-6],
                'track_covariance_lambda_lambda': [1e-6, 1e-6],
                'track_n_valid_pixel_hits': [4, 4],
                'track_label_from_tau': [1, 0],
            },
        ]
        path = self._write_val_fixture(tmp_path, rows)

        loaded = load_val_track_features([str(path)])

        key_a = (1, 10, 1, 2, 3)
        key_b = (1, 20, 1, 2, 4)
        assert key_a in loaded
        assert key_b in loaded

        tracks_a = loaded[key_a]
        np.testing.assert_allclose(tracks_a['pt'], [0.5, 1.0, 2.0])
        np.testing.assert_allclose(tracks_a['eta'], [0.0, 0.1, 0.2])
        assert tracks_a['vertex_z'] == pytest.approx(0.5)
        np.testing.assert_array_equal(tracks_a['n_pixel_hits'], [4, 5, 6])
        np.testing.assert_array_equal(tracks_a['label_from_tau'], [0, 1, 0])

        tracks_b = loaded[key_b]
        assert tracks_b['pt'].shape == (2,)
        assert tracks_b['vertex_z'] == pytest.approx(1.5)
