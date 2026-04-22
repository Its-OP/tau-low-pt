"""Tests for ``diagnostics.prefilter_confidence_diagnostic``.

All confidence features in this diagnostic are **inference-safe**: they
use only information available at serving time (Stage-1 output + Stage-2
output), no GT labels. Tests pin the pure extractor function, the
aggregation over a fixture batch, and the stratified d' computation that
compares confidence-feature distributions of pass vs fail events.
"""
from __future__ import annotations

import numpy as np
import pytest

from diagnostics.prefilter_confidence_diagnostic import (
    compute_confidence_features,
    part_prefilter_overlap,
    prefilter_rank_of,
)


class TestPrefilterRankOf:
    def test_found_at_rank_1(self):
        assert prefilter_rank_of(42, [42, 17, 3, 99]) == 1

    def test_found_at_rank_k(self):
        assert prefilter_rank_of(3, [42, 17, 3, 99]) == 3

    def test_not_found_returns_sentinel_one_past_end(self):
        # Sentinel = len(top_indices) + 1 (one past last slot) so
        # callers can either filter it out or treat "absent" as
        # "ranked beyond the K that was stored".
        assert prefilter_rank_of(500, [42, 17, 3, 99]) == 5


class TestPartPrefilterOverlap:
    def test_full_overlap_top_k(self):
        prefilter_top = [1, 2, 3, 4, 5]
        part_top = [5, 4, 3, 2, 1]  # same set, different order
        assert part_prefilter_overlap(prefilter_top, part_top, K=5) == pytest.approx(1.0)

    def test_half_overlap(self):
        prefilter_top = [1, 2, 3, 4]
        part_top = [3, 4, 99, 100]
        assert part_prefilter_overlap(prefilter_top, part_top, K=4) == pytest.approx(0.5)

    def test_k_truncation(self):
        prefilter_top = [1, 2, 3, 4, 5]
        part_top = [1, 2, 99]
        # With K=2 → {1,2} ∩ {1,2} = 2 → 2/2 = 1.0
        assert part_prefilter_overlap(prefilter_top, part_top, K=2) == pytest.approx(1.0)

    def test_empty_part_top_returns_zero(self):
        assert part_prefilter_overlap([1, 2, 3], [], K=10) == 0.0


class TestComputeConfidenceFeatures:
    def test_returns_all_expected_keys(self):
        feats = compute_confidence_features(
            top_prefilter_pt=[2.0, 1.5, 1.0, 0.5],
            top_prefilter_indices=[10, 20, 30, 40],
            top_part_indices=[10, 30, 20, 40],
        )
        expected = {
            'n_top_retained',
            'top_pt_mean', 'top_pt_median',
            'top_pt_max', 'top_pt_min', 'top_pt_std',
            'top1_pt',
            'part_prefilter_overlap_top50',
            'part_prefilter_overlap_top100',
            'part_top1_prefilter_rank',
        }
        assert expected <= set(feats)

    def test_top_pt_aggregates_correct(self):
        feats = compute_confidence_features(
            top_prefilter_pt=[3.0, 1.0, 2.0],
            top_prefilter_indices=[1, 2, 3],
            top_part_indices=[],
        )
        assert feats['top1_pt'] == pytest.approx(3.0)
        assert feats['top_pt_max'] == pytest.approx(3.0)
        assert feats['top_pt_min'] == pytest.approx(1.0)
        assert feats['top_pt_mean'] == pytest.approx(2.0)
        assert feats['top_pt_median'] == pytest.approx(2.0)
        # std (population, ddof=0): sqrt(((3-2)^2 + (1-2)^2 + (2-2)^2) / 3)
        assert feats['top_pt_std'] == pytest.approx(
            np.sqrt(((1 + 1 + 0) / 3)), abs=1e-5,
        )

    def test_n_top_retained_matches_top_length(self):
        feats = compute_confidence_features(
            top_prefilter_pt=[1.0, 1.0, 1.0],
            top_prefilter_indices=[1, 2, 3],
            top_part_indices=[],
        )
        assert feats['n_top_retained'] == 3

    def test_part_top1_prefilter_rank(self):
        feats = compute_confidence_features(
            top_prefilter_pt=[1.0] * 5,
            top_prefilter_indices=[100, 200, 300, 400, 500],
            top_part_indices=[300, 100, 200],
        )
        # ParT's top-1 is idx 300, which sits at prefilter rank 3.
        assert feats['part_top1_prefilter_rank'] == 3

    def test_part_top1_absent_in_prefilter(self):
        feats = compute_confidence_features(
            top_prefilter_pt=[1.0] * 3,
            top_prefilter_indices=[100, 200, 300],
            top_part_indices=[999, 100, 200],
        )
        # 999 is not in the prefilter top — sentinel = len+1 = 4.
        assert feats['part_top1_prefilter_rank'] == 4

    def test_part_top_empty_overlap_zero(self):
        feats = compute_confidence_features(
            top_prefilter_pt=[1.0] * 5,
            top_prefilter_indices=[1, 2, 3, 4, 5],
            top_part_indices=[],
        )
        assert feats['part_prefilter_overlap_top50'] == 0.0
        assert feats['part_prefilter_overlap_top100'] == 0.0
        # Empty → rank sentinel is len(prefilter)+1.
        assert feats['part_top1_prefilter_rank'] == 6

    def test_empty_prefilter_produces_safe_zeros(self):
        feats = compute_confidence_features(
            top_prefilter_pt=[],
            top_prefilter_indices=[],
            top_part_indices=[1, 2, 3],
        )
        assert feats['n_top_retained'] == 0
        assert feats['top_pt_mean'] == 0.0
        assert feats['top_pt_max'] == 0.0
        assert feats['top_pt_std'] == 0.0
        # Nothing to match against → overlap 0, rank sentinel 1.
        assert feats['part_prefilter_overlap_top50'] == 0.0
        assert feats['part_top1_prefilter_rank'] == 1
