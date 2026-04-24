"""Tests for ``diagnostics.prefilter_stratified_eval``.

Covers rule evaluation (precision/recall of failure capture), threshold
sweeping over a single feature, and combined multi-feature rules.
Fixtures use small in-memory lists shaped like the per-event CSV rows
written by the perfect-recall diagnostic.
"""
from __future__ import annotations

import pytest

from diagnostics.prefilter_stratified_eval import (
    combined_rule_predicate,
    evaluate_stratification,
    find_best_threshold,
    sweep_threshold,
)


# Tiny synthetic dataset: 10 events, 2 features.
# perfect_recall: 1 = pass, 0 = fail.
# gt_pt_sum: low values correlate with failure.
# n_tracks: high values correlate with failure.
_EVENTS = [
    {'gt_pt_sum': 1.5, 'n_tracks': 900, 'perfect_recall': 0},
    {'gt_pt_sum': 2.0, 'n_tracks': 850, 'perfect_recall': 0},
    {'gt_pt_sum': 2.5, 'n_tracks': 820, 'perfect_recall': 0},
    {'gt_pt_sum': 3.0, 'n_tracks': 800, 'perfect_recall': 1},
    {'gt_pt_sum': 3.5, 'n_tracks': 780, 'perfect_recall': 0},
    {'gt_pt_sum': 4.0, 'n_tracks': 700, 'perfect_recall': 1},
    {'gt_pt_sum': 4.5, 'n_tracks': 650, 'perfect_recall': 1},
    {'gt_pt_sum': 5.0, 'n_tracks': 600, 'perfect_recall': 1},
    {'gt_pt_sum': 5.5, 'n_tracks': 550, 'perfect_recall': 1},
    {'gt_pt_sum': 6.0, 'n_tracks': 500, 'perfect_recall': 1},
]


class TestEvaluateStratification:
    def test_rule_captures_all_failures_at_broad_threshold(self):
        # gt_pt_sum < 4.0 → subset = {1.5, 2.0, 2.5, 3.0, 3.5}
        # failures in subset: 4 (1.5, 2.0, 2.5, 3.5); 3.0 passes.
        # total failures: 4.
        rule = lambda e: e['gt_pt_sum'] < 4.0
        result = evaluate_stratification(_EVENTS, 'gt_pt_sum<4.0', rule)

        assert result['n_hard'] == 5
        assert result['n_easy'] == 5
        assert result['fail_in_hard'] == 4
        assert result['fail_in_easy'] == 0
        assert result['n_fail_total'] == 4
        assert result['fraction_hard'] == pytest.approx(0.5)
        # recall = 4/4 = 1.0
        assert result['fail_recall'] == pytest.approx(1.0)
        # precision = 4/5 = 0.8
        assert result['fail_precision'] == pytest.approx(0.8)
        # p_at_k on easy subset = 5/5 = 1.0 (all pass)
        assert result['p_at_k_easy'] == pytest.approx(1.0)
        # p_at_k on hard subset = 1/5 = 0.2
        assert result['p_at_k_hard'] == pytest.approx(0.2)

    def test_rule_captures_no_events(self):
        rule = lambda e: e['gt_pt_sum'] < 0.0
        result = evaluate_stratification(_EVENTS, 'impossible', rule)
        assert result['n_hard'] == 0
        assert result['fail_in_hard'] == 0
        assert result['fail_recall'] == 0.0
        # Precision is 0 (no subset → no true positives).
        assert result['fail_precision'] == 0.0
        # Empty subset → p_at_k is undefined; implementation choice:
        # treat as nan or as 0. Pick nan so plots don't mislead.
        assert result['p_at_k_hard'] != result['p_at_k_hard']  # NaN check

    def test_rule_captures_all_events(self):
        rule = lambda e: True
        result = evaluate_stratification(_EVENTS, 'catch_all', rule)
        assert result['n_hard'] == 10
        assert result['n_easy'] == 0
        assert result['fail_recall'] == pytest.approx(1.0)
        assert result['fail_precision'] == pytest.approx(0.4)  # 4/10


class TestSweepThreshold:
    def test_single_feature_sweep_returns_row_per_threshold(self):
        thresholds = [2.0, 3.0, 4.0, 5.0]
        rows = sweep_threshold(
            _EVENTS, feature='gt_pt_sum',
            thresholds=thresholds, direction='lt',
        )
        assert len(rows) == len(thresholds)
        # Spot-check the 4.0 row matches TestEvaluateStratification.
        match = next(r for r in rows if r['threshold'] == 4.0)
        assert match['fail_recall'] == pytest.approx(1.0)
        assert match['fail_precision'] == pytest.approx(0.8)

    def test_direction_gt_flips_semantics(self):
        rows = sweep_threshold(
            _EVENTS, feature='n_tracks',
            thresholds=[700, 800], direction='gt',
        )
        # n_tracks > 800 → {900, 850, 820}; failures = 3. total = 4.
        match = next(r for r in rows if r['threshold'] == 800)
        assert match['n_hard'] == 3
        assert match['fail_in_hard'] == 3
        assert match['fail_recall'] == pytest.approx(3 / 4)

    def test_rule_name_format(self):
        rows = sweep_threshold(
            _EVENTS, feature='gt_pt_sum',
            thresholds=[3.0], direction='lt',
        )
        assert rows[0]['rule'] == 'gt_pt_sum<3.0'


class TestFindBestThreshold:
    def test_maximizes_f_beta_score(self):
        # Beta=1 (F1): threshold that balances precision + recall.
        rows = [
            {'rule': 't=2', 'fail_recall': 0.5, 'fail_precision': 1.0},
            {'rule': 't=4', 'fail_recall': 1.0, 'fail_precision': 0.8},
            {'rule': 't=6', 'fail_recall': 1.0, 'fail_precision': 0.4},
        ]
        best = find_best_threshold(rows, beta=1.0)
        # F1(t=4) = 2 * (1 * 0.8) / (1 + 0.8) = 0.888...
        # F1(t=2) = 2 * (1 * 0.5) / (1 + 0.5) = 0.666...
        # F1(t=6) = 2 * (1 * 0.4) / (1 + 0.4) = 0.571...
        assert best['rule'] == 't=4'

    def test_beta_greater_than_one_favors_recall(self):
        rows = [
            {'rule': 'narrow', 'fail_recall': 0.6, 'fail_precision': 0.9},
            {'rule': 'wide', 'fail_recall': 1.0, 'fail_precision': 0.5},
        ]
        best = find_best_threshold(rows, beta=2.0)
        assert best['rule'] == 'wide'


class TestCombinedRulePredicate:
    def test_and_combines_predicates(self):
        p_low_pt = lambda e: e['gt_pt_sum'] < 4.0
        p_high_n = lambda e: e['n_tracks'] > 780
        combined = combined_rule_predicate([p_low_pt, p_high_n], mode='AND')
        # Events matching both: gt_pt_sum < 4.0 AND n_tracks > 780
        # → {1.5/900, 2.0/850, 2.5/820, 3.0/800}. Four matches.
        matches = [e for e in _EVENTS if combined(e)]
        assert len(matches) == 4

    def test_or_combines_predicates(self):
        p_low_pt = lambda e: e['gt_pt_sum'] < 2.5
        p_high_n = lambda e: e['n_tracks'] > 850
        combined = combined_rule_predicate([p_low_pt, p_high_n], mode='OR')
        # {1.5 (both), 2.0 (both)} + {nothing beyond that alone at these
        # thresholds}. So {1.5/900, 2.0/850} + {900 already covered,
        # 850 already covered}. → 2.
        matches = [e for e in _EVENTS if combined(e)]
        assert len(matches) == 2

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError):
            combined_rule_predicate([lambda e: True], mode='XOR')
