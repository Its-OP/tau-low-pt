"""Prefilter confidence diagnostic.

Tests whether **inference-safe** confidence signals (no GT labels) can
predict perfect-recall failure. Purpose: decide whether a two-track
operating point (per-event K1 routing) is viable using only features
available at serving time.

Inputs:
    * ``experiments/results.parquet`` — per-event prefilter top-256 +
      ParT top-100 (indices + pT; no scores recorded).
    * ``perfect_recall_per_event.csv`` — per-event pass/fail labels
      (``perfect_recall`` column) produced by
      ``prefilter_perfect_recall_diagnostic.py``.

Outputs:
    * ``prefilter_confidence_<date>.md`` — d' table for each
      confidence feature stratified by perfect-recall pass/fail,
      threshold sweeps for the strongest discriminators, and a
      combined-rule section if any feature clears |d'| ≥ 0.3.

What counts as inference-safe:
    * Statistics of ``top_256_prefilter_pt``: mean, median, max, min,
      std. These are pT values of the prefilter's top picks — not
      scores, but a proxy.
    * ``n_top_retained`` — length of the top-256 list (caps at min
      of 256 and event's track count).
    * Stage-1 / Stage-2 agreement: overlap between prefilter top-K and
      ParT top-K at K ∈ {50, 100}. Reshuffle rate is a post-hoc
      confidence proxy (requires Stage 2 to have run).
    * ``part_top1_prefilter_rank`` — where ParT's #1 sat in the
      prefilter ranking; big rank number = Stage-2 moved a deep track
      to the top, i.e. Stage-1 was wrong about it.

Usage::

    python -m diagnostics.prefilter_confidence_diagnostic \\
        --results-parquet experiments/results.parquet \\
        --per-event-csv part/reports/perfect_recall_per_event.csv \\
        --output-dir part/reports
"""
from __future__ import annotations

import argparse
import csv
import datetime
import logging
import math
import os
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from diagnostics.prefilter_perfect_recall_diagnostic import (  # noqa: E402
    compare_distributions,
)
from diagnostics.prefilter_stratified_eval import (  # noqa: E402
    find_best_threshold,
    quantile_thresholds,
    sweep_threshold,
)


logger = logging.getLogger('prefilter_confidence_diagnostic')
logging.basicConfig(
    level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
)


# ---------------------------------------------------------------------------
# Pure confidence-feature extractors
# ---------------------------------------------------------------------------


def prefilter_rank_of(track_index: int, top_indices: list[int]) -> int:
    """1-indexed position of ``track_index`` in ``top_indices``.

    Returns ``len(top_indices) + 1`` if absent — "one past last stored
    slot", so callers can threshold without branching on None.
    """
    try:
        return top_indices.index(int(track_index)) + 1
    except ValueError:
        return len(top_indices) + 1


def part_prefilter_overlap(
    prefilter_top: list[int],
    part_top: list[int],
    K: int,
) -> float:
    """Fraction of ``part_top[:K]`` present in ``prefilter_top[:K]``.

    Denominator is ``K`` (not ``len(part_top)``) so the metric is
    directly comparable across events with different ParT output
    lengths. Returns 0.0 for empty ParT output.
    """
    if K <= 0 or not part_top:
        return 0.0
    prefilter_set = set(int(i) for i in prefilter_top[:K])
    part_head = [int(i) for i in part_top[:K]]
    if not part_head:
        return 0.0
    hits = sum(1 for i in part_head if i in prefilter_set)
    return hits / K


def compute_confidence_features(
    top_prefilter_pt: list[float],
    top_prefilter_indices: list[int],
    top_part_indices: list[int],
) -> dict[str, float]:
    """Extract the inference-safe confidence feature set for one event.

    All features use only model outputs, no GT labels. Feature list:

        n_top_retained                  — len(top_prefilter)
        top1_pt                         — pT of the prefilter's #1 pick
        top_pt_mean / median / max / min / std
                                        — pT distribution of top-256
        part_prefilter_overlap_top50    — |ParT[:50] ∩ Prefilter[:50]|/50
        part_prefilter_overlap_top100   — same at K=100
        part_top1_prefilter_rank        — rank of ParT's #1 in the
                                          prefilter list (sentinel
                                          len(prefilter)+1 if absent)
    """
    top_pt = np.asarray(top_prefilter_pt, dtype=np.float64)
    n_retained = int(top_pt.size)

    if n_retained == 0:
        features = {
            'n_top_retained': 0.0,
            'top1_pt': 0.0,
            'top_pt_mean': 0.0,
            'top_pt_median': 0.0,
            'top_pt_max': 0.0,
            'top_pt_min': 0.0,
            'top_pt_std': 0.0,
        }
    else:
        features = {
            'n_top_retained': float(n_retained),
            'top1_pt': float(top_pt[0]),
            'top_pt_mean': float(np.mean(top_pt)),
            'top_pt_median': float(np.median(top_pt)),
            'top_pt_max': float(np.max(top_pt)),
            'top_pt_min': float(np.min(top_pt)),
            'top_pt_std': float(np.std(top_pt)),
        }

    features['part_prefilter_overlap_top50'] = part_prefilter_overlap(
        top_prefilter_indices, top_part_indices, K=50,
    )
    features['part_prefilter_overlap_top100'] = part_prefilter_overlap(
        top_prefilter_indices, top_part_indices, K=100,
    )

    if top_part_indices:
        features['part_top1_prefilter_rank'] = float(
            prefilter_rank_of(top_part_indices[0], top_prefilter_indices),
        )
    else:
        # No ParT output → sentinel one past the last slot.
        features['part_top1_prefilter_rank'] = float(n_retained + 1)

    return features


_CONFIDENCE_FEATURES = (
    'n_top_retained',
    'top1_pt',
    'top_pt_mean',
    'top_pt_median',
    'top_pt_max',
    'top_pt_min',
    'top_pt_std',
    'part_prefilter_overlap_top50',
    'part_prefilter_overlap_top100',
    'part_top1_prefilter_rank',
)


# ---------------------------------------------------------------------------
# IO + join layer
# ---------------------------------------------------------------------------


_KEY_COLUMNS = (
    'event_run', 'event_id', 'event_luminosity_block',
    'source_batch_id', 'source_microbatch_id',
)


def iter_results_confidence(results_parquet_path: str):
    """Yield per-event dicts from results.parquet containing the
    prefilter top-K pT, prefilter indices, and ParT indices."""
    table = pq.read_table(
        results_parquet_path,
        columns=[
            *_KEY_COLUMNS,
            'top_256_prefilter_indices',
            'top_256_prefilter_pt',
            'top_100_part_indices',
        ],
    )
    key_cols = {k: table.column(k).to_pylist() for k in _KEY_COLUMNS}
    pre_idx = table.column('top_256_prefilter_indices').to_pylist()
    pre_pt = table.column('top_256_prefilter_pt').to_pylist()
    part_idx = table.column('top_100_part_indices').to_pylist()
    for row in range(table.num_rows):
        yield {
            'key': tuple(int(key_cols[k][row]) for k in _KEY_COLUMNS),
            'top_256_prefilter_indices': [int(i) for i in pre_idx[row]],
            'top_256_prefilter_pt': [float(x) for x in pre_pt[row]],
            'top_100_part_indices': [int(i) for i in part_idx[row]],
        }


def load_perfect_recall_labels(per_event_csv: str) -> dict[tuple, int]:
    """Return ``{composite_key: perfect_recall}`` from the per-event CSV."""
    labels: dict[tuple, int] = {}
    with open(per_event_csv, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = tuple(int(row[k]) for k in _KEY_COLUMNS)
            labels[key] = int(row['perfect_recall'])
    return labels


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------


def _fmt(value, digits: int = 4) -> str:
    if value is None:
        return 'nan'
    if isinstance(value, float) and math.isnan(value):
        return 'nan'
    if isinstance(value, int) or (isinstance(value, float) and value.is_integer()):
        return f'{int(value):,}'
    if abs(value) >= 1000:
        return f'{value:,.0f}'
    return f'{value:.{digits}f}'


def _render_dprime_table(
    rows: list[dict],
    label_a: str,
    label_b: str,
) -> str:
    header = (
        f'| feature | mean ({label_a}) | mean ({label_b}) | '
        f'median ({label_a}) | median ({label_b}) | d\' | '
        f'n ({label_a}) | n ({label_b}) |\n'
    )
    sep = '|---|---:|---:|---:|---:|---:|---:|---:|\n'
    body = ''.join(
        f'| {r["feature"]} | {_fmt(r["mean_a"])} | {_fmt(r["mean_b"])} | '
        f'{_fmt(r["median_a"])} | {_fmt(r["median_b"])} | '
        f'{_fmt(r["d_prime"], 3)} | {r["n_a"]:,} | {r["n_b"]:,} |\n'
        for r in sorted(
            rows, key=lambda x: -abs(x['d_prime']) if not math.isnan(x['d_prime']) else 0,
        )
    )
    return header + sep + body


def _render_sweep_table(rows: list[dict]) -> str:
    header = (
        '| rule | n_hard | fraction_hard | fail_in_hard | '
        'fail_precision | fail_recall | P@K (hard) | P@K (easy) |\n'
    )
    sep = '|---|---:|---:|---:|---:|---:|---:|---:|\n'
    body = ''.join(
        f'| {r["rule"]} | {_fmt(r["n_hard"])} | {_fmt(r["fraction_hard"])} | '
        f'{_fmt(r["fail_in_hard"])} | {_fmt(r["fail_precision"])} | '
        f'{_fmt(r["fail_recall"])} | {_fmt(r["p_at_k_hard"])} | '
        f'{_fmt(r["p_at_k_easy"])} |\n'
        for r in rows
    )
    return header + sep + body


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Check whether inference-safe confidence features predict prefilter failure.',
    )
    parser.add_argument(
        '--results-parquet', type=str, required=True,
        help='Path to experiments/results.parquet.',
    )
    parser.add_argument(
        '--per-event-csv', type=str, required=True,
        help='Path to perfect_recall_per_event.csv (has the pass/fail labels).',
    )
    parser.add_argument(
        '--output-dir', type=str, required=True,
        help='Directory to write the markdown report.',
    )
    parser.add_argument(
        '--d-prime-threshold', type=float, default=0.3,
        help='Threshold sweep kicks in only for features with |d\'| above this value.',
    )
    parser.add_argument(
        '--beta', type=float, default=1.5,
        help='Fβ for picking the best threshold (β>1 favours recall).',
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)

    labels = load_perfect_recall_labels(args.per_event_csv)
    logger.info('Loaded %d perfect-recall labels.', len(labels))

    events: list[dict] = []
    unmatched = 0
    for row in iter_results_confidence(args.results_parquet):
        if row['key'] not in labels:
            unmatched += 1
            continue
        features = compute_confidence_features(
            top_prefilter_pt=row['top_256_prefilter_pt'],
            top_prefilter_indices=row['top_256_prefilter_indices'],
            top_part_indices=row['top_100_part_indices'],
        )
        features['perfect_recall'] = labels[row['key']]
        events.append(features)

    if unmatched:
        logger.warning(
            '%d events in results.parquet had no label in CSV (skipped).',
            unmatched,
        )

    n_total = len(events)
    n_fail = sum(1 for e in events if int(e['perfect_recall']) == 0)
    logger.info(
        'Joined %d events, %d fail (%.4f baseline).',
        n_total, n_fail, n_fail / n_total if n_total else 0.0,
    )

    # ---- d' per feature (pass vs fail) ----
    pass_feats = [e for e in events if int(e['perfect_recall']) == 1]
    fail_feats = [e for e in events if int(e['perfect_recall']) == 0]
    dprime_rows = compare_distributions(
        pass_feats, fail_feats,
        feature_names=list(_CONFIDENCE_FEATURES),
    )

    # ---- Threshold sweep on the strongest discriminator ----
    strong_rows = [
        r for r in dprime_rows
        if not math.isnan(r['d_prime'])
        and abs(r['d_prime']) >= args.d_prime_threshold
    ]
    strong_rows_sorted = sorted(strong_rows, key=lambda r: -abs(r['d_prime']))

    sweep_sections: list[tuple[str, list[dict], dict]] = []
    for row in strong_rows_sorted[:3]:
        feature = row['feature']
        # Direction — sign of (mean_b - mean_a) tells which side is fail.
        # If fail has HIGHER mean, 'hard' events should be above a
        # threshold → direction 'gt'. Otherwise 'lt'.
        direction = 'gt' if row['mean_b'] > row['mean_a'] else 'lt'
        quantiles = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
        if direction == 'gt':
            quantiles = [1.0 - q for q in quantiles]
        thresholds = quantile_thresholds(events, feature, quantiles)
        sweep_rows = sweep_threshold(
            events, feature, thresholds, direction=direction,
        )
        best = find_best_threshold(sweep_rows, beta=args.beta)
        sweep_sections.append((feature, sweep_rows, best))

    # ---- Write markdown ----
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    today = datetime.date.today().isoformat().replace('-', '')
    report_path = out_dir / f'prefilter_confidence_{today}.md'

    lines: list[str] = []
    lines.append(
        f'# Prefilter confidence diagnostic — {datetime.date.today().isoformat()}\n\n',
    )
    lines.append(
        f'Source: `{args.results_parquet}` joined with '
        f'`{args.per_event_csv}` (pass/fail labels). '
        f'{n_total:,} events, {n_fail:,} fails '
        f'(baseline {n_fail / n_total if n_total else 0:.4f}).\n\n'
        'All features are **inference-safe** — no GT labels used.\n\n',
    )

    lines.append('## 1. d\' per confidence feature (pass vs fail)\n\n')
    lines.append(_render_dprime_table(dprime_rows, 'pass', 'fail'))
    lines.append('\n')

    if not strong_rows:
        lines.append(
            f'## 2. No feature cleared |d\'| ≥ {args.d_prime_threshold}.\n\n'
            'Inference-time routing via any single confidence feature '
            'will not give a useful signal above seed variance. '
            'Combined rules at these effect sizes would compound noise '
            'rather than signal.\n',
        )
    else:
        lines.append(
            f'## 2. Threshold sweeps for features with |d\'| ≥ '
            f'{args.d_prime_threshold}\n\n',
        )
        for feature, sweep_rows, best in sweep_sections:
            lines.append(f'### 2.{feature}\n\n')
            lines.append(_render_sweep_table(sweep_rows))
            lines.append(
                f'\nBest Fβ (β={args.beta}) threshold: '
                f'`{best["rule"]}` (Fβ = {best["f_beta"]:.4f}).\n\n',
            )

    lines.append('## 3. Method\n\n')
    lines.append(
        'Features are computed directly from `results.parquet` '
        '(prefilter top-256 indices + pT, ParT top-100 indices — no '
        'scores are stored in the current eval artefact). Pass/fail '
        'labels join by composite event key from the per-event CSV. '
        'd\' = |μ_pass − μ_fail| / √((σ_pass² + σ_fail²)/2). Threshold '
        'sweeps match the stratified-eval quantile scheme so they are '
        'directly comparable to earlier tables.\n',
    )

    report_path.write_text(''.join(lines))
    logger.info('Wrote report to %s', report_path)


if __name__ == '__main__':
    main()
