"""Prefilter perfect-recall diagnostic.

Answers: for the 12.1 % of val events that lose at least one GT pion in
Stage 1's top-K, which events are they and what makes the missed pion
different from its two recalled siblings?

Inputs (both local, no inference):
    * ``experiments/results.parquet`` — cascade eval output with
      ``top_256_prefilter_indices`` and ``gt_pion_indices`` per event.
    * ``part/data/low-pt/val/val_00[0-6].parquet`` — source val parquet
      with the full per-track feature arrays (pt, eta, phi, dxy_sig,
      dz_sig, ...). Joins on the composite event key.

Outputs:
    * ``prefilter_perfect_recall_<date>.md`` — human-readable summary
      with the four diagnostic tables (D1–D4).
    * ``perfect_recall_per_event.csv`` and
      ``perfect_recall_per_pion.csv`` — optional long-form artefacts
      enabled via ``--write-csv``.

Usage::

    python -m diagnostics.prefilter_perfect_recall_diagnostic \\
        --results-parquet experiments/results.parquet \\
        --val-dir part/data/low-pt/val \\
        --output-dir part/reports \\
        --K 256 --knn-k 16 --write-csv
"""
from __future__ import annotations

import argparse
import csv
import datetime
import glob
import logging
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


logger = logging.getLogger('prefilter_perfect_recall_diagnostic')
logging.basicConfig(
    level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
)


# ---------------------------------------------------------------------------
# P@K primitives (pure)
# ---------------------------------------------------------------------------


def event_perfect_recall(
    gt_indices: list[int],
    top_k_indices: list[int],
    K: int = 256,
) -> bool:
    """True iff every GT index is present in ``top_k_indices[:K]``.

    Empty-GT events are vacuously True (no target to miss).
    """
    gt_set = set(int(i) for i in gt_indices)
    if not gt_set:
        return True
    top_set = set(int(i) for i in top_k_indices[:K])
    return gt_set.issubset(top_set)


def count_gt_in_top_k(
    gt_indices: list[int],
    top_k_indices: list[int],
    K: int = 256,
) -> int:
    """Number of GT indices present in ``top_k_indices[:K]``."""
    gt_set = set(int(i) for i in gt_indices)
    top_set = set(int(i) for i in top_k_indices[:K])
    return len(gt_set & top_set)


def gt_rank_in_top_k(
    gt_index: int,
    top_k_indices: list[int],
    K: int | None = None,
) -> int | None:
    """1-indexed rank of ``gt_index`` in ``top_k_indices``.

    Returns ``None`` if ``gt_index`` is absent or its rank exceeds ``K``
    (when provided). Ranks beyond a supplied ``K`` cap are treated as
    absent because the diagnostic's notion of "found" is "found inside
    the cut-off".
    """
    try:
        zero_indexed = top_k_indices.index(int(gt_index))
    except ValueError:
        return None
    rank = zero_indexed + 1
    if K is not None and rank > K:
        return None
    return rank


# ---------------------------------------------------------------------------
# Feature extraction (pure — operates on already-joined numpy arrays)
# ---------------------------------------------------------------------------


_EVENT_FEATURES = (
    'n_tracks',
    'event_pt_median',
    'event_pt_max',
    'event_pt_std',
    'event_pt_p95',
    'mean_abs_dz_sig',
    'mean_abs_dxy_sig',
    'mean_chi2',
    'gt_pt_min',
    'gt_pt_mean',
    'gt_pt_max',
    'gt_pt_sum',
    'gt_pt_spread',
    'vertex_z',
)

_PION_FEATURES = (
    'pt',
    'pt_rank_in_event',
    'dxy_significance',
    'dz_significance',
    'dca_significance',
    'pt_error',
    'norm_chi2',
    'n_pixel_hits',
    'covariance_phi_phi',
    'covariance_lambda_lambda',
)


def per_event_features(
    track_arrays: dict[str, np.ndarray],
    gt_indices: list[int],
    vertex_z: float,
) -> dict[str, float]:
    """Aggregate per-event features used by D1.

    Stats span the whole event's track set (``event_*`` / ``mean_*``
    fields) plus GT-pT aggregates (``gt_pt_*``). Robust to n_tracks = 0
    (returns zeros).
    """
    pt = np.asarray(track_arrays['pt'], dtype=np.float64)
    dz_sig = np.asarray(track_arrays['dz_significance'], dtype=np.float64)
    dxy_sig = np.asarray(track_arrays['dxy_significance'], dtype=np.float64)
    chi2 = np.asarray(track_arrays['norm_chi2'], dtype=np.float64)
    n_tracks = int(pt.size)

    if n_tracks == 0:
        return {name: 0.0 for name in _EVENT_FEATURES}

    feats: dict[str, float] = {
        'n_tracks': float(n_tracks),
        'event_pt_median': float(np.median(pt)),
        'event_pt_max': float(np.max(pt)),
        'event_pt_std': float(np.std(pt)),
        'event_pt_p95': float(np.percentile(pt, 95.0)),
        'mean_abs_dz_sig': float(np.mean(np.abs(dz_sig))),
        'mean_abs_dxy_sig': float(np.mean(np.abs(dxy_sig))),
        'mean_chi2': float(np.mean(chi2)),
        'vertex_z': float(vertex_z),
    }

    gt_list = [int(i) for i in gt_indices if 0 <= int(i) < n_tracks]
    if gt_list:
        gt_pts = pt[gt_list]
        feats['gt_pt_min'] = float(np.min(gt_pts))
        feats['gt_pt_mean'] = float(np.mean(gt_pts))
        feats['gt_pt_max'] = float(np.max(gt_pts))
        feats['gt_pt_sum'] = float(np.sum(gt_pts))
        feats['gt_pt_spread'] = float(np.max(gt_pts) - np.min(gt_pts))
    else:
        for key in (
            'gt_pt_min', 'gt_pt_mean', 'gt_pt_max',
            'gt_pt_sum', 'gt_pt_spread',
        ):
            feats[key] = 0.0
    return feats


def per_pion_features(
    pion_index: int,
    track_arrays: dict[str, np.ndarray],
) -> dict[str, float]:
    """Per-track features at ``pion_index`` plus its pT rank in event.

    Rank semantics: rank 1 is the hardest pT in the event; rank
    ``n_tracks`` is the softest. Ties broken by index order (argsort
    default).
    """
    pt = np.asarray(track_arrays['pt'], dtype=np.float64)
    idx = int(pion_index)
    order_desc = np.argsort(-pt, kind='stable')
    # rank = 1 + position of `idx` in descending-pT order
    rank_position = int(np.where(order_desc == idx)[0][0])
    rank = rank_position + 1

    return {
        'pt': float(pt[idx]),
        'pt_rank_in_event': float(rank),
        'dxy_significance': float(track_arrays['dxy_significance'][idx]),
        'dz_significance': float(track_arrays['dz_significance'][idx]),
        'dca_significance': float(track_arrays['dca_significance'][idx]),
        'pt_error': float(track_arrays['pt_error'][idx]),
        'norm_chi2': float(track_arrays['norm_chi2'][idx]),
        'n_pixel_hits': float(track_arrays['n_pixel_hits'][idx]),
        'covariance_phi_phi': float(track_arrays['covariance_phi_phi'][idx]),
        'covariance_lambda_lambda': float(
            track_arrays['covariance_lambda_lambda'][idx]
        ),
    }


def count_gt_neighbors_eta_phi(
    pion_index: int,
    gt_indices: list[int],
    track_eta: np.ndarray,
    track_phi: np.ndarray,
    k: int = 16,
) -> int:
    """Number of other GT tracks among the k-nearest (η, φ) neighbours
    of ``pion_index``.

    Distance: standard ΔR² proxy with φ wraparound —

        Δη = η_j − η_i
        Δφ = ((φ_j − φ_i + π) mod 2π) − π              ∈ (−π, π]
        d²(i, j) = Δη² + Δφ²

    ``pion_index`` itself is excluded from the neighbour pool so it
    never contributes to the count.
    """
    eta = np.asarray(track_eta, dtype=np.float64)
    phi = np.asarray(track_phi, dtype=np.float64)
    n = int(eta.size)
    idx = int(pion_index)
    gt_set = set(int(i) for i in gt_indices)

    if n <= 1 or not gt_set:
        return 0

    other_mask = np.arange(n) != idx
    other_indices = np.arange(n)[other_mask]
    deta = eta[other_mask] - eta[idx]
    dphi = phi[other_mask] - phi[idx]
    # Wrap Δφ into (−π, π].
    dphi = (dphi + np.pi) % (2 * np.pi) - np.pi
    distances_sq = deta ** 2 + dphi ** 2

    k_effective = min(k, other_indices.size)
    order = np.argsort(distances_sq, kind='stable')[:k_effective]
    neighbors = other_indices[order]

    # Don't double-count the pion itself (already excluded) — only
    # count the *other* GT pions among the neighbours.
    return int(sum(1 for n_idx in neighbors if int(n_idx) in gt_set and int(n_idx) != idx))


# ---------------------------------------------------------------------------
# Aggregation (pure)
# ---------------------------------------------------------------------------


def stratify_by_p_at_k(
    events: list[dict],
    K: int = 256,
) -> dict[str, list[dict]]:
    """Partition events into ``'pass'`` and ``'fail'`` by P@K.

    ``events`` must be dicts with ``gt_indices`` and ``top_k_indices``.
    """
    groups: dict[str, list[dict]] = {'pass': [], 'fail': []}
    for ev in events:
        passed = event_perfect_recall(
            ev['gt_indices'], ev['top_k_indices'], K=K,
        )
        groups['pass' if passed else 'fail'].append(ev)
    return groups


def compare_distributions(
    feats_a: list[dict],
    feats_b: list[dict],
    feature_names: list[str],
) -> list[dict]:
    """Per-feature Gaussian-discriminability summary.

    d-prime:
        d' = |μ_a − μ_b| / sqrt((σ_a² + σ_b²) / 2)

    Returns one dict per feature with the means, medians, stds, sample
    counts, and d'.
    """
    rows: list[dict] = []
    for name in feature_names:
        a_values = np.array(
            [float(f[name]) for f in feats_a if name in f],
            dtype=np.float64,
        )
        b_values = np.array(
            [float(f[name]) for f in feats_b if name in f],
            dtype=np.float64,
        )
        if a_values.size == 0 or b_values.size == 0:
            rows.append({
                'feature': name,
                'mean_a': float('nan'), 'mean_b': float('nan'),
                'median_a': float('nan'), 'median_b': float('nan'),
                'std_a': float('nan'), 'std_b': float('nan'),
                'd_prime': float('nan'),
                'n_a': int(a_values.size), 'n_b': int(b_values.size),
            })
            continue

        mean_a, mean_b = float(np.mean(a_values)), float(np.mean(b_values))
        std_a, std_b = float(np.std(a_values)), float(np.std(b_values))
        pooled = np.sqrt((std_a ** 2 + std_b ** 2) / 2.0)
        d_prime = abs(mean_a - mean_b) / pooled if pooled > 1e-12 else 0.0

        rows.append({
            'feature': name,
            'mean_a': mean_a, 'mean_b': mean_b,
            'median_a': float(np.median(a_values)),
            'median_b': float(np.median(b_values)),
            'std_a': std_a, 'std_b': std_b,
            'd_prime': float(d_prime),
            'n_a': int(a_values.size), 'n_b': int(b_values.size),
        })
    return rows


def miss_cooccurrence_histogram(
    events: list[dict],
    K: int = 256,
) -> dict[int, int]:
    """Histogram of ``count_gt_in_top_k`` over ``events``.

    Returns a dense dict covering {0, 1, 2, 3} keys so downstream
    consumers can iterate without a ``.get`` fallback.
    """
    counts = Counter(
        count_gt_in_top_k(ev['gt_indices'], ev['top_k_indices'], K=K)
        for ev in events
    )
    out = {0: 0, 1: 0, 2: 0, 3: 0}
    out.update(counts)
    return out


# ---------------------------------------------------------------------------
# IO — val-parquet composite-key loader
# ---------------------------------------------------------------------------


_VAL_TRACK_COLUMNS = (
    ('track_pt', 'pt'),
    ('track_eta', 'eta'),
    ('track_phi', 'phi'),
    ('track_dxy_significance', 'dxy_significance'),
    ('track_dz_significance', 'dz_significance'),
    ('track_dca_significance', 'dca_significance'),
    ('track_pt_error', 'pt_error'),
    ('track_norm_chi2', 'norm_chi2'),
    ('track_covariance_phi_phi', 'covariance_phi_phi'),
    ('track_covariance_lambda_lambda', 'covariance_lambda_lambda'),
    ('track_n_valid_pixel_hits', 'n_pixel_hits'),
    ('track_label_from_tau', 'label_from_tau'),
)

_VAL_KEY_COLUMNS = (
    'event_run', 'event_id', 'event_luminosity_block',
    'source_batch_id', 'source_microbatch_id',
)


def load_val_track_features(
    val_parquet_paths: list[str],
) -> dict[tuple, dict[str, np.ndarray]]:
    """Load per-track arrays from val parquet files, keyed by the
    composite event key.

    Each value is a dict of numpy arrays covering pt, eta, phi, dxy_sig,
    dz_sig, dca_sig, pt_error, norm_chi2, cov_phi_phi,
    cov_lambda_lambda, n_pixel_hits, label_from_tau, and the scalar
    ``vertex_z``.
    """
    keyed: dict[tuple, dict[str, np.ndarray]] = {}

    for path in val_parquet_paths:
        columns_to_read = [
            *_VAL_KEY_COLUMNS,
            'event_primary_vertex_z',
            *(src for src, _ in _VAL_TRACK_COLUMNS),
        ]
        table = pq.read_table(path, columns=columns_to_read)
        key_cols = {k: table.column(k).to_pylist() for k in _VAL_KEY_COLUMNS}
        vertex_z_col = table.column('event_primary_vertex_z').to_pylist()
        track_cols = {
            source: table.column(source).to_pylist()
            for source, _ in _VAL_TRACK_COLUMNS
        }

        num_rows = table.num_rows
        for row in range(num_rows):
            key = tuple(int(key_cols[k][row]) for k in _VAL_KEY_COLUMNS)
            entry: dict[str, np.ndarray] = {'vertex_z': float(vertex_z_col[row])}
            for source, dest in _VAL_TRACK_COLUMNS:
                raw = track_cols[source][row]
                if dest == 'n_pixel_hits' or dest == 'label_from_tau':
                    entry[dest] = np.asarray(raw, dtype=np.int32)
                else:
                    entry[dest] = np.asarray(raw, dtype=np.float32)
            keyed[key] = entry

        logger.info(
            'Loaded %d events from %s; cumulative %d',
            num_rows, os.path.basename(path), len(keyed),
        )
    return keyed


def iter_results_events(results_parquet_path: str):
    """Yield per-event dicts straight out of ``results.parquet``."""
    table = pq.read_table(results_parquet_path)
    key_cols = {k: table.column(k).to_pylist() for k in _VAL_KEY_COLUMNS}
    gt_col = table.column('gt_pion_indices').to_pylist()
    top_col = table.column('top_256_prefilter_indices').to_pylist()
    for row in range(table.num_rows):
        yield {
            'key': tuple(int(key_cols[k][row]) for k in _VAL_KEY_COLUMNS),
            'gt_indices': [int(i) for i in gt_col[row]],
            'top_k_indices': [int(i) for i in top_col[row]],
        }


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------


def _fmt(value: float, digits: int = 4) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return 'nan'
    if abs(value) >= 1_000:
        return f'{value:,.0f}'
    return f'{value:.{digits}f}'


def _render_comparison_table(
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
            rows, key=lambda x: -abs(x['d_prime']) if not np.isnan(x['d_prime']) else 0,
        )
    )
    return header + sep + body


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Prefilter perfect-recall diagnostic on local parquets.',
    )
    parser.add_argument(
        '--results-parquet', type=str, required=True,
        help='Path to experiments/results.parquet (cascade eval output).',
    )
    parser.add_argument(
        '--val-dir', type=str, required=True,
        help='Directory containing val_00*.parquet source files.',
    )
    parser.add_argument(
        '--output-dir', type=str, required=True,
        help='Directory where the markdown report (and CSVs) will be written.',
    )
    parser.add_argument(
        '--K', type=int, default=256,
        help='Top-K cutoff to evaluate (default: 256 — matches training default).',
    )
    parser.add_argument(
        '--knn-k', type=int, default=16,
        help='k for (η, φ) kNN GT-neighbour counting in D4 (default: 16).',
    )
    parser.add_argument(
        '--k-sweep', type=str, default='128,200,256,300,400,500',
        help='Comma-separated K values for the overall P@K / R@K table.',
    )
    parser.add_argument(
        '--write-csv', action='store_true',
        help='Also write per-event and per-pion CSVs next to the markdown.',
    )
    return parser


def _format_k_sweep(k_values: list[int], events: list[dict]) -> str:
    header = '| K | P@K | R@K (per-pion) | # events pass | # events fail |\n'
    sep = '|---:|---:|---:|---:|---:|\n'
    rows = []
    total = len(events)
    for k in k_values:
        pass_count = 0
        total_gt = 0
        recalled_gt = 0
        for ev in events:
            gt = ev['gt_indices']
            top = ev['top_k_indices']
            if event_perfect_recall(gt, top, K=k):
                pass_count += 1
            total_gt += len(gt)
            recalled_gt += count_gt_in_top_k(gt, top, K=k)
        p_at_k = pass_count / total if total else 0.0
        r_at_k = recalled_gt / total_gt if total_gt else 0.0
        rows.append(
            f'| {k} | {p_at_k:.4f} | {r_at_k:.4f} | '
            f'{pass_count:,} | {total - pass_count:,} |\n'
        )
    return header + sep + ''.join(rows)


def _format_cooccurrence(hist: dict[int, int], total: int) -> str:
    lines = ['| n_gt_in_top_K | # events | fraction |\n', '|---:|---:|---:|\n']
    for bucket in (0, 1, 2, 3):
        count = hist.get(bucket, 0)
        frac = count / total if total else 0.0
        lines.append(f'| {bucket} | {count:,} | {frac:.4f} |\n')
    return ''.join(lines)


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load val track features ----
    val_files = sorted(glob.glob(os.path.join(args.val_dir, 'val_*.parquet')))
    if not val_files:
        raise FileNotFoundError(
            f'No val parquet files found in {args.val_dir!r}',
        )
    logger.info('Loading val features from %d files...', len(val_files))
    track_by_key = load_val_track_features(val_files)

    # ---- First pass: build event records ----
    logger.info('Reading results.parquet events...')
    events: list[dict] = []
    unmatched = 0
    label_mismatches = 0
    for row in iter_results_events(args.results_parquet):
        entry = track_by_key.get(row['key'])
        if entry is None:
            unmatched += 1
            continue
        # Defensive sanity: the GT indices from results.parquet should all
        # point at positions where track_label_from_tau == 1.
        labels = entry['label_from_tau']
        for gt_idx in row['gt_indices']:
            if 0 <= gt_idx < labels.size and int(labels[gt_idx]) != 1:
                label_mismatches += 1
        events.append({
            'key': row['key'],
            'gt_indices': row['gt_indices'],
            'top_k_indices': row['top_k_indices'],
            'tracks': entry,
        })

    if unmatched:
        raise RuntimeError(
            f'{unmatched} events in results.parquet did not resolve to a '
            f'val entry by composite key; aborting.',
        )
    if label_mismatches:
        logger.warning(
            'label_from_tau mismatched %d times (GT indices point at '
            'non-tau tracks); check join semantics.', label_mismatches,
        )

    logger.info('Joined %d events', len(events))

    # ---- Compute per-event features (D1) ----
    logger.info('Computing per-event features...')
    for ev in events:
        ev['event_features'] = per_event_features(
            ev['tracks'], ev['gt_indices'], ev['tracks']['vertex_z'],
        )

    # ---- Compute per-pion features + D4 kNN (D2, D4) ----
    logger.info('Computing per-pion features + kNN GT-neighbour counts...')
    per_pion_records: list[dict] = []
    for ev in events:
        tracks = ev['tracks']
        top_set = set(ev['top_k_indices'][:args.K])
        gt_indices_list = ev['gt_indices']
        for gt_idx in gt_indices_list:
            n_gt_neighbors = count_gt_neighbors_eta_phi(
                pion_index=gt_idx,
                gt_indices=gt_indices_list,
                track_eta=tracks['eta'],
                track_phi=tracks['phi'],
                k=args.knn_k,
            )
            pion_feats = per_pion_features(gt_idx, tracks)
            pion_feats['gt_n_gt_neighbors'] = float(n_gt_neighbors)
            per_pion_records.append({
                'key': ev['key'],
                'pion_index': int(gt_idx),
                'recalled': int(gt_idx) in top_set,
                'event_passed': event_perfect_recall(
                    gt_indices_list, ev['top_k_indices'], K=args.K,
                ),
                **pion_feats,
            })

    # ---- Partition (D1) ----
    groups = stratify_by_p_at_k(events, K=args.K)
    n_total = len(events)
    n_pass = len(groups['pass'])
    n_fail = len(groups['fail'])

    event_feats_pass = [ev['event_features'] for ev in groups['pass']]
    event_feats_fail = [ev['event_features'] for ev in groups['fail']]
    d1_rows = compare_distributions(
        event_feats_pass, event_feats_fail,
        feature_names=list(_EVENT_FEATURES),
    )

    # ---- Missed vs recalled within failure events (D2) ----
    fail_keys = {ev['key'] for ev in groups['fail']}
    missed_feats = [
        r for r in per_pion_records
        if r['key'] in fail_keys and not r['recalled']
    ]
    recalled_feats = [
        r for r in per_pion_records
        if r['key'] in fail_keys and r['recalled']
    ]
    d2_rows = compare_distributions(
        missed_feats, recalled_feats,
        feature_names=list(_PION_FEATURES) + ['gt_n_gt_neighbors'],
    )

    # ---- Co-occurrence (D3) ----
    d3_hist = miss_cooccurrence_histogram(events, K=args.K)

    # ---- D4: missed vs recalled kNN GT-neighbour counts ----
    d4_rows = compare_distributions(
        missed_feats, recalled_feats,
        feature_names=['gt_n_gt_neighbors'],
    )

    # ---- K sweep (overall P@K / R@K) ----
    k_sweep_values = [int(k) for k in args.k_sweep.split(',') if k.strip()]
    p_r_table = _format_k_sweep(k_sweep_values, events)

    # ---- Render markdown ----
    today = datetime.date.today().isoformat().replace('-', '')
    report_path = out_dir / f'prefilter_perfect_recall_{today}.md'

    lines: list[str] = []
    lines.append(
        f'# Prefilter perfect-recall diagnostic — {datetime.date.today().isoformat()}\n\n',
    )
    lines.append(
        f'Input: `{args.results_parquet}` (joined with `{args.val_dir}/val_00*.parquet` on composite key).\n'
        f'K = {args.K}, kNN k = {args.knn_k}, events = {n_total:,} (pass = {n_pass:,}, fail = {n_fail:,}).\n\n',
    )

    lines.append('## 1. Overall P@K and R@K sweep\n\n')
    lines.append(p_r_table)
    lines.append('\n')

    lines.append('## 2. Co-occurrence histogram (D3)\n\n')
    lines.append(
        f'How many of each event\'s 3 GT pions land in top-{args.K}. '
        f'Sum = {n_total:,}.\n\n',
    )
    lines.append(_format_cooccurrence(d3_hist, n_total))
    lines.append('\n')

    lines.append('## 3. D1 — per-event: pass vs fail\n\n')
    lines.append(
        'Features computed over the full event. '
        'Sorted by |d\'| descending.\n\n',
    )
    lines.append(_render_comparison_table(d1_rows, 'pass', 'fail'))
    lines.append('\n')

    lines.append('## 4. D2 / D4 — missed vs recalled pion inside failure events\n\n')
    lines.append(
        'Restricted to the '
        f'{n_fail:,} failure events. '
        'Three-way slice of the GT pions: `missed` = GT not in top-K; '
        '`recalled` = GT in top-K (the recalled siblings). '
        'Last row `gt_n_gt_neighbors` answers D4 (graph-noise hazard).\n\n',
    )
    lines.append(_render_comparison_table(d2_rows, 'missed', 'recalled'))
    lines.append('\n')

    lines.append('## 5. Method\n\n')
    lines.append(
        'Join: `(event_run, event_id, event_luminosity_block, '
        'source_batch_id, source_microbatch_id)` across `results.parquet` '
        'and `val_00*.parquet`. '
        'Per-pion features read from the val parquet using the GT '
        'indices already stored in `results.parquet`. '
        '(η, φ) kNN uses ΔR² proxy with φ wraparound; self is excluded. '
        'd-prime = |μ_a − μ_b| / √((σ_a² + σ_b²)/2).\n',
    )

    report_path.write_text(''.join(lines))
    logger.info('Wrote report to %s', report_path)

    # ---- Optional CSVs ----
    if args.write_csv:
        per_event_csv = out_dir / 'perfect_recall_per_event.csv'
        per_pion_csv = out_dir / 'perfect_recall_per_pion.csv'
        _write_per_event_csv(per_event_csv, events, args.K)
        _write_per_pion_csv(per_pion_csv, per_pion_records)
        logger.info(
            'Wrote CSVs: %s, %s', per_event_csv, per_pion_csv,
        )


def _write_per_event_csv(
    path: Path,
    events: list[dict],
    K: int,
) -> None:
    fieldnames = [
        *_VAL_KEY_COLUMNS, 'perfect_recall', 'n_gt_in_top_k',
        *_EVENT_FEATURES,
    ]
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for ev in events:
            row = dict(zip(_VAL_KEY_COLUMNS, ev['key']))
            row['perfect_recall'] = int(
                event_perfect_recall(ev['gt_indices'], ev['top_k_indices'], K=K),
            )
            row['n_gt_in_top_k'] = count_gt_in_top_k(
                ev['gt_indices'], ev['top_k_indices'], K=K,
            )
            row.update(ev['event_features'])
            writer.writerow(row)


def _write_per_pion_csv(
    path: Path,
    pion_records: list[dict],
) -> None:
    fieldnames = [
        *_VAL_KEY_COLUMNS, 'pion_index', 'recalled', 'event_passed',
        *_PION_FEATURES, 'gt_n_gt_neighbors',
    ]
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec in pion_records:
            row = dict(zip(_VAL_KEY_COLUMNS, rec['key']))
            row['pion_index'] = rec['pion_index']
            row['recalled'] = int(rec['recalled'])
            row['event_passed'] = int(rec['event_passed'])
            for name in _PION_FEATURES:
                row[name] = rec.get(name)
            row['gt_n_gt_neighbors'] = rec.get('gt_n_gt_neighbors')
            writer.writerow(row)


if __name__ == '__main__':
    main()
