"""Prefilter stratified evaluation — validate the 'hard-event' subset.

Consumes the per-event CSV emitted by
``prefilter_perfect_recall_diagnostic.py`` (one row per val event, with
event-level features + ``perfect_recall`` flag) and measures how well
different stratification rules capture the failure mass.

For every candidate rule the report surfaces:
    * ``n_hard`` — events satisfying the rule (the "hard" class).
    * ``fraction_hard`` — hard as a fraction of all events.
    * ``fail_in_hard`` — failures landing inside hard.
    * ``fail_precision`` — failure rate inside hard (``fail_in_hard /
      n_hard``).
    * ``fail_recall`` — failures captured out of all failures
      (``fail_in_hard / n_fail_total``).
    * ``p_at_k_hard`` and ``p_at_k_easy`` — P@K on each partition; used
      for the "how bad is hard, how clean is easy" read.

Also evaluates combined multi-feature rules (AND / OR) for the two
strongest D1 discriminators and picks the best threshold by Fβ score
(β > 1 favours recall, appropriate when "capture most failures" matters
more than "keep the hard class small").

Usage::

    python -m diagnostics.prefilter_stratified_eval \\
        --per-event-csv part/reports/perfect_recall_per_event.csv \\
        --output-dir part/reports \\
        --beta 1.5
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
from typing import Callable

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


logger = logging.getLogger('prefilter_stratified_eval')
logging.basicConfig(
    level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
)


# ---------------------------------------------------------------------------
# Pure rule-evaluation primitives
# ---------------------------------------------------------------------------


def evaluate_stratification(
    events: list[dict],
    rule_name: str,
    predicate: Callable[[dict], bool],
) -> dict:
    """Partition events by ``predicate`` and compute the hard/easy metrics.

    Returns a dict with keys: ``rule``, ``n_hard``, ``n_easy``,
    ``fraction_hard``, ``fail_in_hard``, ``fail_in_easy``,
    ``n_fail_total``, ``fail_precision``, ``fail_recall``,
    ``p_at_k_hard``, ``p_at_k_easy``.

    P@K definition follows the per-event CSV: ``perfect_recall == 1`` is
    a pass at the K the source diagnostic used (default K=256).
    Empty-subset P@K is reported as NaN so downstream formatters can
    render it distinctly.
    """
    n_total = len(events)
    hard = [e for e in events if predicate(e)]
    easy = [e for e in events if not predicate(e)]
    n_hard = len(hard)
    n_easy = len(easy)

    fail_in_hard = sum(1 for e in hard if int(e['perfect_recall']) == 0)
    fail_in_easy = sum(1 for e in easy if int(e['perfect_recall']) == 0)
    n_fail_total = fail_in_hard + fail_in_easy

    def _p_at_k(subset: list[dict]) -> float:
        if not subset:
            return float('nan')
        passes = sum(1 for e in subset if int(e['perfect_recall']) == 1)
        return passes / len(subset)

    fail_precision = fail_in_hard / n_hard if n_hard else 0.0
    fail_recall = fail_in_hard / n_fail_total if n_fail_total else 0.0
    fraction_hard = n_hard / n_total if n_total else 0.0

    return {
        'rule': rule_name,
        'n_total': n_total,
        'n_hard': n_hard,
        'n_easy': n_easy,
        'fraction_hard': fraction_hard,
        'fail_in_hard': fail_in_hard,
        'fail_in_easy': fail_in_easy,
        'n_fail_total': n_fail_total,
        'fail_precision': fail_precision,
        'fail_recall': fail_recall,
        'p_at_k_hard': _p_at_k(hard),
        'p_at_k_easy': _p_at_k(easy),
    }


def sweep_threshold(
    events: list[dict],
    feature: str,
    thresholds: list[float],
    direction: str = 'lt',
) -> list[dict]:
    """Evaluate a stratification rule at each threshold.

    ``direction``:
        * ``'lt'`` — predicate is ``event[feature] < threshold``
        * ``'gt'`` — predicate is ``event[feature] > threshold``
        * ``'le'`` — ``<=``; ``'ge'`` — ``>=``.
    Rule name is formatted as ``{feature}{op}{threshold}`` so downstream
    code can identify it unambiguously.
    """
    if direction not in ('lt', 'gt', 'le', 'ge'):
        raise ValueError(
            f"direction must be one of lt/gt/le/ge, got {direction!r}",
        )
    op_symbol = {'lt': '<', 'gt': '>', 'le': '<=', 'ge': '>='}[direction]

    def _make_predicate(threshold_value):
        if direction == 'lt':
            return lambda e: float(e[feature]) < threshold_value
        if direction == 'gt':
            return lambda e: float(e[feature]) > threshold_value
        if direction == 'le':
            return lambda e: float(e[feature]) <= threshold_value
        return lambda e: float(e[feature]) >= threshold_value

    rows = []
    for threshold in thresholds:
        rule_name = f'{feature}{op_symbol}{threshold}'
        result = evaluate_stratification(
            events, rule_name, _make_predicate(threshold),
        )
        result['feature'] = feature
        result['threshold'] = threshold
        result['direction'] = direction
        rows.append(result)
    return rows


def find_best_threshold(rows: list[dict], beta: float = 1.0) -> dict:
    """Return the row with the highest Fβ score on (precision, recall).

    ``Fβ = (1 + β²) · P · R / (β² · P + R)``. β > 1 tilts toward recall.
    Rows with zero precision AND zero recall score 0.
    """
    def _fbeta(row: dict) -> float:
        p = row['fail_precision']
        r = row['fail_recall']
        denom = beta ** 2 * p + r
        if denom <= 0:
            return 0.0
        return (1 + beta ** 2) * p * r / denom

    if not rows:
        raise ValueError('rows must be non-empty')
    best = max(rows, key=_fbeta)
    best_with_score = dict(best)
    best_with_score['f_beta'] = _fbeta(best)
    best_with_score['beta'] = beta
    return best_with_score


def combined_rule_predicate(
    predicates: list[Callable[[dict], bool]],
    mode: str = 'AND',
) -> Callable[[dict], bool]:
    """Combine multiple single-feature predicates into one.

    ``mode='AND'`` → event must satisfy every predicate.
    ``mode='OR'`` → event must satisfy at least one predicate.
    """
    if mode not in ('AND', 'OR'):
        raise ValueError(f"mode must be 'AND' or 'OR', got {mode!r}")
    if mode == 'AND':
        return lambda e: all(p(e) for p in predicates)
    return lambda e: any(p(e) for p in predicates)


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------


def load_per_event_csv(path: str) -> list[dict]:
    """Load the per-event CSV emitted by the perfect-recall diagnostic.

    Numeric columns get parsed to float; ``perfect_recall`` stays as
    int-as-str and the caller coerces. Keys from the composite event
    tuple are preserved as strings (they're identifiers, not numerics).
    """
    events: list[dict] = []
    numeric_columns = {
        'n_tracks', 'event_pt_median', 'event_pt_max', 'event_pt_std',
        'event_pt_p95', 'mean_abs_dz_sig', 'mean_abs_dxy_sig',
        'mean_chi2',
        'gt_pt_min', 'gt_pt_mean', 'gt_pt_max', 'gt_pt_sum',
        'gt_pt_spread', 'vertex_z',
    }
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed: dict = {}
            for k, v in row.items():
                if k in numeric_columns:
                    parsed[k] = float(v)
                elif k in ('perfect_recall', 'n_gt_in_top_k'):
                    parsed[k] = int(v)
                else:
                    parsed[k] = v
            events.append(parsed)
    logger.info('Loaded %d events from %s', len(events), path)
    return events


# ---------------------------------------------------------------------------
# Threshold picking (by quantile of the fail subset)
# ---------------------------------------------------------------------------


def quantile_thresholds(
    events: list[dict],
    feature: str,
    quantile_values: list[float],
) -> list[float]:
    """Return thresholds at the given quantiles of ``feature`` over all
    events. Uses the standard linear-interpolation quantile definition.
    """
    values = sorted(float(e[feature]) for e in events)
    n = len(values)
    if n == 0:
        return []
    thresholds = []
    for q in quantile_values:
        q = max(0.0, min(1.0, q))
        pos = q * (n - 1)
        lo = math.floor(pos)
        hi = math.ceil(pos)
        if lo == hi:
            thresholds.append(values[int(pos)])
        else:
            frac = pos - lo
            thresholds.append(values[lo] * (1 - frac) + values[hi] * frac)
    return thresholds


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


def _format_sweep_table(rows: list[dict]) -> str:
    header = (
        '| rule | n_hard | fraction_hard | fail_in_hard | fail_precision | '
        'fail_recall | P@K (hard) | P@K (easy) |\n'
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


_FEATURE_DIRECTIONS = {
    # Feature, direction, long description (where low is hard).
    'gt_pt_sum': ('lt', 'Low total GT pT → event is soft'),
    'gt_pt_mean': ('lt', 'Low mean GT pT'),
    'gt_pt_max': ('lt', 'Low max GT pT'),
    'gt_pt_min': ('lt', 'Low softest-GT pT'),
    # High is hard:
    'n_tracks': ('gt', 'High track count → crowded event'),
}

# Quantiles to sweep (coarse + a dense region near the median).
_SWEEP_QUANTILES = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Validate hard-event stratification rules for the prefilter.',
    )
    parser.add_argument(
        '--per-event-csv', type=str, required=True,
        help='Path to perfect_recall_per_event.csv (written by the perfect-recall diagnostic).',
    )
    parser.add_argument(
        '--output-dir', type=str, required=True,
        help='Directory to write the stratified-eval markdown report.',
    )
    parser.add_argument(
        '--beta', type=float, default=1.5,
        help=(
            'Fβ tradeoff. β>1 weighs recall (capturing failures) more than '
            'precision (keeping hard-class small). Default 1.5.'
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)

    events = load_per_event_csv(args.per_event_csv)
    n_total = len(events)
    n_fail = sum(1 for e in events if int(e['perfect_recall']) == 0)
    baseline_fail_rate = n_fail / n_total if n_total else 0.0
    logger.info(
        'n=%d events, %d fail (%.4f). Baseline fail rate.',
        n_total, n_fail, baseline_fail_rate,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Per-feature sweeps ----
    per_feature_rows: dict[str, list[dict]] = {}
    for feature, (direction, _description) in _FEATURE_DIRECTIONS.items():
        thresholds = quantile_thresholds(
            events, feature, _SWEEP_QUANTILES,
        )
        # For 'gt' rules we want the high-quantile thresholds.
        if direction == 'gt':
            thresholds = quantile_thresholds(
                events, feature, [1.0 - q for q in _SWEEP_QUANTILES],
            )
        per_feature_rows[feature] = sweep_threshold(
            events, feature, thresholds, direction=direction,
        )

    # ---- Best per-feature thresholds by Fβ ----
    best_per_feature = {
        feat: find_best_threshold(rows, beta=args.beta)
        for feat, rows in per_feature_rows.items()
    }

    # ---- Combined rules: best GT-pT rule × best crowding rule ----
    # `gt_pt_sum`, `gt_pt_mean`, `gt_pt_max`, `gt_pt_min` are all
    # per-GT-pT views of the same three pT values, so they're
    # near-redundant (gt_pt_mean ≡ gt_pt_sum / 3 exactly). Pairing two
    # of them together gives an AND/OR rule identical to the stronger
    # one alone. Instead, pair the best GT-pT rule with the best
    # crowding (n_tracks) rule — they are the only orthogonal axes in
    # the D1 feature set.
    gt_pt_features = {
        feat for feat in best_per_feature
        if feat.startswith('gt_pt_')
    }
    best_gt_pt = max(
        (r for feat, r in best_per_feature.items() if feat in gt_pt_features),
        key=lambda r: r['f_beta'],
        default=None,
    )
    best_crowding = max(
        (r for feat, r in best_per_feature.items() if feat not in gt_pt_features),
        key=lambda r: r['f_beta'],
        default=None,
    )

    logger.info(
        'Pairing: GT-pT leader %r (Fβ=%.4f) × crowding leader %r (Fβ=%.4f)',
        best_gt_pt['rule'] if best_gt_pt else None,
        best_gt_pt['f_beta'] if best_gt_pt else float('nan'),
        best_crowding['rule'] if best_crowding else None,
        best_crowding['f_beta'] if best_crowding else float('nan'),
    )

    combined_rows: list[dict] = []
    if best_gt_pt is not None and best_crowding is not None:
        feat_a, feat_b = best_gt_pt, best_crowding
        # Reconstruct predicates from their (feature, direction,
        # threshold) fingerprints.
        def _build_predicate(entry):
            feat = entry['feature']
            thresh = entry['threshold']
            direction = entry['direction']
            if direction == 'lt':
                return lambda e: float(e[feat]) < thresh
            if direction == 'gt':
                return lambda e: float(e[feat]) > thresh
            if direction == 'le':
                return lambda e: float(e[feat]) <= thresh
            return lambda e: float(e[feat]) >= thresh

        p_a = _build_predicate(feat_a)
        p_b = _build_predicate(feat_b)

        for mode in ('AND', 'OR'):
            predicate = combined_rule_predicate([p_a, p_b], mode=mode)
            rule_name = f'{feat_a["rule"]} {mode} {feat_b["rule"]}'
            combined_rows.append(
                evaluate_stratification(events, rule_name, predicate),
            )

    # ---- Render markdown ----
    today = datetime.date.today().isoformat().replace('-', '')
    report_path = out_dir / f'prefilter_stratified_eval_{today}.md'

    lines: list[str] = []
    lines.append(
        f'# Prefilter stratified eval — {datetime.date.today().isoformat()}\n\n',
    )
    lines.append(
        f'Source: `{args.per_event_csv}` ({n_total:,} events, '
        f'{n_fail:,} fails; baseline fail rate = '
        f'{baseline_fail_rate:.4f}).\n'
        f'Scoring rule: Fβ with β = {args.beta} '
        '(recall-weighted).\n\n',
    )

    lines.append('## 1. Per-feature threshold sweeps\n\n')
    for feature, rows in per_feature_rows.items():
        direction, description = _FEATURE_DIRECTIONS[feature]
        lines.append(f'### 1.{feature}  —  *{description}*\n\n')
        lines.append(f'Direction: `{feature} {direction}`.\n\n')
        lines.append(_format_sweep_table(rows))
        lines.append('\n')

    lines.append('## 2. Best standalone thresholds (ranked by Fβ)\n\n')
    ranked = sorted(
        best_per_feature.values(),
        key=lambda r: -r['f_beta'],
    )
    lines.append(_format_sweep_table(ranked))
    lines.append('\n')
    lines.append('F_β scores (recall-weighted, β={:.1f}):\n\n'.format(args.beta))
    for r in ranked:
        lines.append(f'- `{r["rule"]}` → Fβ = {r["f_beta"]:.4f}\n')
    lines.append('\n')

    if combined_rows:
        lines.append('## 3. Combined rules (top-2 features)\n\n')
        lines.append(_format_sweep_table(combined_rows))
        lines.append('\n')

    lines.append('## 4. Method\n\n')
    lines.append(
        'Predicate-based partition of val events into `hard` (satisfies '
        'rule) and `easy` (does not). P@K on each partition uses the '
        '`perfect_recall` flag from the source diagnostic — pass iff all '
        'GT pions were in top-K at the K that diagnostic used. '
        '`fail_precision` is failure rate inside `hard`; `fail_recall` '
        'is the fraction of all failures captured by the rule. '
        'Thresholds are sampled at quantiles of the global feature '
        'distribution (not the failure subset) so the rule can be '
        'ported to train-time sampling without re-calibration.\n',
    )

    report_path.write_text(''.join(lines))
    logger.info('Wrote report to %s', report_path)


if __name__ == '__main__':
    main()
