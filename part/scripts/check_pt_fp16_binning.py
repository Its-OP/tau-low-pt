"""Check whether ``track_pt`` values in the parquet dataset are binned at
IEEE-754 half-precision (fp16) resolution.

Hypothesis
----------
Upstream NanoAOD compression may have stored pion pT in fp16 before CMSSW
read it back as fp32. If so, every stored value ``x`` satisfies
``float(np.float16(x)) == x`` exactly, and there are at most ~65,500 unique
positive finite fp16 values in the entire dataset — even when the total
count is in the hundreds of millions.

Confirmation criteria
---------------------
1. **FP16-representable fraction** — fraction of values where
   ``np.float16(x) == x``. Close to 1.0 → confirmed.
2. **Unique cardinality** — # distinct values across all files. fp16 caps
   the positive finite range at ~65,500 distinct values; fp32 has ~2^31.
3. **ULP spacing** — for sorted unique values, compute gaps ``x[i+1] - x[i]``
   and compare to the fp16 unit-in-last-place at that magnitude,
   ``ulp_fp16(x) = 2**(floor(log2(x)) - 10)``.
   In fp16, consecutive representable values differ by exactly 1 ULP.
4. **Duplicate histogram** — top-N most common values with counts.

Usage (lxplus, LCG_106 env)
---------------------------
    cd /eos/user/o/oprostak/tau_data/parquet_low_pt_cutoff/train
    python3 check_pt_fp16_binning.py \\
        --input-dir . \\
        --log-file pt_fp16_check.log

Output
------
Writes a verbose log (one line per file + summary) to ``--log-file``
and mirrors it to stdout. Memory ceiling: aborts if the global unique
set exceeds ``--unique-cap`` (default 10,000,000) entries — that case
itself refutes the fp16 hypothesis.
"""

import os
import sys
import glob
import math
import time
import argparse
import logging
from collections import Counter

import numpy as np
import awkward as ak


# Fixed fp16 constants for the ULP check.
#   fp16 precision: 10 mantissa bits ⇒ ULP(x) = 2^(floor(log2(|x|)) - 10)
#   fp16 max finite: 65504.0
#   fp16 min normal: 2^-14 ≈ 6.103515625e-5
FP16_MANTISSA_BITS = 10
FP16_MAX_FINITE = 65504.0
FP16_MIN_NORMAL = 2.0 ** -14


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--input-dir', default='.',
        help='Directory containing *.parquet files.',
    )
    parser.add_argument(
        '--glob-pattern', default='*.parquet',
        help='Glob pattern for parquet files (relative to --input-dir).',
    )
    parser.add_argument(
        '--column', default='track_pt',
        help='Column name to analyze.',
    )
    parser.add_argument(
        '--log-file', default='pt_fp16_check.log',
        help='Path to log file (stdout is also used).',
    )
    parser.add_argument(
        '--unique-cap', type=int, default=10_000_000,
        help='Abort if unique value set grows beyond this.',
    )
    parser.add_argument(
        '--top-k', type=int, default=30,
        help='How many most-common values to report.',
    )
    parser.add_argument(
        '--max-files', type=int, default=0,
        help='If > 0, process only the first N files (for a smoke run).',
    )
    return parser.parse_args()


def configure_logging(log_path):
    logger = logging.getLogger('fp16_check')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter('%(asctime)s  %(message)s', datefmt='%H:%M:%S')

    file_handler = logging.FileHandler(log_path, mode='w')
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(fmt)
    logger.addHandler(stream_handler)
    return logger


def fp16_ulp(values):
    """ULP of fp16 at every magnitude in ``values`` (numpy float64 array).

    Formula: ulp(x) = 2^(floor(log2(|x|)) - mantissa_bits)
    For x < FP16_MIN_NORMAL we'd be in the subnormal regime where all ULPs
    equal 2^-24; we mask those separately.
    """
    magnitude = np.abs(values)
    exponents = np.floor(np.log2(magnitude)).astype(np.int64)
    return np.power(2.0, exponents - FP16_MANTISSA_BITS)


def analyze_file(path, column, logger):
    """Load one parquet, extract the track_pt column, return per-file stats.

    Returns
    -------
    dict with keys:
        total_count, fp16_exact_count, unique_counts (Counter),
        min_value, max_value, dtype_str, out_of_fp16_range_count
    """
    t0 = time.time()
    array = ak.from_parquet(path, columns=[column])[column]
    flat = ak.to_numpy(ak.flatten(array, axis=None))
    dtype_str = str(flat.dtype)
    # Cast to float64 for invariant math, but keep the original bits for
    # the exact-equality check against np.float16 round-trip.
    flat_f64 = flat.astype(np.float64, copy=False)

    total_count = flat.size
    if total_count == 0:
        return dict(
            total_count=0, fp16_exact_count=0, unique_counts=Counter(),
            min_value=float('nan'), max_value=float('nan'),
            dtype_str=dtype_str, out_of_fp16_range_count=0,
            elapsed_s=time.time() - t0,
        )

    # FP16 exact-round-trip check on the *original* dtype (not f64).
    # Equality of 0.0 after cast still counts; NaN never equals itself.
    roundtrip = flat.astype(np.float16).astype(flat.dtype)
    fp16_exact_count = int(np.sum(roundtrip == flat))

    # Range note: out-of-fp16 values (x > 65504) cannot be represented.
    finite_mask = np.isfinite(flat_f64)
    out_of_fp16_range_count = int(
        np.sum(np.abs(flat_f64[finite_mask]) > FP16_MAX_FINITE)
    )

    # Per-file unique counts. For fp16-binned data, this returns ≤ 65k entries.
    unique_values, counts = np.unique(flat_f64, return_counts=True)
    unique_counts = Counter(dict(zip(unique_values.tolist(), counts.tolist())))

    finite_values = flat_f64[finite_mask]
    if finite_values.size:
        min_value = float(finite_values.min())
        max_value = float(finite_values.max())
    else:
        min_value = max_value = float('nan')

    logger.info(
        f'FILE {os.path.basename(path)}: '
        f'N={total_count:,}  unique={len(unique_counts):,}  '
        f'fp16_exact={fp16_exact_count / total_count:.6f}  '
        f'min={min_value:.6f}  max={max_value:.6f}  '
        f'dtype={dtype_str}  elapsed={time.time() - t0:.1f}s'
    )
    return dict(
        total_count=total_count,
        fp16_exact_count=fp16_exact_count,
        unique_counts=unique_counts,
        min_value=min_value,
        max_value=max_value,
        dtype_str=dtype_str,
        out_of_fp16_range_count=out_of_fp16_range_count,
        elapsed_s=time.time() - t0,
    )


def ulp_spacing_report(unique_values_sorted, logger):
    """Gap-vs-fp16-ULP analysis on a sorted array of unique positive values."""
    values = unique_values_sorted[unique_values_sorted > 0]
    if values.size < 2:
        logger.info('ULP SPACING: <2 positive unique values; skipping.')
        return

    gaps = np.diff(values)
    midpoints = 0.5 * (values[:-1] + values[1:])

    # Only compare gaps at magnitudes in the fp16 normal range.
    normal_mask = midpoints >= FP16_MIN_NORMAL
    if not np.any(normal_mask):
        logger.info('ULP SPACING: all midpoints subnormal; skipping.')
        return

    expected_ulp = fp16_ulp(midpoints[normal_mask])
    observed = gaps[normal_mask]

    # fp16 grid gap = k * ULP for k = 1, 2, 3, ...; round ratios and check
    # how many fall within 1% of an integer.
    ratio = observed / expected_ulp
    nearest_integer = np.round(ratio)
    relative_mismatch = np.abs(ratio - nearest_integer) / np.maximum(nearest_integer, 1)
    is_integer_multiple = relative_mismatch < 0.01

    integer_multiple_rate = float(is_integer_multiple.mean())
    single_ulp_rate = float((nearest_integer == 1).mean())

    # Histogram of the nearest-integer multiplier for context.
    integer_values, integer_counts = np.unique(
        nearest_integer[is_integer_multiple].astype(np.int64), return_counts=True
    )
    histogram_preview = ', '.join(
        f'{int(k)}×ULP: {int(c):,}' for k, c in zip(integer_values[:10], integer_counts[:10])
    )

    logger.info('ULP SPACING ANALYSIS (fp16-normal range)')
    logger.info(f'  gaps analyzed:        {observed.size:,}')
    logger.info(f'  integer-multiple of ULP (≤1% err): {integer_multiple_rate:.6f}')
    logger.info(f'  exactly 1×ULP (adjacent fp16):     {single_ulp_rate:.6f}')
    logger.info(f'  multiplier histogram (first 10):   {histogram_preview}')


def main():
    args = parse_args()
    logger = configure_logging(args.log_file)

    files = sorted(glob.glob(os.path.join(args.input_dir, args.glob_pattern)))
    if args.max_files:
        files = files[: args.max_files]
    if not files:
        logger.error(f'No files matched {args.glob_pattern!r} in {args.input_dir!r}')
        sys.exit(1)

    logger.info('=' * 72)
    logger.info(f'fp16-binning check on column {args.column!r}')
    logger.info(f'input_dir={args.input_dir}  files={len(files)}')
    logger.info(f'unique_cap={args.unique_cap:,}  top_k={args.top_k}')
    logger.info('=' * 72)

    total_count = 0
    fp16_exact_count = 0
    out_of_fp16_range_count = 0
    global_counter = Counter()
    dtypes = set()
    global_min = math.inf
    global_max = -math.inf
    aborted = False

    for index, path in enumerate(files, 1):
        try:
            stats = analyze_file(path, args.column, logger)
        except Exception as exc:
            logger.error(f'FAILED on {path}: {exc!r}')
            continue

        total_count += stats['total_count']
        fp16_exact_count += stats['fp16_exact_count']
        out_of_fp16_range_count += stats['out_of_fp16_range_count']
        dtypes.add(stats['dtype_str'])
        if stats['total_count']:
            global_min = min(global_min, stats['min_value'])
            global_max = max(global_max, stats['max_value'])
        global_counter.update(stats['unique_counts'])

        logger.info(
            f'PROGRESS [{index}/{len(files)}]: '
            f'running N={total_count:,}  unique={len(global_counter):,}  '
            f'fp16_exact={fp16_exact_count / max(total_count, 1):.6f}'
        )

        if len(global_counter) > args.unique_cap:
            logger.error(
                f'ABORT: unique cardinality {len(global_counter):,} exceeded '
                f'--unique-cap {args.unique_cap:,}. Not fp16-binned.'
            )
            aborted = True
            break

    logger.info('=' * 72)
    logger.info('SUMMARY')
    logger.info('=' * 72)
    logger.info(f'files_processed:        {index if files else 0} / {len(files)}')
    logger.info(f'total_value_count:      {total_count:,}')
    logger.info(f'unique_value_count:     {len(global_counter):,}')
    logger.info(f'duplicate_rate:         '
                f'{1.0 - len(global_counter) / max(total_count, 1):.6f}')
    if total_count:
        logger.info(f'fp16_exact_rate:        '
                    f'{fp16_exact_count / total_count:.6f}')
    logger.info(f'out_of_fp16_range:      {out_of_fp16_range_count:,}  '
                f'(|x| > {FP16_MAX_FINITE})')
    logger.info(f'value_min / max:        {global_min:.6g} / {global_max:.6g}')
    logger.info(f'column_dtype(s):        {sorted(dtypes)}')

    if global_counter and not aborted:
        unique_sorted = np.array(sorted(global_counter.keys()), dtype=np.float64)
        ulp_spacing_report(unique_sorted, logger)

        top_k = global_counter.most_common(args.top_k)
        logger.info(f'TOP-{args.top_k} MOST COMMON VALUES (value, count, frac):')
        for value, count in top_k:
            logger.info(f'  {value:>14.10f}  {count:>12,}  '
                        f'{count / total_count:.6f}')

    verdict = (
        'CONFIRMED' if (
            not aborted
            and total_count
            and fp16_exact_count / total_count > 0.999
            and len(global_counter) < 200_000
        ) else 'REFUTED / inconclusive'
    )
    logger.info(f'VERDICT: fp16-binning hypothesis {verdict}')

    sys.exit(0 if not aborted else 2)


if __name__ == '__main__':
    main()
