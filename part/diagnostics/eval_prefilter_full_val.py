"""Evaluate a trained TrackPreFilter checkpoint on the full validation set.

Loads the checkpoint, infers the model config from the state dict,
applies ``disable_bn_running_stats`` (so BN forwards use batch statistics
regardless of train/eval mode), switches the model to ``.eval()``, and
reports R@K / P@K / d' / median rank over the full val loop.

This is the canonical post-BN-fix evaluation path — confirms that
``disable_bn_running_stats`` restores the training-time R@K metrics
when the model is run in ``.eval()`` (no Dropout active).

Usage:
    python -m diagnostics.eval_prefilter_full_val \\
        --checkpoint path/to/best.pt \\
        --data-config data/low-pt/lowpt_tau_trackfinder.yaml \\
        --data-dir data/low-pt/val/ \\
        --batch-size 64 --device cuda:0

Pass ``--no-disable-bn`` to skip the BN fix and reproduce the stale-
running-stats behaviour for comparison.
"""
from __future__ import annotations

import argparse
import glob
import logging
import os
import sys
import time

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from weaver.nn.model.stateless_bn import disable_bn_running_stats
from weaver.nn.model.TrackPreFilter import TrackPreFilter
from weaver.utils.dataset import SimpleIterDataset

from networks.lowpt_tau_CascadeReranker import infer_stage1_kwargs
from utils.training_utils import (
    MetricsAccumulator,
    extract_label_from_inputs,
    trim_to_max_valid_tracks,
)

VAL_METRICS_K_VALUES: tuple[int, ...] = (
    10, 20, 30, 50, 100, 200, 256, 300, 400, 500, 600, 800,
)

logger = logging.getLogger('eval_prefilter_full_val')


def _parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            'Full-val evaluation of a TrackPreFilter checkpoint with '
            'BN running-stats disabled (batch-stat mode in .eval()).'
        ),
    )
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data-config', required=True)
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--max-steps', type=int, default=None)
    parser.add_argument(
        '--stage1-num-neighbors', type=int, default=16,
        help='kNN k at inference (not recoverable from state dict).',
    )
    parser.add_argument(
        '--no-disable-bn', action='store_true',
        help='Skip disable_bn_running_stats — reproduces the stale-BN bug.',
    )
    return parser.parse_args()


def _build_model(checkpoint_path, data_config, stage1_num_neighbors):
    checkpoint = torch.load(
        checkpoint_path, map_location='cpu', weights_only=False,
    )
    state_dict = checkpoint.get('model_state_dict', checkpoint)

    inferred = infer_stage1_kwargs(
        state_dict, stage1_num_neighbors=stage1_num_neighbors,
    )
    inferred['input_dim'] = len(data_config.input_dicts['pf_features'])
    model = TrackPreFilter(**inferred)
    model.load_state_dict(state_dict)
    return model, inferred


def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
    )
    args = _parse_arguments()
    device = torch.device(args.device)

    parquet_files = sorted(glob.glob(f'{args.data_dir}/*.parquet'))
    if not parquet_files:
        raise FileNotFoundError(
            f'No .parquet files under {args.data_dir}',
        )
    dataset = SimpleIterDataset(
        {'data': parquet_files},
        data_config_file=args.data_config,
        for_training=False,
        load_range_and_fraction=((0.0, 1.0), 1.0),
        fetch_by_files=True,
        fetch_step=len(parquet_files),
        in_memory=False,
    )
    data_config = dataset.config
    loader = DataLoader(
        dataset, batch_size=args.batch_size, drop_last=False,
        num_workers=args.num_workers,
    )

    input_names = list(data_config.input_names)
    mask_input_index = input_names.index('pf_mask')
    label_input_index = input_names.index('pf_label')

    model, inferred = _build_model(
        args.checkpoint, data_config, args.stage1_num_neighbors,
    )
    model = model.to(device)
    logger.info('Inferred config: %s', inferred)
    logger.info(
        'Parameters: %d', sum(p.numel() for p in model.parameters()),
    )

    if args.no_disable_bn:
        logger.info(
            'BN running-stats NOT disabled — expect stale-stat behaviour.',
        )
    else:
        disable_bn_running_stats(model)
        logger.info('disable_bn_running_stats applied to all BN submodules.')
    model.eval()

    metrics_accumulator = MetricsAccumulator(
        k_values=VAL_METRICS_K_VALUES,
    )
    batch_count = 0
    total_start = time.time()
    with torch.no_grad():
        for batch_index, (X, _y, _z) in enumerate(loader):
            if args.max_steps is not None and batch_index >= args.max_steps:
                break
            inputs = [X[k].to(device) for k in data_config.input_names]
            inputs = trim_to_max_valid_tracks(inputs, mask_input_index)
            model_inputs, track_labels = extract_label_from_inputs(
                inputs, label_input_index,
            )
            points, features, lorentz_vectors, mask = model_inputs
            scores = model(points, features, lorentz_vectors, mask)
            metrics_accumulator.update(scores, track_labels, mask)
            batch_count += 1

    elapsed = time.time() - total_start
    metrics = metrics_accumulator.compute()
    logger.info(
        'Processed %d batches in %.1fs (%.3fs/batch)',
        batch_count, elapsed,
        elapsed / max(1, batch_count),
    )

    # Headline summary ordered the same way as the training-log val line
    # for direct eyeball comparison.
    logger.info(
        'Eval | R@30: %.4f | R@100: %.4f | R@200: %.4f | R@256: %.4f | '
        'R@500: %.4f | R@600: %.4f | P@200: %.4f | P@256: %.4f | '
        "d': %.3f | rank: %d (p90=%d)",
        metrics['recall_at_30'], metrics['recall_at_100'],
        metrics['recall_at_200'], metrics['recall_at_256'],
        metrics['recall_at_500'], metrics['recall_at_600'],
        metrics['perfect_at_200'], metrics['perfect_at_256'],
        metrics['d_prime'],
        int(metrics['median_gt_rank']),
        int(metrics['gt_rank_p90']),
    )


if __name__ == '__main__':
    main()
