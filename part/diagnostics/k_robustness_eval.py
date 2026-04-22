"""K-robustness eval for the couple reranker.

For a trained couple-reranker checkpoint + the frozen cascade it was
trained against, sweep the Stage-2 → Stage-3 top-K2 operating point and
measure how `mean_first_gt_rank_couples` scales with K. A position-aware
loss is expected to keep `mean_rank(K=200) / mean_rank(K=60) < ~3`; the
current softmax-CE baseline lands around 5.6 (17 → 96, per the Apr-14
topk2 sweep + post-training eval).

Usage:
    python diagnostics/k_robustness_eval.py \\
        --cascade-checkpoint models/cascade_best.pt \\
        --couple-checkpoint models/couple_v3_best.pt \\
        --data-config data/low-pt/lowpt_tau_trackfinder.yaml \\
        --data-dir data/low-pt/val/ \\
        --ks 60 100 150 200 \\
        --output experiments/eval_k_robustness/result.json

Output: JSON with one entry per K containing mean_first_gt_rank and
C@{50, 75, 100, 200} computed on the full val set.
"""
import argparse
import glob
import json
import logging
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from diagnostics.eval_couple_reranker import (  # noqa: E402
    evaluate_batch,
    extract_label_from_inputs,
    load_network_module,
    trim_to_max_valid_tracks,
)
from utils.training_utils import CoupleMetricsAccumulator  # noqa: E402
from weaver.utils.data.config import DataConfig  # noqa: E402
from weaver.utils.dataset import SimpleIterDataset  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

logger = logging.getLogger('k_robustness_eval')
logging.basicConfig(
    level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
)


def _build_model(cascade_checkpoint, couple_checkpoint, network, data_config, top_k2, device):
    checkpoint = torch.load(
        couple_checkpoint, map_location=device, weights_only=False,
    )
    ckpt_args = checkpoint.get('args', {}) or {}
    model_kwargs = {
        'cascade_checkpoint': cascade_checkpoint,
        'top_k2': top_k2,
    }
    for key in (
        'couple_embed_mode', 'couple_projector_dim',
        'pair_kinematics_v2', 'pair_physics_v3', 'pair_physics_signif',
        'couple_hidden_dim', 'couple_num_residual_blocks',
        'couple_dropout',
        'couple_loss', 'couple_label_smoothing',
        'couple_ranking_num_samples', 'couple_ranking_temperature',
        'couple_hardneg_fraction', 'couple_hardneg_margin',
        'tokenize_d', 'tokenize_blocks', 'tokenize_heads',
        'event_context', 'context_dim',
        'ndcg_k', 'lambda_sigma', 'ndcg_alpha',
    ):
        if key in ckpt_args:
            model_kwargs[key] = ckpt_args[key]
    network_module = load_network_module(network)
    model, _ = network_module.get_model(data_config, **model_kwargs)
    model = model.to(device)
    model.couple_reranker.load_state_dict(
        checkpoint['couple_reranker_state_dict'],
    )
    model.eval()
    return model, ckpt_args


def _run_eval(model, loader, data_config, mask_idx, label_idx, device, k_values_couples, n_gt_in_top_k1_key=None):
    accumulator = CoupleMetricsAccumulator(
        k_values_tracks=[],  # not needed here
        k_values_couples=list(k_values_couples),
        duplet_threshold=2,
        full_triplet_threshold=3,
    )
    pair_kinematics_v2 = getattr(model.couple_reranker, 'input_dim', 0) > 0  # unused flag, detected below
    # Detect feature flags from ckpt_args in the outer loop — here we
    # mirror the main eval's pass-through.
    pair_physics_v3 = False
    pair_physics_signif = False
    with torch.no_grad():
        for X, _, _ in loader:
            inputs = [X[k].to(device) for k in data_config.input_names]
            inputs = trim_to_max_valid_tracks(inputs, mask_idx)
            model_inputs, _ = extract_label_from_inputs(inputs, label_idx)
            # evaluate_batch returns index-level results; we need
            # couple-level scores to feed the accumulator. Use model's
            # full forward via evaluate_batch (which already computes
            # scores internally) by monkey-patching — or simpler, call
            # compute_loss to retrieve `_scores` + labels + mask.
            points, features, lorentz, mask = model_inputs
            track_labels = inputs[label_idx]
            loss_dict = model.compute_loss(
                points, features, lorentz, mask, track_labels,
            )
            scores = loss_dict['_scores']
            labels = loss_dict['_couple_labels']
            cmask = loss_dict['_couple_mask']
            accumulator.update(scores, labels, cmask)
    return accumulator.compute()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cascade-checkpoint', required=True)
    parser.add_argument('--couple-checkpoint', required=True)
    parser.add_argument('--data-config', required=True)
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--network', default='networks/lowpt_tau_CoupleReranker.py')
    parser.add_argument('--ks', type=int, nargs='+', default=[60, 100, 150, 200])
    parser.add_argument(
        '--k-values-couples', type=int, nargs='+',
        default=[50, 75, 100, 200],
    )
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--output', default='experiments/k_robustness.json')
    args = parser.parse_args()

    device = torch.device(args.device)
    data_config = DataConfig.load(args.data_config, load_observers=True)
    parquet_files = sorted(glob.glob(f'{args.data_dir}/*.parquet'))
    logger.info(f'Val files: {len(parquet_files)}')

    input_names = list(data_config.input_names)
    mask_idx = input_names.index('pf_mask')
    label_idx = input_names.index('pf_label')

    results: dict[str, dict] = {}
    for k in args.ks:
        logger.info(f'--- Evaluating at top_k2 = {k} ---')
        model, _ = _build_model(
            args.cascade_checkpoint, args.couple_checkpoint, args.network,
            data_config, k, device,
        )
        dataset = SimpleIterDataset(
            {'_all_': parquet_files}, args.data_config,
            for_training=False, extra_selection=None, remake_weights=False,
            load_range_and_fraction=((0, 1), 1.0), file_fraction=1.0,
            fetch_by_files=True, fetch_step=1, infinity_mode=False,
            in_memory=True, name='val',
        )
        loader = DataLoader(
            dataset, batch_size=args.batch_size, drop_last=False,
            pin_memory=True, num_workers=0,
        )
        metrics = _run_eval(
            model, loader, data_config, mask_idx, label_idx, device,
            args.k_values_couples,
        )
        mean_rank = metrics.get('mean_first_gt_rank_couples', 0.0)
        c_metrics = {
            f'c_at_{kv}_couples': metrics.get(f'c_at_{kv}_couples', 0.0)
            for kv in args.k_values_couples
        }
        results[str(k)] = {
            'mean_first_gt_rank_couples': mean_rank,
            **c_metrics,
        }
        logger.info(
            f'top_k2={k} | mean_rank={mean_rank:.2f} | '
            + ' | '.join(f'{k_}={v:.4f}' for k_, v in c_metrics.items()),
        )

    # Scaling ratio — acceptance criterion #4 in the plan
    if str(args.ks[0]) in results and str(args.ks[-1]) in results:
        base = results[str(args.ks[0])]['mean_first_gt_rank_couples']
        tip = results[str(args.ks[-1])]['mean_first_gt_rank_couples']
        ratio = tip / base if base > 0 else 0.0
        results['_mean_rank_ratio'] = ratio
        logger.info(
            f'mean_rank(K={args.ks[-1]}) / mean_rank(K={args.ks[0]}) = '
            f'{ratio:.2f} (target < 3.0)',
        )

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f'Wrote {args.output}')


if __name__ == '__main__':
    main()
