"""Evaluate the trained CoupleReranker on the validation set and export
results as a parquet file.

Output columns (each field stored in both index and pT forms):
    ``event_run``, ``event_id``, ``event_luminosity_block``,
    ``source_batch_id``, ``source_microbatch_id`` — composite event key
    ``couple_indices`` / ``couple_pt`` — up to 200 track pairs,
        sorted by reranker confidence descending
    ``remaining_pion_indices`` / ``remaining_pion_pt`` — up to 256
        tracks from the Stage 1 (prefilter) top-K1 selection
    ``gt_pion_indices`` / ``gt_pion_pt`` — ground-truth pion tracks

Index columns use per-event array positions; pT columns use the
original ``track_pt`` from the source parquet (preloaded before eval).
Requires ``num_workers=0`` to guarantee positional alignment between
the DataLoader output and the preloaded track_pt arrays.

Usage::

    python diagnostics/eval_couple_reranker.py \\
        --cascade-checkpoint models/cascade_best.pt \\
        --couple-checkpoint models/couple_reranker_best.pt \\
        --data-config data/low-pt/lowpt_tau_trackfinder.yaml \\
        --data-dir data/low-pt/val/ \\
        --output data/low-pt/eval/couple_reranker_val.parquet \\
        --top-k2 80 --device cuda:0 --batch-size 64 --num-workers 10
"""
from __future__ import annotations

import argparse
import glob
import logging
import math
import os
import sys
import time

import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.couple_features import build_couple_features_batched
from utils.training_utils import (
    extract_label_from_inputs,
    load_network_module,
    trim_to_max_valid_tracks,
)
from weaver.utils.dataset import SimpleIterDataset

logger = logging.getLogger('eval_couple_reranker')


# ---------------------------------------------------------------------------
# Core evaluation function (no data loading — pure model + tensor logic)
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_batch(
    model: torch.nn.Module,
    model_inputs: tuple[torch.Tensor, ...],
    top_k_output_couples: int = 200,
    pair_kinematics_v2: bool = False,
    pair_physics_v3: bool = False,
    pair_physics_signif: bool = False,
) -> list[dict]:
    """Run the couple reranker on one batch and return per-event results.

    Each result dict contains:
        ``couples``: list of ``[idx_i, idx_j]`` index pairs (up to
            ``top_k_output_couples``), sorted by reranker score
            descending. Indices are positions in the original event's
            track arrays.
        ``remaining_pions``: list of track indices from Stage 1 top-K1,
            sorted by Stage 1 score descending. Padded positions excluded.

    Args:
        model: ``CoupleCascadeModel`` (frozen cascade + reranker).
        model_inputs: ``(points, features, lorentz_vectors, mask)``
            tensors, already on the target device.
        top_k_output_couples: Maximum number of couples to return per
            event. Default 200.

    Returns:
        List of dicts, one per event in the batch.
    """
    points, features, lorentz_vectors, mask = model_inputs
    batch_size = points.shape[0]
    dummy_labels = torch.zeros_like(mask)

    # ---- Stage 1 → top-K1 tracks ----
    # train() mode for BatchNorm: use batch statistics instead of
    # running stats. The cascade's running stats are stale (training
    # validation loop used train() mode). The couple reranker also
    # uses train() mode to match training-time conditions — batch
    # stats are more accurate than calibrated running stats.
    model.cascade.train()
    model.couple_reranker.train()
    filtered = model.cascade._run_stage1(
        points, features, lorentz_vectors, mask, dummy_labels,
    )
    # selected_indices: (B, K1) — maps K1 positions → original event positions
    stage1_indices = filtered['selected_indices']
    stage1_valid = filtered['mask'].squeeze(1) > 0.5  # (B, K1)

    # ---- Stage 2 → scores within K1, take top-K2 ----
    stage2_scores = model.cascade.stage2(
        filtered['points'],
        filtered['features'],
        filtered['lorentz_vectors'],
        filtered['mask'],
        filtered['stage1_scores'],
    )  # (B, K1)

    top_k2 = model.top_k2
    top_k2_in_k1 = stage2_scores.topk(top_k2, dim=1).indices  # (B, K2)

    # Map K2 positions → original event positions:
    #   original_k2_positions[b, k] = stage1_indices[b, top_k2_in_k1[b, k]]
    original_k2_positions = stage1_indices.gather(1, top_k2_in_k1)  # (B, K2)

    # ---- Gather K2-level data for couple building ----
    def _gather_along_track_dim(tensor: torch.Tensor) -> torch.Tensor:
        """Gather (B, C, K1) → (B, C, K2) using top_k2_in_k1."""
        num_channels = tensor.shape[1]
        expanded = top_k2_in_k1.unsqueeze(1).expand(-1, num_channels, -1)
        return tensor.gather(2, expanded)

    k2_features = _gather_along_track_dim(filtered['features'])
    k2_points = _gather_along_track_dim(filtered['points'])
    k2_lorentz = _gather_along_track_dim(filtered['lorentz_vectors'])
    k2_stage1_scores = filtered['stage1_scores'].gather(1, top_k2_in_k1)
    k2_stage2_scores = stage2_scores.gather(1, top_k2_in_k1)
    k2_labels = filtered['track_labels'].squeeze(1).gather(1, top_k2_in_k1)

    # ---- Build couple features + score ----
    couple_inputs = build_couple_features_batched(
        top_k2_features=k2_features,
        top_k2_points=k2_points,
        top_k2_lorentz=k2_lorentz,
        top_k2_stage1_scores=k2_stage1_scores,
        top_k2_stage2_scores=k2_stage2_scores,
        top_k2_track_labels=k2_labels,
        pair_kinematics_v2=pair_kinematics_v2,
        pair_physics_v3=pair_physics_v3,
        pair_physics_signif=pair_physics_signif,
    )
    couple_features = couple_inputs['couple_features']
    filter_a_mask = couple_inputs['filter_a_mask']

    # Pass k2_features to support H6 event-context mode; unused for all
    # other couple_embed_mode / event_context combinations.
    scores = model.couple_reranker(
        couple_features, k2_features=k2_features,
    )  # (B, n_couples)

    # ---- Canonical couple indices within K2 ----
    couple_i, couple_j = torch.triu_indices(
        top_k2, top_k2, offset=1, device=scores.device,
    )

    # ---- Per-event: sort, take top-K, map to original indices ----
    # Top-256 prefilter tracks: sort stage1_scores desc, valid only, up to 256
    stage1_scores_masked = filtered['stage1_scores'].clone()
    stage1_scores_masked[~stage1_valid] = float('-inf')
    stage1_sorted_positions = torch.argsort(
        stage1_scores_masked, dim=1, descending=True,
    )  # (B, K1) positions within K1

    # Top-100 ParT tracks: stage2_scores are already -inf on invalid K1
    # positions (CascadeReranker.forward masked_fill). Sort desc, take 100.
    top100_part_in_k1 = stage2_scores.topk(
        min(100, stage2_scores.shape[1]), dim=1,
    ).indices  # (B, K_out)
    top100_part_original = stage1_indices.gather(
        1, top100_part_in_k1,
    )  # (B, K_out) positions in original event
    top100_part_scores = stage2_scores.gather(
        1, top100_part_in_k1,
    )  # (B, K_out)

    results: list[dict] = []
    for batch_index in range(batch_size):
        event_scores = scores[batch_index].clone()
        event_filter = filter_a_mask[batch_index]
        event_scores[~event_filter] = float('-inf')

        sorted_couple_indices = torch.argsort(event_scores, descending=True)
        top_indices = sorted_couple_indices[:top_k_output_couples]
        # Keep only valid (Filter A passing) couples
        valid_in_top = event_filter[top_indices]
        top_indices = top_indices[valid_in_top]

        # Map couple K2-positions → original event track indices
        track_i_original = original_k2_positions[
            batch_index, couple_i[top_indices]
        ]
        track_j_original = original_k2_positions[
            batch_index, couple_j[top_indices]
        ]
        couples = torch.stack(
            [track_i_original, track_j_original], dim=1,
        ).tolist()

        # Top-256 prefilter: sorted by stage1 score, valid only, in original indices
        event_valid = stage1_valid[batch_index]
        sorted_positions = stage1_sorted_positions[batch_index]
        valid_sorted = sorted_positions[event_valid[sorted_positions]]
        top256_prefilter = stage1_indices[
            batch_index, valid_sorted[:256],
        ].tolist()

        # Top-100 ParT: filter -inf (padding) entries, already sorted
        event_part_scores = top100_part_scores[batch_index]
        valid_part_mask = torch.isfinite(event_part_scores)
        top100_part = top100_part_original[
            batch_index, valid_part_mask,
        ].tolist()

        # Remaining pions: valid Stage 1 tracks as original indices
        remaining = stage1_indices[batch_index, event_valid].tolist()

        results.append({
            'couples': couples,
            'remaining_pions': remaining,
            'top256_prefilter': top256_prefilter,
            'top100_part': top100_part,
        })

    return results


# ---------------------------------------------------------------------------
# Parquet writer
# ---------------------------------------------------------------------------

def write_results_parquet(
    results_rows: list[dict],
    output_path: str,
) -> None:
    """Write evaluation results to a parquet file.

    Each row has both index-based and pT-based representations for
    couples, remaining pions, and GT pions.

    Args:
        results_rows: List of per-event result dicts.
        output_path: Path to the output parquet file.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    table = pa.table({
        'event_run': pa.array(
            [r['event_run'] for r in results_rows], type=pa.int32()),
        'event_id': pa.array(
            [r['event_id'] for r in results_rows], type=pa.int64()),
        'event_luminosity_block': pa.array(
            [r['event_luminosity_block'] for r in results_rows],
            type=pa.int32()),
        'source_batch_id': pa.array(
            [r['source_batch_id'] for r in results_rows], type=pa.int32()),
        'source_microbatch_id': pa.array(
            [r['source_microbatch_id'] for r in results_rows],
            type=pa.int32()),
        'top_256_prefilter_indices': pa.array(
            [r['top_256_prefilter_indices'] for r in results_rows],
            type=pa.list_(pa.int32())),
        'top_256_prefilter_pt': pa.array(
            [r['top_256_prefilter_pt'] for r in results_rows],
            type=pa.list_(pa.float32())),
        'top_100_part_indices': pa.array(
            [r['top_100_part_indices'] for r in results_rows],
            type=pa.list_(pa.int32())),
        'top_100_part_pt': pa.array(
            [r['top_100_part_pt'] for r in results_rows],
            type=pa.list_(pa.float32())),
        'couple_indices': pa.array(
            [r['couple_indices'] for r in results_rows],
            type=pa.list_(pa.list_(pa.int32()))),
        'couple_pt': pa.array(
            [r['couple_pt'] for r in results_rows],
            type=pa.list_(pa.list_(pa.float32()))),
        'remaining_pion_indices': pa.array(
            [r['remaining_pion_indices'] for r in results_rows],
            type=pa.list_(pa.int32())),
        'remaining_pion_pt': pa.array(
            [r['remaining_pion_pt'] for r in results_rows],
            type=pa.list_(pa.float32())),
        'gt_pion_indices': pa.array(
            [r['gt_pion_indices'] for r in results_rows],
            type=pa.list_(pa.int32())),
        'gt_pion_pt': pa.array(
            [r['gt_pion_pt'] for r in results_rows],
            type=pa.list_(pa.float32())),
    })
    pq.write_table(table, output_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            'Evaluate the CoupleReranker on the validation set and '
            'export results as a parquet file with track indices.'
        ),
    )
    parser.add_argument(
        '--cascade-checkpoint', type=str, required=True,
        help='Path to the frozen cascade checkpoint (Stage 1 + 2).',
    )
    parser.add_argument(
        '--couple-checkpoint', type=str, required=True,
        help='Path to the trained CoupleReranker slim checkpoint.',
    )
    parser.add_argument(
        '--data-config', type=str,
        default='data/low-pt/lowpt_tau_trackfinder.yaml',
    )
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Directory containing .parquet files to evaluate.')
    parser.add_argument(
        '--network', type=str,
        default='networks/lowpt_tau_CoupleReranker.py',
    )
    parser.add_argument('--output', type=str,
                        default='data/low-pt/eval/couple_reranker_val.parquet')
    parser.add_argument('--top-k2', type=int, default=80)
    parser.add_argument('--top-k-output-couples', type=int, default=200,
                        help='Max couples per event in output (default 200).')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--num-workers', type=int, default=10)
    parser.add_argument('--max-events', type=int, default=None,
                        help='Max events to process (default: all). '
                             'Useful for quick smoke tests.')
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    device = torch.device(args.device)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    )

    # ---- Discover parquet files ----
    parquet_files = sorted(glob.glob(os.path.join(args.data_dir, '*.parquet')))
    if not parquet_files:
        logger.error(f'No .parquet files found in {args.data_dir}')
        sys.exit(1)
    logger.info(f'Found {len(parquet_files)} parquet files in {args.data_dir}')

    # Count total events for max_steps (avoid infinite re-iteration in
    # in-memory mode).
    total_events = sum(
        pq.read_metadata(file_path).num_rows for file_path in parquet_files
    )
    logger.info(f'Total events: {total_events}')

    # ---- Data loading ----
    # Requires the auto-generated .auto.yaml (standardization params)
    # next to the data config. If missing, generate it first:
    #   python -c "from weaver.utils.dataset import SimpleIterDataset; \
    #     SimpleIterDataset({'data': ['data/low-pt/train/*.parquet']}, \
    #     'data/low-pt/lowpt_tau_trackfinder.yaml', for_training=True)"
    file_dict = {'data': parquet_files}
    # in_memory=False so the dataset iterates once through all files and
    # raises StopIteration naturally — no infinite-loop risk.
    dataset = SimpleIterDataset(
        file_dict, args.data_config,
        for_training=False,
        load_range_and_fraction=((0, 1), 1),
        in_memory=False,
    )
    data_config_obj = dataset._data_config

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        drop_last=False,
        pin_memory=device.type == 'cuda',
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
    )

    events_to_process = total_events
    if args.max_events is not None:
        events_to_process = min(total_events, args.max_events)
    max_steps = math.ceil(events_to_process / args.batch_size)
    logger.info(
        f'Batch size: {args.batch_size}, events: {events_to_process}, '
        f'max steps: {max_steps}, workers: {args.num_workers}',
    )

    # ---- Load checkpoint first to recover training-time config ----
    checkpoint = torch.load(
        args.couple_checkpoint, map_location=device, weights_only=False,
    )
    ckpt_args = checkpoint.get('args', {}) or {}

    # Reranker architecture knobs that affect input_dim / param shape.
    # Pulled from the checkpoint so eval doesn't need to mirror each
    # training flag via CLI.
    model_kwargs = {
        'cascade_checkpoint': args.cascade_checkpoint,
        'top_k2': args.top_k2,
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
    ):
        if key in ckpt_args:
            model_kwargs[key] = ckpt_args[key]

    # Feature-builder flags need to be threaded into evaluate_batch so the
    # per-batch couple feature tensor has the right number of dims.
    pair_kinematics_v2 = bool(ckpt_args.get('pair_kinematics_v2', False))
    pair_physics_v3 = bool(ckpt_args.get('pair_physics_v3', False))
    pair_physics_signif = bool(ckpt_args.get('pair_physics_signif', False))
    logger.info(
        f"Eval config from checkpoint: embed_mode="
        f"{ckpt_args.get('couple_embed_mode', 'concat')}, "
        f"projector_dim={ckpt_args.get('couple_projector_dim', 0)}, "
        f"pair_kinematics_v2={pair_kinematics_v2}",
    )

    # ---- Build model ----
    network_module = load_network_module(args.network)
    model, _ = network_module.get_model(data_config_obj, **model_kwargs)
    model = model.to(device)

    model.couple_reranker.load_state_dict(
        checkpoint['couple_reranker_state_dict'],
    )
    model.eval()

    trainable_params = sum(
        p.numel() for p in model.couple_reranker.parameters()
    )
    logger.info(f'CoupleReranker loaded: {trainable_params:,} params')

    # ---- Resolve input indices ----
    input_names = list(data_config_obj.input_names)
    mask_input_index = input_names.index('pf_mask')
    label_input_index = input_names.index('pf_label')

    # ---- Preload track_pt from source parquet, keyed by composite key ----
    # Loaded separately (not as an observer) to avoid the overhead of
    # collating variable-length awkward arrays in every DataLoader batch.
    # Keyed on (event_run, event_id, event_luminosity_block, source_batch_id,
    # source_microbatch_id) so the dataloader's `selection` filter (drops
    # events with event_n_tracks == 0) does not break alignment.
    logger.info(
        'Preloading track_pt from source parquet files (composite-key '
        'indexed)...',
    )
    key_columns = [
        'event_run', 'event_id', 'event_luminosity_block',
        'source_batch_id', 'source_microbatch_id',
    ]
    track_pt_by_key: dict[tuple, list[float]] = {}
    for file_path in parquet_files:
        file_table = pq.read_table(
            file_path, columns=['track_pt', *key_columns],
        )
        key_cols = {k: file_table.column(k).to_pylist() for k in key_columns}
        pt_col = file_table.column('track_pt').to_pylist()
        for row_index in range(file_table.num_rows):
            key = (
                int(key_cols['event_run'][row_index]),
                int(key_cols['event_id'][row_index]),
                int(key_cols['event_luminosity_block'][row_index]),
                int(key_cols['source_batch_id'][row_index]),
                int(key_cols['source_microbatch_id'][row_index]),
            )
            track_pt_by_key[key] = pt_col[row_index]
    logger.info(f'Preloaded track_pt for {len(track_pt_by_key)} events')

    # ---- Evaluation loop ----
    all_results: list[dict] = []
    start_time = time.time()

    for batch_index, (X, _, Z) in enumerate(loader):
        if batch_index >= max_steps:
            break

        inputs = [X[k].to(device) for k in data_config_obj.input_names]
        inputs = trim_to_max_valid_tracks(inputs, mask_input_index)
        model_inputs, track_labels = extract_label_from_inputs(
            inputs, label_input_index,
        )

        batch_results = evaluate_batch(
            model, model_inputs,
            top_k_output_couples=args.top_k_output_couples,
            pair_kinematics_v2=pair_kinematics_v2,
            pair_physics_v3=pair_physics_v3,
            pair_physics_signif=pair_physics_signif,
        )

        # Attach event metadata and build both index + pT forms.
        # pT comes from the preloaded track_pt dict, looked up by the event's
        # composite key (not positional — the dataloader's selection filter
        # drops events with zero tracks, which would otherwise break positional
        # alignment).
        actual_batch_size = model_inputs[0].shape[0]
        labels_flat = track_labels.squeeze(1)  # (B, P)
        for event_index in range(actual_batch_size):
            result = batch_results[event_index]
            event_key = (
                int(Z['event_run'][event_index]),
                int(Z['event_id'][event_index]),
                int(Z['event_luminosity_block'][event_index]),
                int(Z['source_batch_id'][event_index]),
                int(Z['source_microbatch_id'][event_index]),
            )
            event_pt = track_pt_by_key.get(event_key)
            if event_pt is None:
                raise RuntimeError(
                    f'No track_pt for event key {event_key} — composite key '
                    f'preload missing this event.'
                )
            n_tracks = len(event_pt)

            # Rename evaluate_batch output → index columns
            result['couple_indices'] = result.pop('couples')
            result['remaining_pion_indices'] = result.pop('remaining_pions')
            result['top_256_prefilter_indices'] = result.pop('top256_prefilter')
            result['top_100_part_indices'] = result.pop('top100_part')

            # Build pT columns from indices (skip out-of-range from
            # batch padding artifacts). pT comes from the preloaded
            # track_pt list (source parquet input), not recovered from
            # model tensors.
            result['couple_indices'] = [
                [idx_i, idx_j] for idx_i, idx_j in result['couple_indices']
                if idx_i < n_tracks and idx_j < n_tracks
            ]
            result['couple_pt'] = [
                [event_pt[idx_i], event_pt[idx_j]]
                for idx_i, idx_j in result['couple_indices']
            ]
            result['remaining_pion_indices'] = [
                idx for idx in result['remaining_pion_indices']
                if idx < n_tracks
            ]
            result['remaining_pion_pt'] = [
                event_pt[idx] for idx in result['remaining_pion_indices']
            ]
            result['top_256_prefilter_indices'] = [
                idx for idx in result['top_256_prefilter_indices']
                if idx < n_tracks
            ]
            result['top_256_prefilter_pt'] = [
                event_pt[idx] for idx in result['top_256_prefilter_indices']
            ]
            result['top_100_part_indices'] = [
                idx for idx in result['top_100_part_indices']
                if idx < n_tracks
            ]
            result['top_100_part_pt'] = [
                event_pt[idx] for idx in result['top_100_part_indices']
            ]

            # GT pions: both index and pT forms
            event_labels = labels_flat[event_index]
            gt_indices = (event_labels > 0.5).nonzero(
                as_tuple=True,
            )[0].cpu().tolist()
            result['gt_pion_indices'] = gt_indices
            result['gt_pion_pt'] = [
                event_pt[idx] for idx in gt_indices
                if idx < n_tracks
            ]

            # Event metadata (composite key)
            result['event_run'] = int(Z['event_run'][event_index])
            result['event_id'] = int(Z['event_id'][event_index])
            result['event_luminosity_block'] = int(
                Z['event_luminosity_block'][event_index],
            )
            result['source_batch_id'] = int(
                Z['source_batch_id'][event_index],
            )
            result['source_microbatch_id'] = int(
                Z['source_microbatch_id'][event_index],
            )

        all_results.extend(batch_results)

        if (batch_index + 1) % 10 == 0 or batch_index == max_steps - 1:
            elapsed = time.time() - start_time
            events_per_sec = len(all_results) / elapsed if elapsed > 0 else 0
            logger.info(
                f'Batch {batch_index + 1}/{max_steps} | '
                f'{len(all_results)} events | '
                f'{events_per_sec:.0f} events/s',
            )

    # ---- Write parquet ----
    elapsed_total = time.time() - start_time
    logger.info(
        f'Evaluation complete: {len(all_results)} events in '
        f'{elapsed_total:.1f}s',
    )
    write_results_parquet(all_results, args.output)
    logger.info(f'Wrote: {args.output}')

    # Quick summary
    n_with_couples = sum(
        1 for r in all_results if len(r['couple_indices']) > 0
    )
    avg_couples = (
        sum(len(r['couple_indices']) for r in all_results)
        / max(1, len(all_results))
    )
    avg_remaining = (
        sum(len(r['remaining_pion_indices']) for r in all_results)
        / max(1, len(all_results))
    )
    logger.info(
        f'Events with ≥1 couple: {n_with_couples}/{len(all_results)} '
        f'({100 * n_with_couples / max(1, len(all_results)):.1f}%)',
    )
    logger.info(f'Avg couples/event: {avg_couples:.1f}')
    logger.info(f'Avg remaining_pions/event: {avg_remaining:.1f}')


if __name__ == '__main__':
    main()
