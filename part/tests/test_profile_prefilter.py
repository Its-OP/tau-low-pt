"""Tests for torch.profiler integration in train_prefilter.py.

Covers:
- `--profile-steps` CLI flag parses correctly (default 0, positive integer).
- `--profile-output` CLI flag parses correctly (optional path override).
- `run_profile` function is importable and has the expected signature.
- `run_profile` runs N fwd+bwd steps under torch.profiler on CPU with a
  tiny model, emits the chrome trace JSON and summary .txt to the output
  directory.
"""
from __future__ import annotations

import argparse
import inspect
import os
import sys
import tempfile
from pathlib import Path

import pytest
import torch

PART_ROOT = Path(__file__).resolve().parent.parent
if str(PART_ROOT) not in sys.path:
    sys.path.insert(0, str(PART_ROOT))


def test_profile_steps_flag_default_zero():
    """--profile-steps defaults to 0 (profiling disabled)."""
    import train_prefilter

    parser = train_prefilter._build_argument_parser()
    args = parser.parse_args([
        '--data-config', 'x',
        '--data-dir', 'x',
        '--network', 'x',
    ])
    assert args.profile_steps == 0


def test_profile_steps_flag_positive_value():
    """--profile-steps N sets a positive integer."""
    import train_prefilter

    parser = train_prefilter._build_argument_parser()
    args = parser.parse_args([
        '--data-config', 'x',
        '--data-dir', 'x',
        '--network', 'x',
        '--profile-steps', '50',
    ])
    assert args.profile_steps == 50


def test_profile_output_flag_default_none():
    """--profile-output defaults to None (uses experiment dir)."""
    import train_prefilter

    parser = train_prefilter._build_argument_parser()
    args = parser.parse_args([
        '--data-config', 'x',
        '--data-dir', 'x',
        '--network', 'x',
    ])
    assert args.profile_output is None


def test_run_profile_importable_and_signature():
    """`run_profile` exists and has the expected parameters."""
    import train_prefilter

    assert hasattr(train_prefilter, 'run_profile')
    signature = inspect.signature(train_prefilter.run_profile)
    required_params = {
        'model', 'train_loader', 'device', 'data_config',
        'mask_input_index', 'label_input_index',
        'num_steps', 'output_dir',
    }
    assert required_params.issubset(signature.parameters.keys())


class _TinyMockDataConfig:
    """Minimal data_config stub exposing just input_names."""

    input_names = ['pf_points', 'pf_features', 'pf_vectors', 'pf_mask',
                   'pf_label']


class _TinyMockModel(torch.nn.Module):
    """Mirror the TrackPreFilter.compute_loss interface with a tiny MLP.

    Accepts (points, features, lorentz_vectors, mask, labels) — returns a
    dict with 'total_loss'. No kNN, no message passing — just sufficient
    to exercise the profiler control flow.
    """

    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(16, 1)

    def compute_loss(self, points, features, lorentz_vectors, mask,
                     labels, **kwargs):
        scores = self.linear(features.transpose(1, 2)).squeeze(-1)
        target = labels.squeeze(1).float() if labels.dim() == 3 else labels.float()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            scores, target,
        )
        return {'total_loss': loss, '_scores': scores}


def _make_tiny_loader(num_batches: int = 10, batch_size: int = 2,
                     num_tracks: int = 20):
    """Build a DataLoader that yields TrackPreFilter-shaped inputs."""
    batches = []
    for _ in range(num_batches):
        points = torch.randn(batch_size, 2, num_tracks)
        features = torch.randn(batch_size, 16, num_tracks)
        vectors = torch.randn(batch_size, 4, num_tracks)
        mask = torch.ones(batch_size, 1, num_tracks)
        labels = torch.zeros(batch_size, 1, num_tracks)
        labels[:, 0, :3] = 1
        batches.append({
            'pf_points': points,
            'pf_features': features,
            'pf_vectors': vectors,
            'pf_mask': mask,
            'pf_label': labels,
        })

    class _BatchedDataset(torch.utils.data.Dataset):
        def __init__(self, batches):
            self.batches = batches

        def __len__(self):
            return len(self.batches)

        def __getitem__(self, idx):
            batch = self.batches[idx]
            return batch, torch.zeros(1), torch.zeros(1)

    class _WrappedLoader:
        def __init__(self, dataset):
            self.dataset = dataset

        def __iter__(self):
            for item in self.dataset:
                yield item

    return _WrappedLoader(_BatchedDataset(batches))


def test_run_profile_emits_summary_by_default(tmp_path):
    """Default (lean) run_profile writes summary only — no chrome trace."""
    import train_prefilter

    model = _TinyMockModel()
    loader = _make_tiny_loader(num_batches=10)
    data_config = _TinyMockDataConfig()

    output_dir = tmp_path / 'profile'
    train_prefilter.run_profile(
        model=model,
        train_loader=loader,
        device=torch.device('cpu'),
        data_config=data_config,
        mask_input_index=3,
        label_input_index=4,
        num_steps=3,
        output_dir=str(output_dir),
    )

    trace_file = output_dir / 'profile_trace.json'
    summary_file = output_dir / 'profile_summary.txt'
    assert not trace_file.exists(), (
        f'chrome trace written despite export_chrome_trace=False: {trace_file}'
    )
    assert summary_file.exists(), f'summary not written at {summary_file}'
    summary_content = summary_file.read_text()
    assert 'Name' in summary_content or 'aten::' in summary_content


def test_run_profile_emits_chrome_trace_when_requested(tmp_path):
    """export_chrome_trace=True writes the full chrome trace JSON."""
    import train_prefilter

    model = _TinyMockModel()
    loader = _make_tiny_loader(num_batches=10)
    data_config = _TinyMockDataConfig()

    output_dir = tmp_path / 'profile'
    train_prefilter.run_profile(
        model=model,
        train_loader=loader,
        device=torch.device('cpu'),
        data_config=data_config,
        mask_input_index=3,
        label_input_index=4,
        num_steps=3,
        output_dir=str(output_dir),
        export_chrome_trace=True,
    )

    trace_file = output_dir / 'profile_trace.json'
    summary_file = output_dir / 'profile_summary.txt'
    assert trace_file.exists(), f'chrome trace not written at {trace_file}'
    assert summary_file.exists(), f'summary not written at {summary_file}'


def test_profile_verbosity_flags_default_off():
    """--profile-record-shapes / --profile-memory / --profile-chrome-trace
    all default to False so that ``--profile-steps N`` alone produces the
    smallest viable output (summary table only)."""
    import train_prefilter

    parser = train_prefilter._build_argument_parser()
    args = parser.parse_args([
        '--data-config', 'x',
        '--data-dir', 'x',
        '--network', 'x',
    ])
    assert args.profile_record_shapes is False
    assert args.profile_memory is False
    assert args.profile_chrome_trace is False


def test_profile_verbosity_flags_opt_in():
    """Opt-in flags flip their bools when passed."""
    import train_prefilter

    parser = train_prefilter._build_argument_parser()
    args = parser.parse_args([
        '--data-config', 'x',
        '--data-dir', 'x',
        '--network', 'x',
        '--profile-record-shapes',
        '--profile-memory',
        '--profile-chrome-trace',
    ])
    assert args.profile_record_shapes is True
    assert args.profile_memory is True
    assert args.profile_chrome_trace is True
