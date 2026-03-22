This directory has 2 projects — `weaver/` and `part/`.

- **`weaver/`** is a fork of [weaver-core](https://github.com/hqucms/weaver-core) — the production-grade ML framework for high-energy physics (HEP). It contains PyTorch implementations of ParT, ParticleNet, PFN, and P-CNN.
- **`part/`** is a fork of [particle-transformer](https://github.com/jet-universe/particle_transformer) — a HEP model widely used for jet tagging. The repository acts as a wrapper around the implementation provided by weaver.

The goal of the project is to adapt the Particle Transformer for the low-pT regime. In this regime, jets degenerate into clouds of particles, which makes the standard implementation ineffective.

## Low-pT Tau Dataset

The `part/data/low-pt/` directory contains CMS NanoAOD ROOT files with B-meson decays (B⁰ → τ⁺τ⁻ K⁰* and Bₛ⁰ → τ⁺τ⁻ φ), simulated with Pythia 8 + EvtGen + CMSSW 14. See `part/data/low-pt/description.tex` for details.

- `merged_noBKstar.root` — 19,094 events, 275 branches (primary dataset)
- `bph_nano_test_mc_saved.root` — 2 events, 1,513 branches (test file)

### Converting ROOT to Parquet

```bash
# Using the shell script (recommended):
./part/scripts/convert_lowpt_tau.sh [input.root] [output.parquet]

# Or directly:
python part/utils/convert_lowpt_tau_dataset.py \
  -i part/data/low-pt/merged_noBKstar.root \
  -o part/data/low-pt/lowpt_tau_trackorigin.parquet
```

The script extracts:
- **Pion tokens** (23 features): all charged pion tracks (|pdgId| == 211) with kinematics, impact parameters, track quality, and helix covariance matrix.
- **Per-track label** (`track_label_from_tau`): 1 if the pion track originates from a tau decay, 0 otherwise (background / pile-up). From gen-level `Track_trackFromTau` truth.
- **Event-level scalars** (4 features): primary vertex coordinates and pion track count.

Output is a Parquet file with jagged awkward arrays, consumable by the weaver framework.

## Hierarchical Graph Backbone

A custom backbone that compresses ~1130 pion tracks into 64 dense tokens via PointNet++ set abstraction with physics-informed pairwise Lorentz-vector features (ln kT, ln z, ln ΔR, ln m²).

### Architecture

```
Input: (B, 7, P) features + (B, 4, P) 4-vectors + (B, 2, P) η,φ coords + (B, 1, P) mask

Stage 0: Conv1d(7 → 64) + BN + ReLU              →  (B, 64, P)
Stage 1: FPS + kNN(K=32) + EdgeConv → 256 points  →  (B, 128, 256)
Stage 2: FPS + kNN(K=24) + EdgeConv → 128 points  →  (B, 192, 128)
Stage 3: FPS + kNN(K=16) + EdgeConv →  64 points  →  (B, 256, 64)
```

Each stage: FPS selects centroids in (η, φ) → kNN finds neighbors → EdgeConv MLP processes edge features → attention-weighted aggregation → 4-vector propagation.

### Pretraining

Self-supervised masked track reconstruction (40% masking). A lightweight decoder cross-attends from masked track queries to backbone tokens and reconstructs features + 4-vectors. The decoder is discarded after pretraining.

```bash
# Quick start (screen sessions with training + GPU monitoring):
cd part && bash train_pretrain.sh

# Or run directly:
python part/pretrain_backbone.py \
    --data-config part/data/low-pt/lowpt_tau_pretrain.yaml \
    --data-dir part/data/low-pt/ \
    --network part/networks/lowpt_tau_BackbonePretrain.py \
    --epochs 100 --batch-size 32 --lr 1e-3 --amp

# Resume from checkpoint:
bash part/train_pretrain.sh --resume experiments/BackbonePretrain_20260219_173000/checkpoints/checkpoint_epoch_50.pt
```

Each run creates a new experiment folder in `part/experiments/`:
```
experiments/
└── BackbonePretrain_20260219_173000/
    ├── training.log          # full console output
    ├── loss_history.json     # per-epoch loss values
    ├── loss_curves.png       # loss and LR plots
    ├── checkpoints/
    │   ├── checkpoint_epoch_10.pt
    │   ├── best_model.pt
    │   └── backbone_best.pt
    └── tensorboard/
        └── events.out.tfevents.*
```

TensorBoard across all experiments: `tensorboard --logdir part/experiments --bind_all`

### Key Files

| File | Description |
|------|-------------|
| `weaver/weaver/nn/model/HierarchicalGraphBackbone.py` | Backbone: FPS, cross-set kNN, SetAbstractionStage, HierarchicalGraphBackbone |
| `weaver/weaver/nn/model/BackbonePretraining.py` | Pretraining: MaskedTrackDecoder, MaskedTrackPretrainer |
| `part/networks/lowpt_tau_BackbonePretrain.py` | Network wrapper with default hyperparameters |
| `part/data/low-pt/lowpt_tau_pretrain.yaml` | Data config for pretraining |
| `part/pretrain_backbone.py` | Custom training script (AdamW, cosine schedule, AMP, TensorBoard) |
| `part/train_pretrain.sh` | Training launcher (screen with GPU monitoring) |
| `setup_server.sh` | Server setup (clone, conda, dependencies, dataset) |

## Tau-Origin Track Finder

Three head architectures share the same frozen backbone and training infrastructure:

| Head | Approach | Trainable params | Key idea |
|------|----------|------------------|----------|
| **OC** (Object Condensation) | Per-track MLP | ~67K | Beta scores + clustering, no matching needed |
| **DETR** (Mask Transformer) | Encoder-decoder + queries | ~4.6M | Hungarian matching, cross-entropy mask + confidence |
| **V2** (Neighbor + Refine) | kNN messages + top-K self-attention | ~521K | Neighborhood-aware scoring + pairwise refinement |
| **V3** (GAPLayer / ABCNet) | Dual kNN attention + global context | ~TBD | ABCNet-inspired GAPLayers, feature-space kNN |

### Architecture

```
Shared: EnrichCompactBackbone (pretrained, FROZEN)
├── enrich()  → (B, 256, P)  per-track enriched features
└── compact() → (B, 256, 128) compact spatial tokens

OC Head:                              DETR Head:
├── beta_head (Conv1d MLP)            ├── FPS query init → 4L decoder (self+cross attn)
├── clustering_head (Conv1d MLP)      ├── Mask Head (dot-product + temperature)
└── Loss: focal BCE + OC potential    ├── Confidence Head
                                      └── Loss: CE mask + conf BCE

V2 Head (Neighbor + Refine):
├── kNN(16) in (η,φ) → max-pool → MLP → neighbor messages (M1)
├── Skip features: dxy_significance + pT (M5)
├── Per-track scoring: Conv1d MLP on [enriched, messages, skip]
├── Top-256 → 2L self-attention → refined scores (M2)
└── Loss: focal BCE (all tracks) + focal BCE (top-K combined)

V3 Head (ABCNet-inspired GAPLayers):
├── GAPLayer 1: kNN(16) in (η,φ) → attention-weighted edge features
├── MLPs → intermediate features
├── GAPLayer 2: kNN(16) in learned feature space
├── Global context: masked avg pool → project → tile
├── Multi-scale concat: [GAP1, GAP2, backbone, raw, LV, global]
├── Per-track scoring: Conv1d MLP on concatenated features
└── Loss: focal BCE (all tracks)
```

### Training

The training script is head-agnostic — select the head via `--network`:

```bash
# V2 head (neighbor + refine, default in train_trackfinder.sh):
python part/train_trackfinder.py \
    --network networks/lowpt_tau_TrackFinderV2.py \
    --data-config data/low-pt/lowpt_tau_trackfinder.yaml \
    --data-dir data/low-pt/ \
    --pretrained-backbone models/backbone_best.pt \
    --epochs 50 --batch-size 96 --lr 1e-4 --amp

# OC head (baseline):
python part/train_trackfinder.py \
    --network networks/lowpt_tau_TrackFinderOC.py \
    --data-config data/low-pt/lowpt_tau_trackfinder.yaml \
    --data-dir data/low-pt/ \
    --pretrained-backbone models/backbone_best.pt \
    --epochs 50 --batch-size 96 --lr 5e-4 --amp

# V3 head (ABCNet-inspired GAPLayers):
bash part/train_trackfinder_v3.sh

# Or directly:
python part/train_trackfinder.py \
    --network networks/lowpt_tau_TrackFinderV3.py \
    --data-config data/low-pt/lowpt_tau_trackfinder.yaml \
    --data-dir data/low-pt/ \
    --pretrained-backbone models/backbone_best.pt \
    --epochs 50 --batch-size 96 --lr 1e-4 --amp

# DETR head:
python part/train_trackfinder.py \
    --network networks/lowpt_tau_TrackFinder.py \
    --data-config data/low-pt/lowpt_tau_trackfinder.yaml \
    --data-dir data/low-pt/ \
    --pretrained-backbone models/backbone_best.pt \
    --mask-ce-loss-weight 2.0 --confidence-loss-weight 2.0 \
    --epochs 50 --batch-size 96 --lr 5e-4 --amp

# Quick start via shell script:
cd part && bash train_trackfinder.sh
```

### Key Files

| File | Description |
|------|-------------|
| `weaver/weaver/nn/model/TauTrackFinderV2.py` | V2 top-level module (neighbor scoring + top-K refinement) |
| `weaver/weaver/nn/model/TauTrackFinderOC.py` | OC top-level module (backbone + OC head + loss) |
| `weaver/weaver/nn/model/ObjectCondensationHead.py` | OC head (beta + clustering per track) |
| `weaver/weaver/nn/model/TauTrackFinder.py` | DETR top-level module (backbone + DETR head + loss) |
| `weaver/weaver/nn/model/TauTrackFinderHead.py` | DETR head (encoder-decoder + mask + confidence) |
| `part/networks/lowpt_tau_TrackFinderV2.py` | V2 network wrapper |
| `part/networks/lowpt_tau_TrackFinderOC.py` | OC network wrapper |
| `part/networks/lowpt_tau_TrackFinder.py` | DETR network wrapper |
| `part/train_trackfinder.py` | Head-agnostic training script |
| `part/train_trackfinder.sh` | Training launcher (screen + GPU monitoring) |

## HTCondor Training on lxplus

For large-scale training on CERN lxplus batch (GPU nodes):

```bash
cd /eos/user/o/oleh/src/part

# Submit with default settings (100 epochs, batch_size=48):
condor_submit condor/pretrain.sub

# Override training arguments at submit time:
condor_submit condor/pretrain.sub \
  training_args="--model-name MyExperiment --epochs 200 --batch-size 32"
```

The job runs training in `/tmp` on the worker node (fast local I/O) and copies results to `experiments/` on EOS when done. A placeholder directory `experiments/.running_condor_<id>/` with `job_info.txt` and periodically synced logs is created so you can monitor progress.

```bash
# Check running jobs:
condor_q

# View placeholder info:
cat experiments/.running_condor_*/job_info.txt

# SSH into the worker node:
condor_ssh_to_job <cluster_id>

# HTCondor logs (stdout/stderr from the wrapper):
ls condor/logs/
```

| File | Description |
|------|-------------|
| `part/condor/pretrain.sub` | HTCondor submit description (resource requests, training args) |
| `part/condor/run_pretrain.sh` | Worker node execution script (conda, /tmp I/O, EOS copy-back) |
| `part/condor/logs/` | HTCondor stdout/stderr/log files |

## Server Setup

For training on an external GPU server:

```bash
# Upload setup_server.sh to the server workspace, then:
chmod +x setup_server.sh
./setup_server.sh
```

This will clone both repos, create the conda environment, install all dependencies, and download the dataset from Google Drive. After setup:

```bash
conda activate part
cd part && bash train_pretrain.sh
```

## Utility Scripts

| Script | Description |
|--------|-------------|
| `setup_server.sh` | Server setup: clone repos, conda env, dependencies, dataset download |
| `part/pull.sh` | Pull latest changes for part (`bash pull.sh [branch]`, default: master) |
| `weaver/pull.sh` | Pull latest changes for weaver (`bash pull.sh [branch]`, default: master) |
| `part/train_pretrain.sh` | Launch pretraining in screen (training + nvidia-smi monitoring) |
| `part/pretrain_backbone.py` | Custom training script (AdamW, cosine schedule, AMP, TensorBoard) |
| `part/train_trackfinder.sh` | Launch track finder training in screen |
| `part/train_trackfinder.py` | Track finder training script (frozen backbone) |
| `part/scripts/convert_lowpt_tau.sh` | Run the low-pT tau ROOT→Parquet conversion |
| `part/utils/convert_lowpt_tau_dataset.py` | Convert low-pT tau ROOT (NanoAOD) to Parquet |
| `part/utils/validate_parquet_quality.py` | Validate parquet files for simulation artefacts (NaN, unphysical values, corruption) |
| `part/condor/pretrain.sub` | HTCondor submit description for lxplus GPU training |
| `part/condor/run_pretrain.sh` | Worker node execution script (conda setup, /tmp I/O, EOS copy-back) |
| `part/utils/split_parquet.py` | Split a Parquet file into N parts for multi-worker DataLoader |
| `part/utils/convert_qg_datasets.py` | Convert QuarkGluon .npz to Parquet |
| `part/utils/convert_top_datasets.py` | Convert TopLandscape .h5 to Parquet |
| `part/get_datasets.py` | Download benchmark datasets from Zenodo |