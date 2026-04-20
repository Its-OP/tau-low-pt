---
name: setup_server
description: Bootstrap the tau-low-pt monorepo on a bare vast.ai GPU server via SSH. Uploads setup_server.sh, runs it (clones repo, downloads dataset from GDrive, installs miniconda + torch + deps), then scp's the gitignored cascade_best.pt (269 MB). Use when the user asks to "setup the server", "/setup-server", or references deploying to a newly provisioned remote box. Requires the user to have supplied an SSH connection string earlier in the conversation.
---

# Setup Server

End-to-end bootstrap of a bare vast.ai Linux box so the tau-low-pt monorepo is ready for training.

## Preconditions

1. **SSH connection must already be known.** Look for an `ssh -p PORT root@IP [-L 8080:localhost:8080]` string earlier in the conversation, or check `~/.claude/projects/-Users-oleh-Projects-masters/memory/reference_server_ssh.md`. If none is present or it is stale, ask the user for it before proceeding. Do NOT invent a host.
2. **Local artifacts that must exist:**
   - `/Users/oleh/Projects/masters/setup_server.sh` (bootstrap script; ~7 KB)
   - `/Users/oleh/Projects/masters/part/models/cascade_best.pt` (269 MB; gitignored — must be scp'd separately)

## Steps

### 1. Sanity-check the remote
```bash
ssh -o StrictHostKeyChecking=no -p PORT root@IP "echo ALIVE && ls -la /workspace/ && nvidia-smi | head -10"
```
Confirm: box responds, `/workspace` exists, GPU visible. Abort if any fail.

### 2. Clean prior partial state
Vast.ai volumes are often reused. A prior run may have left a partial `dataset.zip`, orphan processes, or incomplete extractions. Kill procs and clean before starting:
```bash
ssh -p PORT root@IP "pkill -9 -f setup_server.sh 2>/dev/null; pkill -9 gdown 2>/dev/null; pkill -9 unzip 2>/dev/null; sleep 2; \
  rm -f /workspace/tau-low-pt/part/data/low-pt/dataset.zip*.part; \
  ls /workspace/tau-low-pt/part/data/low-pt/ 2>/dev/null"
```
If `/workspace/tau-low-pt/part/data/low-pt/dataset.zip` exists and `unzip -tq` passes, keep it — skip the GDrive download by extracting manually.

### 3. Upload the bootstrap script
```bash
scp -P PORT /Users/oleh/Projects/masters/setup_server.sh root@IP:/workspace/setup_server.sh
```

### 4. Run setup in the background
`nohup` so the run survives if the SSH session drops. Tee output to `/workspace/setup.log`.
```bash
ssh -p PORT root@IP "cd /workspace && chmod +x setup_server.sh && \
  nohup bash -c 'bash setup_server.sh > /workspace/setup.log 2>&1' > /dev/null 2>&1 & echo PID=\$!"
```
Use `run_in_background: true` on the Bash call. Total ~5–10 min: dataset download ~1 min, conda install ~2 min, torch + pip ~3 min, unzip if needed ~2 min.

### 5. Monitor progress
Use `ScheduleWakeup` with `delaySeconds` ~180–240 to re-enter and check. Each wakeup:
```bash
ssh -p PORT root@IP "ps auxww | grep -E 'setup_server|pip|gdown|unzip' | grep -v grep || echo DONE; tail -10 /workspace/setup.log"
```
Look for `Setup complete!` as the terminal line.

### 6. SCP the model checkpoint
`cascade_best.pt` exceeds GitHub's 100 MB hard limit and is in `.gitignore`. Copy once `part/models/` exists (available shortly after clone, before full setup finishes):
```bash
scp -P PORT /Users/oleh/Projects/masters/part/models/cascade_best.pt \
  root@IP:/workspace/tau-low-pt/part/models/cascade_best.pt
```
Can run in parallel with the main setup.

### 7. Verify
```bash
ssh -p PORT root@IP "source /root/miniconda3/etc/profile.d/conda.sh && conda activate part && \
  cd /workspace/tau-low-pt/part && \
  python -c 'import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))' && \
  ls -la models/cascade_best.pt && \
  python -m pytest tests/ -q 2>&1 | tail -5"
```
Expect: torch ≥2.11, `cuda=True`, GPU name printed, cascade_best.pt 269 MB, all tests pass.

## Known issues (already patched in `setup_server.sh`)

| Symptom | Cause | Fix |
|---|---|---|
| `CondaToSNonInteractiveError` | conda ≥25.x requires explicit ToS accept for `pkgs/main` and `pkgs/r` | Script auto-runs `conda tos accept --override-channels --channel ...` |
| Re-downloads dataset every run | Old script checked `VAL_COUNT -lt 10`; val set has exactly 7 files | Threshold is now `-lt 7` |
| `No module named pytest` during verify | Previous script did not install pytest | Added `pytest pytest-cov` to pip install list |

If `setup_server.sh` gets reverted, re-apply those three edits before uploading.

## Vast.ai quirks

- The `part` conda env may resolve to a pre-existing vast.ai venv at `/venv/part/bin/python` rather than `/root/miniconda3/envs/part/`. That is fine — torch + deps install into whichever path is active.
- Root overlay is often only ~24 GB. Training with tensorboard + multiple checkpoints can exhaust it fast. Warn the user if `df -h /workspace` shows <10 GB free after setup.

## Task tracking

For this multi-step flow, create TaskCreate entries up-front:
1. Upload `setup_server.sh`
2. Run `setup_server.sh` (background)
3. SCP `cascade_best.pt`
4. Verify (env + CUDA + tests)

Mark completed as each finishes so the user gets a live progress view.
