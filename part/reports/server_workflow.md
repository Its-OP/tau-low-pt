# Server Workflow — vast.ai GPU

Operational reference for running scripts on the vast.ai GPU server from a Claude Code session. Captures conventions settled during the 2026-04-16 monorepo migration + couple-reranker eval debugging session. See also memory files `reference_server_ssh.md` (credentials, host layout) and `reference_repo_structure.md` (monorepo conventions).

## 1. Connection

```bash
# Verify current port/IP in reference_server_ssh.md first — vast.ai reassigns them per session.
ssh -p 26893 root@69.63.236.192 -L 8080:localhost:8080
```

Conda env activation (required for every remote command that imports torch/weaver):

```bash
source /root/miniconda3/etc/profile.d/conda.sh && conda activate part
```

Resolves to `/venv/part/bin/python` (Python 3.13.13, torch 2.11.0+cu130, CUDA 13.0 / RTX PRO 6000 96 GB).

## 2. Paths on server

| Path | Purpose |
|------|---------|
| `/workspace/tau-low-pt/` | Monorepo root (flat, post-migration). Contains `part/` and `weaver/` as plain subdirs. |
| `/workspace/tau-low-pt/part/models/` | All checkpoints (`prefilter_best.pt`, `couple_reranker_best.pt`, `cascade_best.pt`). |
| `/workspace/tau-low-pt/part/data/low-pt/{train,val}/` | Parquet shards (10 train, 7 val). |
| `/workspace/*.log` | All training/eval logs. Keep logs OUT of the repo tree. |
| `/workspace/setup_server.sh` | Bootstrap (clone + dataset + miniconda + deps, ~15 min). Mirror of repo-root copy. |

`cascade_best.pt` (269 MB) is gitignored (>100 MB GitHub limit). On a fresh box, `scp` it explicitly:

```bash
scp -P 26893 /Users/oleh/Projects/masters/part/models/cascade_best.pt \
    root@69.63.236.192:/workspace/tau-low-pt/part/models/cascade_best.pt
```

## 3. Running long jobs — nohup + disown

NEVER run a long job as a plain `ssh ... "cmd"`. The process dies when ssh disconnects.

Canonical launch pattern (pair with `run_in_background: true` so Bash returns immediately):

```bash
ssh -p 26893 root@69.63.236.192 "
  source /root/miniconda3/etc/profile.d/conda.sh && conda activate part &&
  cd /workspace/tau-low-pt/part &&
  nohup python eval_cascade.py --batch-size 256 --data-config ... \
    > /workspace/eval.log 2>&1 &
  echo PID=\$!
  disown
"
```

Notes:
- Escape `\$!` — the outer shell is local, we want the remote PID.
- `run_in_background: true` (on the Bash tool call) detaches the ssh invocation itself so the harness isn't blocked.
- Log to `/workspace/*.log`, never inside the repo (keeps `git status` clean, avoids accidentally committing).

## 4. Monitoring

| Task | Command |
|------|---------|
| Tail progress (tqdm safe — `\r` → `\n`) | `ssh ... "tail -200 /workspace/eval.log \| tr '\r' '\n' \| tail -40"` |
| Filter a fat log | `ssh ... "grep -E 'Epoch\|loss\|accuracy' /workspace/eval.log \| tail -50"` |
| GPU utilisation | `ssh ... "nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader"` |
| Process still alive | `ssh ... "pgrep -af eval_cascade.py"` |
| Output file appeared | `ssh ... "ls -la /workspace/tau-low-pt/part/reports/*.parquet 2>/dev/null"` |

### Wait-for-file loop (preferred over `sleep`)

Use the Monitor tool with an `until` on the file check. Example body:

```bash
until ssh -p 26893 root@69.63.236.192 \
  "test -f /workspace/tau-low-pt/part/reports/eval_cascade_gt_pions.parquet"; do
  sleep 30
done
```

The harness pings once per iteration and notifies on exit; no manual polling loop.

## 5. SSH exit-code gotcha

A piped remote command like `ssh ... "pgrep -f foo | grep bar"` can return non-zero (often 255 via ssh's exit remapping) when `grep` has no match, even though nothing actually failed. Treat a non-zero ssh exit as **advisory**: verify the intended side-effect with a separate check.

```bash
# Don't trust this alone:
ssh ... "pgrep -f eval_cascade | xargs -r kill"
# Follow up with:
ssh ... "pgrep -af eval_cascade || echo 'no process'"
```

## 6. Missing `.auto.yaml` on fresh server

Weaver computes per-field standardization params and caches them next to the data config as `<config>.<hash>.auto.yaml`. On `for_training=False` runs (eval/inference), weaver refuses to regenerate and dies with:

```
ValueError: No valid standardization params for <field>
```

**Fix (sticky):** copy the per-experiment auto-yaml from the training run's checkpoint dir and commit it to `main` so subsequent `git pull`s on fresh boxes pick it up:

```bash
# Local: find the right one (hash must match the active data config)
ls part/models/debug_checkpoints/<run>/lowpt_tau_trackfinder.*.auto.yaml

# Commit + push
git add part/data/configs/lowpt_tau_trackfinder.<hash>.auto.yaml
git commit -m "add standardization auto-yaml for <run>"
git push origin main

# Server
cd /workspace/tau-low-pt && git pull origin main
```

## 7. Batch size — don't leave the default

Eval scripts default to BS=64. On the RTX PRO 6000 (96 GB VRAM) we measured **305 events/s at BS=256** with plenty of headroom. Always pass `--batch-size 256` (or larger if you verify headroom). BS=64 wastes ~4x wall time.

## 8. One-shot remote edits

Upload a file via heredoc rather than scp'ing for tiny patches:

```bash
ssh -p 26893 root@69.63.236.192 "cat > /workspace/tau-low-pt/part/foo.py" << 'EOF'
print("hello")
EOF
```

## 9. Checklist before declaring a run started

1. `nvidia-smi` shows the expected GPU.
2. `which python` → `/venv/part/bin/python`.
3. `pgrep -af <script>` returns a PID.
4. First `tail` of the log shows model loaded, dataset opened, epoch 0 iterating.
5. Repo checkout is clean (`git status` on `/workspace/tau-low-pt`) — no accidental local edits that will vanish on the next `git pull`.

## 10. Common mistakes recap

| Mistake | Symptom | Fix |
|---------|---------|-----|
| `ssh ... "cmd"` without nohup | Job dies mid-run | Use the launch snippet in §3. |
| Log inside repo | `git status` polluted | Log to `/workspace/*.log`. |
| Cat-ing a tqdm log | Walls of `\rXX%` junk | `tr '\r' '\n'` before `tail`. |
| BS=64 default | 4x slow | Pass `--batch-size 256`. |
| Missing `.auto.yaml` | Weaver ValueError | Commit it to main (§6). |
| Trusting ssh exit code | Spurious failure reports | Verify with a second `pgrep`/`ls` (§5). |
| Forgetting `cascade_best.pt` | FileNotFoundError on load | `scp` it (§2). |
| `git submodule` commands | No-op / confusion | The repo is flat (see `reference_repo_structure.md`). |
