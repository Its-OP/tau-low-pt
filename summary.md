# Session Summary — CoupleReranker Eval Debugging

**Date:** 2026-04-16
**Branch:** main (working dir has uncommitted changes in `part/`, `setup_server.sh`, `weaver/`)
**Status:** Eval completed on full validation set; metrics are ~24pp below expected. Root cause hypothesis identified but not yet applied.

---

## 1. Primary Objective

Debug why the 3-stage cascade (prefilter + ParT reranker + couple reranker) produces eval metrics far below training metrics. Before the debug, several plumbing tasks were completed to make the eval output usable for downstream analysis at CERN.

---

## 2. Completed Work

### 2.1 Composite key validation
- Analyzed `part/scripts/validate_composite_key.py` — streams ROOT files in 50K-entry chunks via `uproot.iterate()` and performs 6 uniqueness checks.
- Ran against all 10 merged ROOT files (2,311,077 entries, 708 s runtime). **All checks passed.**
  - Schema contains the 5 composite-key fields
  - `source_batch_id` consistent within each merged file
  - No `source_microbatch_id == -1` leaks
  - Unique within every file
  - Unique across all files
  - 1,019 unique CMS `(run, event, lumi)` keys, each appearing 2,119–2,410× (MC generation job numbering reuses CMS events across batches — expected, not a bug).
- Analyzed a single parquet shard (`val_000.parquet`, 7,500 rows, all from batch 9, 443 microbatches, 100% events have exactly 3 GT pions).
- Analyzed the full train/val parquet dataset:
  - TRAIN: 10 files, 300,000 rows, batches 1–9
  - VAL: 7 files, 52,500 rows, batches 9–10
  - **Zero composite-key overlap between train and val** (no leakage).

**Conclusion:** the composite key `(event_run, event_id, event_luminosity_block, source_batch_id, source_microbatch_id)` is unique at every level and safe to use as a stable identifier for downstream joins.

### 2.2 YAML config — carry composite key through the DataLoader
Modified `part/data/low-pt/lowpt_tau_trackfinder.yaml` to add observers so the full composite key flows through weaver:

```yaml
observers:
   - event_n_tracks
   - event_run
   - event_id
   - event_luminosity_block
   - source_batch_id
   - source_microbatch_id
```

### 2.3 Eval script — write full composite key to parquet
Modified `part/diagnostics/eval_couple_reranker.py`:

- `main()` now reads all 5 composite-key fields from the DataLoader's `Z` observer dict per event:
```python
result['event_run'] = int(Z['event_run'][event_index])
result['event_id'] = int(Z['event_id'][event_index])
result['event_luminosity_block'] = int(Z['event_luminosity_block'][event_index])
result['source_batch_id'] = int(Z['source_batch_id'][event_index])
result['source_microbatch_id'] = int(Z['source_microbatch_id'][event_index])
```

- `write_results_parquet()` now includes `source_batch_id` and `source_microbatch_id` columns as `pa.int32()`.

### 2.4 Tests updated
Modified `part/tests/test_eval_couple_reranker.py`:
- Added `source_batch_id` / `source_microbatch_id` to the synthetic test rows.
- Updated the `expected_columns` set to include the two new fields.

All 444 tests pass.

### 2.5 auto.yaml regeneration
Regenerated the cached standardization params file twice after the YAML observer change invalidated its hash:
- First from `data/low-pt/subset/train/*.parquet` (subset, quick verification)
- Then from `data/low-pt/train/*.parquet` (full train, 300,000 events)

**Both auto.yaml variants produced identical eval metrics** → auto.yaml / standardization is NOT the cause of the low metrics.

Final file: `part/data/low-pt/lowpt_tau_trackfinder.c8a40f560c44edfe47c8f0fc25230de1.auto.yaml`

### 2.6 Full-validation-set eval run
- Ran `diagnostics/eval_couple_reranker.py` on all 52,500 val events with `num_workers=0` (MPS, ~42 min, ~23 events/s).
- Output parquet: `data/low-pt/eval/couple_reranker_val.parquet`
  - 52,500 rows
  - Always hits the K=200 couple cap (200 couples / event)
  - Mean 255.7 remaining pions per event

---

## 3. Current Results (unexpected)

```
Couple Reranker Metrics — 52,500 events
+-----+--------+--------+
|  K  |  C@K   |  RC@K  |
+-----+--------+--------+
|  1  | 0.2178 | 0.2005 |
|  50 | 0.4913 | 0.4244 |
| 100 | 0.6012 | 0.5212 |
| 200 | 0.6617 | 0.5771 |
+-----+--------+--------+
eligible:      47,758 / 52,500  (91.0%)
full_triplet:  39,034           (74.3%)
mean_rank:     29.6
```

**Expected (per MEMORY.md → `project_couple_reranker_sweep.md`):** K=125 gave C@100 = 0.8392.
**Gap:** about 24 percentage points.

---

## 4. Ruled Out as Causes

| Hypothesis | Status |
|---|---|
| Composite key corruption / event duplication | Ruled out — unique at all levels |
| Train/val leakage | Ruled out — zero overlap |
| Dataset mismatch (val not low-pT cutoff) | Ruled out — user confirmed val IS low cutoff |
| Stale / subset auto.yaml | Ruled out — full-train auto.yaml gave identical results |

---

## 5. Leading Hypothesis — CoupleReranker NaN BatchNorm

Per the loaded plan at `/Users/oleh/.claude/plans/groovy-riding-aho.md`:

> The `CoupleReranker` (Stage 3) has **all 10 BatchNorm running_mean/running_var set to NaN** in saved checkpoints. Training achieved C@100=0.84 because it always ran in `train()` mode (batch stats), so the corrupted running stats were never used. But `eval()` mode — needed for inference — uses running stats and produces garbage.
>
> **Root cause**: In `train_couple_reranker.py`, line 153 calls `model.train()`. Forward pass updates BN running stats via EMA. If any batch produces NaN activations in couple features, `running_mean = (1-m)*NaN + m*batch_mean = NaN` — permanently corrupted. The non-finite loss check at line 192 skips the gradient step but BN damage is already done.

**Plan item #7 — immediate eval fix:**
> The eval script already puts `model.cascade.train()`. Also add `model.couple_reranker.train()` so the reranker uses batch stats (matching training conditions). This is the immediate workaround until clean checkpoints are produced by the retrained sweep.

Our eval script currently calls `model.cascade.train()` but leaves `model.couple_reranker` in eval mode → corrupted running stats are used → garbage output. This matches the ~24pp drop.

---

## 6. Next Actions (for the debugging session)

1. **Verify the hypothesis:** load `models/couple_reranker_best.pt` and inspect all BN layer `running_mean` / `running_var` for NaN. If NaN → hypothesis confirmed.
2. **Apply quick fix** (plan item #7): add `model.couple_reranker.train()` to `diagnostics/eval_couple_reranker.py` alongside the existing `model.cascade.train()` call.
3. **Re-run full eval on 52,500 val events** (another ~42 min) and check C@100 climbs to ~0.83.
4. **(Later) apply the full plan:**
   - `NanSafeBatchNorm1d` in `weaver/weaver/nn/model/CoupleReranker.py`
   - `calibrate_reranker_bn()` in `train_couple_reranker.py`
   - Non-finite batch counter in `train_one_epoch()`
   - Sweep settings: `KEEP_BEST_K=5`, `EPOCHS=80`, adaptive batch size for large K
   - BN health check before saving checkpoints

**Ask user before re-running full eval** — the first run took 42 min, so confirm before spending that again.

---

## 7. Key Files Touched This Session

| File | Change |
|---|---|
| `part/data/low-pt/lowpt_tau_trackfinder.yaml` | Added 5 composite-key observers |
| `part/diagnostics/eval_couple_reranker.py` | Pass composite key from `Z` dict into output parquet |
| `part/tests/test_eval_couple_reranker.py` | New composite-key columns in fixtures + expected set |
| `part/data/low-pt/lowpt_tau_trackfinder.c8a40f560c44edfe47c8f0fc25230de1.auto.yaml` | Regenerated (full train) |

## 8. Key Reference Paths

- Plan: `/Users/oleh/.claude/plans/groovy-riding-aho.md`
- Eval output: `part/data/low-pt/eval/couple_reranker_val.parquet`
- Checkpoint: `part/models/couple_reranker_best.pt`
- Previous transcript (if needed): `/Users/oleh/.claude/projects/-Users-oleh-Projects-masters/8f1dc8c4-4757-4473-b4fd-813491816a46.jsonl`

## 9. Mistakes and Course Corrections

- **Smoke-test mistake:** I initially ran the eval with `--max-events 1500` thinking the user wanted a cap. The user corrected: "I LITERALLY ASKED FOR A FULL RUN. NOT A SMOKE TEST, THE FULL RUN." Restarted with `--max-events 52500`.
- **auto.yaml source:** Regenerated from subset first; user asked what data I used, so I regenerated from the full train set. Both produced identical results → not the bug.

## 10. Constraints to Honor (from MEMORY.md / CLAUDE.md)

- Use `num_workers=0` locally on MPS (worker crashes otherwise).
- TDD mandatory: tests first, then implementation; full suite (`python -m pytest tests/` from `part/`) before claiming done.
- Always smoke test after changes.
- Use direct Python path: `/opt/miniconda3/envs/part/bin/python`.
- Never use `eval` (Python builtin — strictly forbidden).
- No fake data for tests — use real subset data.
- Edit tool (not `sed`/`cat` append) for file edits.
