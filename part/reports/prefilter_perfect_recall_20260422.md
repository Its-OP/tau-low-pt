# Prefilter perfect-recall diagnostic — 2026-04-22

Input: `/Users/oleh/Projects/masters/experiments/results.parquet` (joined with `/Users/oleh/Projects/masters/part/data/low-pt/val/val_00*.parquet` on composite key).
K = 256, kNN k = 16, events = 52,284 (pass = 45,941, fail = 6,343).

## 1. Overall P@K and R@K sweep

| K | P@K | R@K (per-pion) | # events pass | # events fail |
|---:|---:|---:|---:|---:|
| 128 | 0.7521 | 0.8677 | 39,323 | 12,961 |
| 200 | 0.8317 | 0.9174 | 43,486 | 8,798 |
| 256 | 0.8787 | 0.9432 | 45,941 | 6,343 |
| 300 | 0.8787 | 0.9432 | 45,941 | 6,343 |
| 400 | 0.8787 | 0.9432 | 45,941 | 6,343 |
| 500 | 0.8787 | 0.9432 | 45,941 | 6,343 |

## 2. Co-occurrence histogram (D3)

How many of each event's 3 GT pions land in top-256. Sum = 52,284.

| n_gt_in_top_K | # events | fraction |
|---:|---:|---:|
| 0 | 570 | 0.0109 |
| 1 | 1,420 | 0.0272 |
| 2 | 4,353 | 0.0833 |
| 3 | 45,941 | 0.8787 |

## 3. D1 — per-event: pass vs fail

Features computed over the full event. Sorted by |d'| descending.

| feature | mean (pass) | mean (fail) | median (pass) | median (fail) | d' | n (pass) | n (fail) |
|---|---:|---:|---:|---:|---:|---:|---:|
| gt_pt_sum | 4.5799 | 2.9442 | 3.8413 | 2.6885 | 0.779 | 45,941 | 6,343 |
| gt_pt_mean | 1.5266 | 0.9814 | 1.2804 | 0.8962 | 0.779 | 45,941 | 6,343 |
| gt_pt_max | 2.3529 | 1.4115 | 1.8750 | 1.2148 | 0.702 | 45,941 | 6,343 |
| n_tracks | 693.2754 | 830.4129 | 683.0000 | 822.0000 | 0.696 | 45,941 | 6,343 |
| gt_pt_spread | 1.4967 | 0.7669 | 1.0537 | 0.5693 | 0.602 | 45,941 | 6,343 |
| gt_pt_min | 0.8562 | 0.6446 | 0.7085 | 0.5928 | 0.599 | 45,941 | 6,343 |
| mean_chi2 | 0.7302 | 0.7466 | 0.7158 | 0.7350 | 0.132 | 45,941 | 6,343 |
| event_pt_median | 0.7896 | 0.7922 | 0.7893 | 0.7915 | 0.105 | 45,941 | 6,343 |
| mean_abs_dz_sig | 371.3211 | 380.5012 | 333.4611 | 340.8578 | 0.065 | 45,941 | 6,343 |
| event_pt_max | 16.1295 | 18.3967 | 7.8555 | 8.5625 | 0.063 | 45,941 | 6,343 |
| mean_abs_dxy_sig | 5.1198 | 5.1754 | 4.9183 | 5.0050 | 0.045 | 45,941 | 6,343 |
| event_pt_std | 0.9597 | 0.9865 | 0.6972 | 0.7032 | 0.022 | 45,941 | 6,343 |
| event_pt_p95 | 2.2271 | 2.2252 | 2.2168 | 2.2203 | 0.011 | 45,941 | 6,343 |
| vertex_z | 0.4389 | 0.4557 | 0.4440 | 0.4359 | 0.005 | 45,941 | 6,343 |

## 4. D2 / D4 — missed vs recalled pion inside failure events

Restricted to the 6,343 failure events. Three-way slice of the GT pions: `missed` = GT not in top-K; `recalled` = GT in top-K (the recalled siblings). Last row `gt_n_gt_neighbors` answers D4 (graph-noise hazard).

| feature | mean (missed) | mean (recalled) | median (missed) | median (recalled) | d' | n (missed) | n (recalled) |
|---|---:|---:|---:|---:|---:|---:|---:|
| pt_rank_in_event | 445.2332 | 395.8007 | 434.0000 | 365.0000 | 0.191 | 8,903 | 10,126 |
| pt_error | 0.0170 | 0.0126 | 0.0122 | 0.0099 | 0.111 | 8,903 | 10,126 |
| pt | 0.9488 | 1.0101 | 0.7568 | 0.8208 | 0.106 | 8,903 | 10,126 |
| gt_n_gt_neighbors | 0.5177 | 0.5794 | 0.0000 | 1.0000 | 0.098 | 8,903 | 10,126 |
| dca_significance | 4.7807 | 2.9809 | 0.6952 | 1.5369 | 0.046 | 8,903 | 10,126 |
| covariance_phi_phi | 0.0001 | 0.0000 | 0.0000 | 0.0000 | 0.038 | 8,903 | 10,126 |
| norm_chi2 | 0.7691 | 0.8293 | 0.0000 | 0.0000 | 0.017 | 8,903 | 10,126 |
| dz_significance | -1.4172 | -10.3853 | -0.2403 | 0.0649 | 0.014 | 8,903 | 10,126 |
| dxy_significance | 0.3804 | -0.0766 | -0.0000 | 0.0049 | 0.012 | 8,903 | 10,126 |
| n_pixel_hits | 3.7493 | 3.7397 | 4.0000 | 4.0000 | 0.009 | 8,903 | 10,126 |
| covariance_lambda_lambda | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.004 | 8,903 | 10,126 |

## 5. Method

Join: `(event_run, event_id, event_luminosity_block, source_batch_id, source_microbatch_id)` across `results.parquet` and `val_00*.parquet`. Per-pion features read from the val parquet using the GT indices already stored in `results.parquet`. (η, φ) kNN uses ΔR² proxy with φ wraparound; self is excluded. d-prime = |μ_a − μ_b| / √((σ_a² + σ_b²)/2).
