# Prefilter confidence diagnostic — 2026-04-22

Source: `/Users/oleh/Projects/masters/experiments/results.parquet` joined with `/Users/oleh/Projects/masters/.worktrees/prefilter-perfect-recall/part/reports/perfect_recall_per_event.csv` (pass/fail labels). 52,284 events, 6,343 fails (baseline 0.1213).

All features are **inference-safe** — no GT labels used.

## 1. d' per confidence feature (pass vs fail)

| feature | mean (pass) | mean (fail) | median (pass) | median (fail) | d' | n (pass) | n (fail) |
|---|---:|---:|---:|---:|---:|---:|---:|
| part_prefilter_overlap_top50 | 0.5904 | 0.5244 | 0.6000 | 0.5200 | 0.678 | 45,941 | 6,343 |
| part_prefilter_overlap_top100 | 0.6639 | 0.6208 | 0.6600 | 0.6200 | 0.649 | 45,941 | 6,343 |
| part_top1_prefilter_rank | 7.8626 | 16.8230 | 3 | 5 | 0.361 | 45,941 | 6,343 |
| top_pt_median | 0.7719 | 0.7821 | 0.7676 | 0.7778 | 0.233 | 45,941 | 6,343 |
| top1_pt | 1.5391 | 1.3511 | 1.1963 | 1.0859 | 0.163 | 45,941 | 6,343 |
| n_top_retained | 255.7078 | 256 | 256 | 256 | 0.088 | 45,941 | 6,343 |
| top_pt_mean | 0.9946 | 1.0009 | 0.9853 | 0.9931 | 0.077 | 45,941 | 6,343 |
| top_pt_min | 0.5014 | 0.5015 | 0.5010 | 0.5010 | 0.037 | 45,941 | 6,343 |
| top_pt_std | 0.6807 | 0.6669 | 0.6231 | 0.6166 | 0.031 | 45,941 | 6,343 |
| top_pt_max | 6.0719 | 5.8639 | 5.0195 | 4.9102 | 0.029 | 45,941 | 6,343 |

## 2. Threshold sweeps for features with |d'| ≥ 0.3

### 2.part_prefilter_overlap_top50

| rule | n_hard | fraction_hard | fail_in_hard | fail_precision | fail_recall | P@K (hard) | P@K (easy) |
|---|---:|---:|---:|---:|---:|---:|---:|
| part_prefilter_overlap_top50<0.42 | 2,525 | 0.0483 | 647 | 0.2562 | 0.1020 | 0.7438 | 0.8855 |
| part_prefilter_overlap_top50<0.43999999999999995 | 3,821 | 0.0731 | 922 | 0.2413 | 0.1454 | 0.7587 | 0.8881 |
| part_prefilter_overlap_top50<0.48 | 7,613 | 0.1456 | 1,707 | 0.2242 | 0.2691 | 0.7758 | 0.8962 |
| part_prefilter_overlap_top50<0.5 | 10,130 | 0.1937 | 2,219 | 0.2191 | 0.3498 | 0.7809 | 0.9022 |
| part_prefilter_overlap_top50<0.5 | 10,130 | 0.1937 | 2,219 | 0.2191 | 0.3498 | 0.7809 | 0.9022 |
| part_prefilter_overlap_top50<0.52 | 13,101 | 0.2506 | 2,746 | 0.2096 | 0.4329 | 0.7904 | 0.9082 |
| part_prefilter_overlap_top50<0.56 | 20,092 | 0.3843 | 3,898 | 0.1940 | 0.6145 | 0.8060 | 0.9240 |
| part_prefilter_overlap_top50<0.58 | 23,971 | 0.4585 | 4,445 | 0.1854 | 0.7008 | 0.8146 | 0.9330 |

Best Fβ (β=1.5) threshold: `part_prefilter_overlap_top50<0.58` (Fβ = 0.3778).

### 2.part_prefilter_overlap_top100

| rule | n_hard | fraction_hard | fail_in_hard | fail_precision | fail_recall | P@K (hard) | P@K (easy) |
|---|---:|---:|---:|---:|---:|---:|---:|
| part_prefilter_overlap_top100<0.54 | 1,919 | 0.0367 | 461 | 0.2402 | 0.0727 | 0.7598 | 0.8832 |
| part_prefilter_overlap_top100<0.57 | 4,726 | 0.0904 | 1,079 | 0.2283 | 0.1701 | 0.7717 | 0.8893 |
| part_prefilter_overlap_top100<0.59 | 7,806 | 0.1493 | 1,686 | 0.2160 | 0.2658 | 0.7840 | 0.8953 |
| part_prefilter_overlap_top100<0.6 | 9,857 | 0.1885 | 2,097 | 0.2127 | 0.3306 | 0.7873 | 0.8999 |
| part_prefilter_overlap_top100<0.61 | 12,098 | 0.2314 | 2,519 | 0.2082 | 0.3971 | 0.7918 | 0.9048 |
| part_prefilter_overlap_top100<0.62 | 14,640 | 0.2800 | 2,975 | 0.2032 | 0.4690 | 0.7968 | 0.9105 |
| part_prefilter_overlap_top100<0.64 | 20,223 | 0.3868 | 3,775 | 0.1867 | 0.5951 | 0.8133 | 0.9199 |
| part_prefilter_overlap_top100<0.66 | 26,059 | 0.4984 | 4,597 | 0.1764 | 0.7247 | 0.8236 | 0.9334 |

Best Fβ (β=1.5) threshold: `part_prefilter_overlap_top100<0.66` (Fβ = 0.3704).

### 2.part_top1_prefilter_rank

| rule | n_hard | fraction_hard | fail_in_hard | fail_precision | fail_recall | P@K (hard) | P@K (easy) |
|---|---:|---:|---:|---:|---:|---:|---:|
| part_top1_prefilter_rank>37.0 | 2,557 | 0.0489 | 754 | 0.2949 | 0.1189 | 0.7051 | 0.8876 |
| part_top1_prefilter_rank>20.0 | 4,978 | 0.0952 | 1,340 | 0.2692 | 0.2113 | 0.7308 | 0.8942 |
| part_top1_prefilter_rank>12.0 | 7,842 | 0.1500 | 1,973 | 0.2516 | 0.3111 | 0.7484 | 0.9017 |
| part_top1_prefilter_rank>9.0 | 9,887 | 0.1891 | 2,328 | 0.2355 | 0.3670 | 0.7645 | 0.9053 |
| part_top1_prefilter_rank>7.0 | 12,100 | 0.2314 | 2,679 | 0.2214 | 0.4224 | 0.7786 | 0.9088 |
| part_top1_prefilter_rank>6.0 | 13,628 | 0.2607 | 2,865 | 0.2102 | 0.4517 | 0.7898 | 0.9100 |
| part_top1_prefilter_rank>4.0 | 19,097 | 0.3653 | 3,445 | 0.1804 | 0.5431 | 0.8196 | 0.9127 |
| part_top1_prefilter_rank>3.0 | 23,902 | 0.4572 | 3,871 | 0.1620 | 0.6103 | 0.8380 | 0.9129 |

Best Fβ (β=1.5) threshold: `part_top1_prefilter_rank>4.0` (Fβ = 0.3355).

## 3. Method

Features are computed directly from `results.parquet` (prefilter top-256 indices + pT, ParT top-100 indices — no scores are stored in the current eval artefact). Pass/fail labels join by composite event key from the per-event CSV. d' = |μ_pass − μ_fail| / √((σ_pass² + σ_fail²)/2). Threshold sweeps match the stratified-eval quantile scheme so they are directly comparable to earlier tables.
