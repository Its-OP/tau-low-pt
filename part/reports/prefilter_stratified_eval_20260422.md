# Prefilter stratified eval — 2026-04-22

Source: `/Users/oleh/Projects/masters/.worktrees/prefilter-perfect-recall/part/reports/perfect_recall_per_event.csv` (52,284 events, 6,343 fails; baseline fail rate = 0.1213).
Scoring rule: Fβ with β = 1.5 (recall-weighted).

## 1. Per-feature threshold sweeps

### 1.gt_pt_sum  —  *Low total GT pT → event is soft*

Direction: `gt_pt_sum lt`.

| rule | n_hard | fraction_hard | fail_in_hard | fail_precision | fail_recall | P@K (hard) | P@K (easy) |
|---|---:|---:|---:|---:|---:|---:|---:|
| gt_pt_sum<2.015625 | 2,611 | 0.0499 | 872 | 0.3340 | 0.1375 | 0.6660 | 0.8899 |
| gt_pt_sum<2.228515625 | 5,222 | 0.0999 | 1,608 | 0.3079 | 0.2535 | 0.6921 | 0.8994 |
| gt_pt_sum<2.40966796875 | 7,842 | 0.1500 | 2,239 | 0.2855 | 0.3530 | 0.7145 | 0.9077 |
| gt_pt_sum<2.58134765625 | 10,457 | 0.2000 | 2,853 | 0.2728 | 0.4498 | 0.7272 | 0.9166 |
| gt_pt_sum<2.7440185546875 | 13,071 | 0.2500 | 3,345 | 0.2559 | 0.5274 | 0.7441 | 0.9235 |
| gt_pt_sum<2.907666015625 | 15,685 | 0.3000 | 3,826 | 0.2439 | 0.6032 | 0.7561 | 0.9312 |
| gt_pt_sum<3.2608398437500004 | 20,914 | 0.4000 | 4,584 | 0.2192 | 0.7227 | 0.7808 | 0.9439 |
| gt_pt_sum<3.6611328125 | 26,140 | 0.5000 | 5,173 | 0.1979 | 0.8155 | 0.8021 | 0.9552 |

### 1.gt_pt_mean  —  *Low mean GT pT*

Direction: `gt_pt_mean lt`.

| rule | n_hard | fraction_hard | fail_in_hard | fail_precision | fail_recall | P@K (hard) | P@K (easy) |
|---|---:|---:|---:|---:|---:|---:|---:|
| gt_pt_mean<0.671875 | 2,611 | 0.0499 | 872 | 0.3340 | 0.1375 | 0.6660 | 0.8899 |
| gt_pt_mean<0.7428385416666665 | 5,222 | 0.0999 | 1,608 | 0.3079 | 0.2535 | 0.6921 | 0.8994 |
| gt_pt_mean<0.80322265625 | 7,842 | 0.1500 | 2,239 | 0.2855 | 0.3530 | 0.7145 | 0.9077 |
| gt_pt_mean<0.86044921875 | 10,457 | 0.2000 | 2,853 | 0.2728 | 0.4498 | 0.7272 | 0.9166 |
| gt_pt_mean<0.9146728515625 | 13,071 | 0.2500 | 3,345 | 0.2559 | 0.5274 | 0.7441 | 0.9235 |
| gt_pt_mean<0.9692220052083332 | 15,685 | 0.3000 | 3,826 | 0.2439 | 0.6032 | 0.7561 | 0.9312 |
| gt_pt_mean<1.0869466145833335 | 20,914 | 0.4000 | 4,584 | 0.2192 | 0.7227 | 0.7808 | 0.9439 |
| gt_pt_mean<1.2203776041666667 | 26,140 | 0.5000 | 5,173 | 0.1979 | 0.8155 | 0.8021 | 0.9552 |

### 1.gt_pt_max  —  *Low max GT pT*

Direction: `gt_pt_max lt`.

| rule | n_hard | fraction_hard | fail_in_hard | fail_precision | fail_recall | P@K (hard) | P@K (easy) |
|---|---:|---:|---:|---:|---:|---:|---:|
| gt_pt_max<0.78857421875 | 2,608 | 0.0499 | 839 | 0.3217 | 0.1323 | 0.6783 | 0.8892 |
| gt_pt_max<0.91455078125 | 5,219 | 0.0998 | 1,554 | 0.2978 | 0.2450 | 0.7022 | 0.8982 |
| gt_pt_max<1.0244140625 | 7,824 | 0.1496 | 2,142 | 0.2738 | 0.3377 | 0.7262 | 0.9055 |
| gt_pt_max<1.12109375 | 10,451 | 0.1999 | 2,723 | 0.2605 | 0.4293 | 0.7395 | 0.9135 |
| gt_pt_max<1.2197265625 | 13,048 | 0.2496 | 3,191 | 0.2446 | 0.5031 | 0.7554 | 0.9197 |
| gt_pt_max<1.3203125 | 15,675 | 0.2998 | 3,660 | 0.2335 | 0.5770 | 0.7665 | 0.9267 |
| gt_pt_max<1.53125 | 20,893 | 0.3996 | 4,387 | 0.2100 | 0.6916 | 0.7900 | 0.9377 |
| gt_pt_max<1.7724609375 | 26,141 | 0.5000 | 4,991 | 0.1909 | 0.7869 | 0.8091 | 0.9483 |

### 1.gt_pt_min  —  *Low softest-GT pT*

Direction: `gt_pt_min lt`.

| rule | n_hard | fraction_hard | fail_in_hard | fail_precision | fail_recall | P@K (hard) | P@K (easy) |
|---|---:|---:|---:|---:|---:|---:|---:|
| gt_pt_min<0.5126953125 | 2,598 | 0.0497 | 590 | 0.2271 | 0.0930 | 0.7729 | 0.8842 |
| gt_pt_min<0.52685546875 | 5,217 | 0.0998 | 1,137 | 0.2179 | 0.1793 | 0.7821 | 0.8894 |
| gt_pt_min<0.54150390625 | 7,840 | 0.1500 | 1,668 | 0.2128 | 0.2630 | 0.7872 | 0.8948 |
| gt_pt_min<0.55712890625 | 10,421 | 0.1993 | 2,176 | 0.2088 | 0.3431 | 0.7912 | 0.9005 |
| gt_pt_min<0.57470703125 | 13,067 | 0.2499 | 2,681 | 0.2052 | 0.4227 | 0.7948 | 0.9066 |
| gt_pt_min<0.59228515625 | 15,639 | 0.2991 | 3,152 | 0.2015 | 0.4969 | 0.7985 | 0.9129 |
| gt_pt_min<0.634765625 | 20,884 | 0.3994 | 4,040 | 0.1934 | 0.6369 | 0.8066 | 0.9267 |
| gt_pt_min<0.6875 | 26,137 | 0.4999 | 4,766 | 0.1823 | 0.7514 | 0.8177 | 0.9397 |

### 1.n_tracks  —  *High track count → crowded event*

Direction: `n_tracks gt`.

| rule | n_hard | fraction_hard | fail_in_hard | fail_precision | fail_recall | P@K (hard) | P@K (easy) |
|---|---:|---:|---:|---:|---:|---:|---:|
| n_tracks>1066.0 | 2,598 | 0.0497 | 672 | 0.2587 | 0.1059 | 0.7413 | 0.8859 |
| n_tracks>981.0 | 5,206 | 0.0996 | 1,284 | 0.2466 | 0.2024 | 0.7534 | 0.8925 |
| n_tracks>926.0 | 7,811 | 0.1494 | 1,862 | 0.2384 | 0.2936 | 0.7616 | 0.8992 |
| n_tracks>883.0 | 10,452 | 0.1999 | 2,358 | 0.2256 | 0.3717 | 0.7744 | 0.9047 |
| n_tracks>846.0 | 13,006 | 0.2488 | 2,829 | 0.2175 | 0.4460 | 0.7825 | 0.9105 |
| n_tracks>813.0 | 15,652 | 0.2994 | 3,280 | 0.2096 | 0.5171 | 0.7904 | 0.9164 |
| n_tracks>755.0 | 20,897 | 0.3997 | 4,047 | 0.1937 | 0.6380 | 0.8063 | 0.9268 |
| n_tracks>701.0 | 26,130 | 0.4998 | 4,722 | 0.1807 | 0.7444 | 0.8193 | 0.9380 |

## 2. Best standalone thresholds (ranked by Fβ)

| rule | n_hard | fraction_hard | fail_in_hard | fail_precision | fail_recall | P@K (hard) | P@K (easy) |
|---|---:|---:|---:|---:|---:|---:|---:|
| gt_pt_sum<3.2608398437500004 | 20,914 | 0.4000 | 4,584 | 0.2192 | 0.7227 | 0.7808 | 0.9439 |
| gt_pt_mean<1.0869466145833335 | 20,914 | 0.4000 | 4,584 | 0.2192 | 0.7227 | 0.7808 | 0.9439 |
| gt_pt_max<1.53125 | 20,893 | 0.3996 | 4,387 | 0.2100 | 0.6916 | 0.7900 | 0.9377 |
| gt_pt_min<0.6875 | 26,137 | 0.4999 | 4,766 | 0.1823 | 0.7514 | 0.8177 | 0.9397 |
| n_tracks>701.0 | 26,130 | 0.4998 | 4,722 | 0.1807 | 0.7444 | 0.8193 | 0.9380 |

F_β scores (recall-weighted, β=1.5):

- `gt_pt_sum<3.2608398437500004` → Fβ = 0.4234
- `gt_pt_mean<1.0869466145833335` → Fβ = 0.4234
- `gt_pt_max<1.53125` → Fβ = 0.4055
- `gt_pt_min<0.6875` → Fβ = 0.3833
- `n_tracks>701.0` → Fβ = 0.3798

## 3. Combined rules (top-2 features)

| rule | n_hard | fraction_hard | fail_in_hard | fail_precision | fail_recall | P@K (hard) | P@K (easy) |
|---|---:|---:|---:|---:|---:|---:|---:|
| gt_pt_sum<3.2608398437500004 AND n_tracks>701.0 | 10,467 | 0.2002 | 3,377 | 0.3226 | 0.5324 | 0.6774 | 0.9291 |
| gt_pt_sum<3.2608398437500004 OR n_tracks>701.0 | 36,577 | 0.6996 | 5,929 | 0.1621 | 0.9347 | 0.8379 | 0.9736 |

## 4. Method

Predicate-based partition of val events into `hard` (satisfies rule) and `easy` (does not). P@K on each partition uses the `perfect_recall` flag from the source diagnostic — pass iff all GT pions were in top-K at the K that diagnostic used. `fail_precision` is failure rate inside `hard`; `fail_recall` is the fraction of all failures captured by the rule. Thresholds are sampled at quantiles of the global feature distribution (not the failure subset) so the rule can be ported to train-time sampling without re-calibration.
