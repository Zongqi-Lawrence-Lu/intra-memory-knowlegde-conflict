# gpt2-small-openwebtext-T1280: intervention baseline (top-7 relations only, checkpoint step 61035)

Retained: alma_mater, birthplace, current_residence, employer_role, field_expertise, license_certification, mentor.
Dropped (unreliable storage, excluded from the intervention population): affiliation, authored_work, award_honor, civic_role, funding_source, publication_venue, working_language.

## Table 1: background (uncontested) top1/top5 accuracy, top-7 only

| relation | top1 % | top5 % | n |
|---|---|---|---|
| alma_mater | 65.1 | 87.8 | 2400 |
| birthplace | 91.4 | 98.8 | 2400 |
| current_residence | 88.7 | 97.3 | 2400 |
| employer_role | 63.7 | 76.7 | 2400 |
| field_expertise | 70.6 | 98.2 | 2400 |
| license_certification | 22.5 | 30.5 | 2400 |
| mentor | 28.7 | 57.8 | 2400 |
| **POOLED** | **61.5** | **78.2** | 16800 |

## Table 4: contested-pair relative rate (forced A-vs-B logit comparison), top-7 only

| relation | (640, 640) | (800, 480) | (960, 320) | (1088, 192) | (1168, 112) | (1232, 48) |
|---|---|---|---|---|---|---|
| alma_mater | (66.0%,34.0%) | (61.0%,39.0%) | (52.0%,48.0%) | (61.0%,39.0%) | (78.0%,22.0%) | (91.0%,9.0%) |
| birthplace | (55.0%,45.0%) | (77.0%,23.0%) | (85.0%,15.0%) | (88.0%,12.0%) | (91.0%,9.0%) | (100.0%,0.0%) |
| current_residence | (56.0%,44.0%) | (51.0%,49.0%) | (65.0%,35.0%) | (74.0%,26.0%) | (99.0%,1.0%) | (100.0%,0.0%) |
| employer_role | (48.0%,52.0%) | (44.0%,56.0%) | (70.0%,30.0%) | (68.0%,32.0%) | (83.0%,17.0%) | (90.0%,10.0%) |
| field_expertise | (47.0%,53.0%) | (62.0%,38.0%) | (80.0%,20.0%) | (82.0%,18.0%) | (99.0%,1.0%) | (100.0%,0.0%) |
| license_certification | (55.0%,45.0%) | (51.0%,49.0%) | (70.0%,30.0%) | (62.0%,38.0%) | (61.0%,39.0%) | (84.0%,16.0%) |
| mentor | (51.0%,49.0%) | (55.0%,45.0%) | (35.0%,65.0%) | (62.0%,38.0%) | (82.0%,18.0%) | (87.0%,13.0%) |
| **POOLED** | (54.0%,46.0%) | (57.3%,42.7%) | (65.3%,34.7%) | (71.0%,29.0%) | (84.7%,15.3%) | (93.1%,6.9%) |

## Calibration-target summary (contested, top-7 pooled)

- n_entities_scored: 839
- n_entities_skipped_divergence_failure: 1
- monotonicity_violations: 0
- symmetry_at_balance_mean_logit_gap: 0.22340940424716837

## Cross-entropy-to-proportional-target (candidate metric, experimental_plans.tex Sec.xent-metric -- one of several under consideration, not settled)

- overall_mean_cross_entropy_to_proportional_target (nats): 0.8746
- overall_mean_kl_to_proportional_target (nats): 0.4088

By split level (freq_gap, accuracy, mean logit gap, mean confidence in higher-freq side, mean cross-entropy, mean KL -- both to the proportional target):

| n_a | n_b | freq_gap | n_entities | accuracy | mean_logit_gap_a_minus_b | mean_confidence_higher_freq_side | mean_cross_entropy | mean_kl |
|---|---|---|---|---|---|---|---|---|
| 640 | 640 | 0 | 139 | 54.0% | 0.223 | 53.4% | 1.2553 | 0.5622 |
| 800 | 480 | 320 | 140 | 60.7% | 0.515 | 56.6% | 1.2040 | 0.5424 |
| 960 | 320 | 640 | 140 | 69.3% | 0.980 | 64.4% | 1.1153 | 0.5530 |
| 1088 | 192 | 896 | 140 | 75.7% | 1.578 | 70.2% | 0.8492 | 0.4265 |
| 1168 | 112 | 1056 | 140 | 90.0% | 3.315 | 85.6% | 0.5513 | 0.2546 |
| 1232 | 48 | 1184 | 140 | 98.6% | 5.578 | 94.9% | 0.2753 | 0.1154 |
