# Model review: application scorecard and GBM challenger vs LendingClub sub-grade

Data: `accepted_2007_to_2018Q4.csv.gz`. Run: 2026-10-04T11:11:03+00:00. Every number below is read from `outputs/results.json`.

## 1. Purpose

LendingClub's sub-grade (A1 to G5) is treated as the incumbent underwriting model. Two candidate models are built only from information available at application time and reviewed the way a model risk team reviews a decision model: discrimination, calibration, stability, driver sense-check, limitations, verdict.

## 2. Data

Population filters (row count after each step):

| step | rows | dropped |
|---|---|---|
| rows read from raw file | 2260701 | 0 |
| valid issue date | 2260668 | 33 |
| issued 2012-01 to 2015-12 | 844905 | 1415763 |
| term = 36 months | 589635 | 255270 |
| individual applications | 589396 | 239 |
| final outcome (paid off or charged off) | 589249 | 147 |

Target: bad = charged off or default; good = fully paid. Loans still open at the snapshot are excluded (see `outputs/tables/excluded_statuses.csv`).

| sample | definition | loans | bad rate |
|---|---|---|---|
| train | 2012-2014, random 80% | 245,169 | 13.2% |
| holdout | 2012-2014, random 20% | 61,293 | 13.2% |
| out-of-time | 2015 | 282,787 | 14.9% |

## 3. Design choices

- 19 candidate application-time features. Excluded on purpose:
  - `addr_state`: geographic proxy for protected characteristics (fair-lending risk); also unstable across vintages
  - `emp_title`: free text with thousands of values; would need its own NLP treatment
  - `grade / sub_grade / int_rate / installment`: LendingClub's own model outputs; kept only as the champion benchmark
- Binning: 20 quantile fine classes, merged to monotonic bad rates, minimum 5% of rows per bin, at most 8 bins, missing values in their own bin.
- Selection: IV >= 0.02, pairwise |correlation| <= 0.7 (keep higher IV), every coefficient negative on WoE.
- Selection log:
  - dropped emp_length_yrs: IV 0.0188 < 0.02
  - dropped revol_bal: IV 0.0170 < 0.02
  - dropped total_acc: IV 0.0094 < 0.02
  - dropped verification_status: IV 0.0083 < 0.02
  - dropped loan_amnt: IV 0.0080 < 0.02
  - dropped pub_rec: IV 0.0018 < 0.02
  - dropped delinq_2yrs: IV 0.0013 < 0.02
  - dropped pub_rec_bankruptcies: IV 0.0012 < 0.02
  - dropped open_acc: IV 0.0002 < 0.02
- Final scorecard features (10): `fico`, `annual_inc`, `dti`, `mort_acc`, `loan_to_income`, `home_ownership`, `inq_last_6mths`, `revol_util`, `credit_hist_months`, `purpose`
- Largest VIF among final features: 1.92
- Scaling: 600 points at 50:1 good:bad odds, 20 points to double the odds (`outputs/tables/scorecard_points.csv`).
- Challenger: histogram gradient boosting on raw features, early stopping on 15% of train (343 trees kept), monotone constraints: `fico` down, `dti` up, `inq_last_6mths` up, `revol_util` up, `loan_to_income` up.
- Overlay: logistic regression on [logit of grade-implied PD, scorecard log-odds], fitted on train.

## 4. Discrimination

| model | train Gini | holdout Gini | OOT Gini (95% CI) | OOT KS | OOT Gini vs champion (95% CI) |
|---|---|---|---|---|---|
| LC sub-grade (champion) | 0.299 | 0.304 | 0.357 (0.353-0.363) | 0.262 | - |
| WoE scorecard | 0.299 | 0.305 | 0.304 (0.300-0.310) | 0.220 | -0.053 (-0.058 to -0.048) |
| GBM challenger | 0.412 | 0.340 | 0.341 (0.337-0.347) | 0.247 | -0.016 (-0.020 to -0.011) |
| Grade + scorecard overlay | 0.334 | 0.340 | 0.371 (0.366-0.376) | 0.271 | +0.013 (+0.011 to +0.016) |

Challenger minus scorecard on OOT: +0.037 Gini (95% CI +0.034 to +0.039). Confidence intervals come from 200 paired bootstrap resamples of the 2015 loans.

![Gini with confidence intervals](../outputs/figures/gini_with_ci.png)

## 5. Stability

| model | score PSI, train vs 2015 |
|---|---|
| LC sub-grade (champion) | 0.027 |
| WoE scorecard | 0.001 |
| GBM challenger | 0.002 |
| Grade + scorecard overlay | 0.004 |

Feature-level CSI (scorecard features):

| feature | csi | status |
|---|---|---|
| mort_acc | 0.2002 | watch |
| inq_last_6mths | 0.0413 | stable |
| revol_util | 0.0246 | stable |
| dti | 0.0237 | stable |
| credit_hist_months | 0.0087 | stable |
| fico | 0.0042 | stable |
| annual_inc | 0.004 | stable |
| home_ownership | 0.0039 | stable |
| loan_to_income | 0.0017 | stable |
| purpose | 0.0008 | stable |

## 6. Calibration

| model | predicted 2015 bad rate | actual | gap |
|---|---|---|---|
| LC sub-grade (champion) | 12.85% | 14.88% | +2.04 pp |
| WoE scorecard | 13.03% | 14.88% | +1.85 pp |
| GBM challenger | 13.44% | 14.88% | +1.44 pp |
| Grade + scorecard overlay | 12.85% | 14.88% | +2.03 pp |

A PD fitted on 2012-2014 is applied to 2015 loans. If 2015 performed worse than earlier vintages at the same risk profile, the model ranks correctly but under-predicts the level: that is a calibration problem, fixed by re-estimating the intercept, not by rebuilding the model.

![Calibration](../outputs/figures/calibration_oot.png)

## 7. Challenger drivers

Permutation importance on 2015 loans (Gini lost when the feature is shuffled):

| feature | gini_drop |
|---|---|
| fico | 0.095 |
| revol_bal | 0.0419 |
| dti | 0.0267 |
| loan_to_income | 0.026 |
| inq_last_6mths | 0.0259 |
| open_acc | 0.0226 |
| purpose | 0.0211 |
| annual_inc | 0.0161 |

Direction check on the top drivers (average predicted PD as the feature is swept low to high):

| feature | model | expected |
|---|---|---|
| dti | rises | rises |
| fico | falls | falls |
| loan_to_income | rises | rises |
| revol_bal | falls | n/a |

## 8. Limitations

- Reject inference: only loans LendingClub approved are observed. Every model here is trained and tested on loans the champion had already accepted, so none of these results says how the models would rank the applicants LendingClub declined.
- The sub-grade uses bureau and platform data this dataset does not contain, so the comparison is partly a test of data, not only of method.
- 2012-2015 is one benign US credit cycle. No recession is in the sample.
- US unsecured personal loans. An Indian personal-loan book differs in bureau depth, income verification and collections; the method transfers, the coefficients do not.
- Realised return ignores servicing fees, funding cost and time value of money.

## 9. Conclusion

**APPROVE AS AN OVERLAY, NOT A REPLACEMENT: neither new model beats the sub-grade on its own, but adding the scorecard to the sub-grade improves ranking (Gini +0.013, 95% CI +0.011 to +0.016), so the scorecard carries information the grade does not.**

Conditions:

1. The scorecard under-predicts the 2015 bad rate by 1.9 pp: recalibrate the intercept on the most recent complete vintage before any PD is used for pricing or provisioning.
1. The challenger under-predicts the 2015 bad rate by 1.4 pp: recalibrate the intercept on the most recent complete vintage before any PD is used for pricing or provisioning.
1. The challenger loses 0.071 Gini from train to out-of-time; it fits the training sample more closely than it generalises, so monitor it monthly.
1. Monitor monthly: score PSI and feature CSI (watch at 0.10, act at 0.25), Gini on each new matured vintage, and calibration by decile.
