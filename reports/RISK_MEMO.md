# Risk committee memo

**Subject:** 2015 vintage performance, decision-model review and two underwriting options

**Prepared by:** Akhand Raj  
**Basis:** 589,249 LendingClub 36-month personal loans issued 2012-2015, all at final outcome

## Findings

1. **The bad rate rose from 12.3% (2013 loans) to 14.9% (2015 loans), +2.56 pp.** Lending more to riskier grades explains -0.56 pp; the same grades performing worse explains +3.11 pp. Inside grades, the sharpest deterioration was in **fico_band = <680** (+4.5 pp, 38% of 2015 loans). The earliest sustained warning came from **fico_band = <680** in 2012Q1.
2. **On 2015 loans, the LendingClub sub-grade ranks risk with Gini 0.357; my application scorecard 0.304; the gradient-boosting challenger 0.341; grade plus scorecard together 0.371.** Both new models were built on 2012-2014 data: the scorecard under-predicts the 2015 bad rate by 1.9 pp and the challenger under-predicts it by 1.4 pp.
3. **At an 80% approval rate, replacing the sub-grade with the challenger swaps 29,907 loans: those brought in go bad at 22.2%, those dropped at 21.4%** (portfolio bad rate 11.9% on the sub-grade vs 12.0% on the challenger). A post-loan DTI cap of 35 (existing debts plus the new instalment) would decline 21% of loans with a 20.6% bad rate against 13.3% for those kept; inside score quintiles the cap separates risk significantly in 5 of 5.

## Recommendations

1. **Keep the sub-grade as the primary model and add the scorecard as an overlay** (a second look at the cut-off margin), after recalibrating its intercept to the latest complete vintage.
2. **Add a post-loan DTI cap of 35** as a hard policy rule: it finds risk the score has not priced in most score bands.
3. **Monitor monthly:** score PSI and feature CSI (watch at 0.10, act at 0.25), the vintage curves at months-on-book 6 and 12, and the early-warning list, starting with fico_band = <680.

Supporting detail: `reports/MODEL_REVIEW.md`, `outputs/tables/`, `outputs/figures/`.
