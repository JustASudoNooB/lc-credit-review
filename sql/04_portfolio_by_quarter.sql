-- Stage 1c: portfolio health by issue quarter (the monitoring "front page").
-- Volume, bad rate, average borrower quality and price, plus quarter-on-quarter change
-- in bad rate computed with LAG so a deteriorating trend stands out.

WITH q AS (
    SELECT
        issue_q,
        issue_year,
        COUNT(*)                                  AS loans,
        SUM(funded_amnt)                          AS funded_amnt,
        AVG(bad)                                  AS bad_rate,
        AVG(fico)                                 AS avg_fico,
        AVG(dti)                                  AS avg_dti,
        AVG(int_rate)                             AS avg_int_rate,
        AVG(CASE WHEN grade IN ('D', 'E', 'F', 'G') THEN 1.0 ELSE 0 END) AS share_grade_d_or_worse,
        AVG(CASE WHEN verification_status = 'Not Verified' THEN 1.0 ELSE 0 END) AS share_income_not_verified,
        SUM(total_pymnt - funded_amnt) / SUM(funded_amnt) AS realised_net_return
    FROM loans
    GROUP BY issue_q, issue_year
)
SELECT
    *,
    bad_rate - LAG(bad_rate) OVER (ORDER BY issue_q)                  AS bad_rate_change_qoq,
    AVG(bad_rate) OVER (ORDER BY issue_q
                        ROWS BETWEEN 3 PRECEDING AND CURRENT ROW)     AS bad_rate_rolling_4q
FROM q
ORDER BY issue_q;
