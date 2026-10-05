-- Stage 1b: months-on-book (MOB) cumulative charge-off curves by issue quarter.
--
-- APPROXIMATION (state it whenever this chart is shown): the file has no monthly payment
-- history, so default timing is taken as the month after the last payment received
-- (the issue month for loans that never paid). The true charge-off is booked a few months
-- later, so these curves are shifted earlier than an accounting charge-off curve.
-- The end point of every curve (MOB 36) equals the exact static-pool bad rate.

WITH timing AS (
    SELECT
        issue_q,
        bad,
        CASE WHEN bad = 1 THEN
            LEAST(36, GREATEST(1,
                date_diff('month', issue_d, COALESCE(last_pymnt_d, issue_d)) + 1))
        END AS default_mob
    FROM loans
),
cohort AS (
    SELECT issue_q, COUNT(*) AS cohort_loans
    FROM timing
    GROUP BY issue_q
),
events AS (
    SELECT issue_q, default_mob AS mob, COUNT(*) AS new_defaults
    FROM timing
    WHERE bad = 1
    GROUP BY issue_q, default_mob
),
grid AS (
    SELECT c.issue_q, c.cohort_loans, m.mob
    FROM cohort AS c
    CROSS JOIN (SELECT range AS mob FROM range(1, 37)) AS m
)
SELECT
    g.issue_q,
    CAST(substr(g.issue_q, 1, 4) AS INTEGER)                            AS issue_year,
    g.mob,
    g.cohort_loans,
    COALESCE(e.new_defaults, 0)                                          AS new_defaults,
    SUM(COALESCE(e.new_defaults, 0)) OVER (
        PARTITION BY g.issue_q ORDER BY g.mob
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)                AS cum_defaults,
    SUM(COALESCE(e.new_defaults, 0)) OVER (
        PARTITION BY g.issue_q ORDER BY g.mob
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) * 1.0
        / g.cohort_loans                                                 AS cum_default_rate
FROM grid AS g
LEFT JOIN events AS e
       ON e.issue_q = g.issue_q AND e.mob = g.mob
ORDER BY g.issue_q, g.mob;
