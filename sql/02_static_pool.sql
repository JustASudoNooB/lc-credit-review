-- Stage 1a: static-pool lifetime bad rate by issue quarter x grade.
-- A static pool is a fixed cohort of loans issued in the same period, followed to the end.
-- Each row answers: of the loans we booked in this quarter at this grade, what share went bad,
-- by count and by amount, and how big was this grade within the quarter's bookings?

WITH pool AS (
    SELECT issue_q, issue_year, grade, bad, funded_amnt
    FROM loans
),
by_cell AS (
    SELECT
        issue_q,
        issue_year,
        grade,
        COUNT(*)                         AS loans,
        SUM(bad)                         AS bads,
        SUM(funded_amnt)                 AS funded_amnt,
        SUM(bad * funded_amnt)           AS bad_amnt
    FROM pool
    GROUP BY issue_q, issue_year, grade
)
SELECT
    issue_q,
    issue_year,
    grade,
    loans,
    bads,
    bads * 1.0 / loans                                                  AS bad_rate,
    bad_amnt / funded_amnt                                              AS bad_rate_by_amount,
    funded_amnt,
    loans * 1.0 / SUM(loans) OVER (PARTITION BY issue_q)                AS grade_share_of_quarter,
    -- quarter-level bad rate repeated on each row, so Tableau can draw both without a join
    SUM(bads) OVER (PARTITION BY issue_q) * 1.0
        / SUM(loans) OVER (PARTITION BY issue_q)                        AS quarter_bad_rate
FROM by_cell
ORDER BY issue_q, grade;
