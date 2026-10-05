-- Stage 0: modelling population.
-- 36-month, individual loans issued Jan 2012 - Dec 2015 with a final outcome.
-- Every one of them has passed its scheduled maturity by the Dec 2018 data snapshot,
-- so each loan is either fully paid or charged off: no censoring.
-- :start, :end, :term, :bad_list and :good_list are substituted by load.py.

CREATE OR REPLACE TABLE loans AS
SELECT
    p.*,
    CASE WHEN loan_status IN (:bad_list) THEN 1 ELSE 0 END                 AS bad,
    year(issue_d)                                                          AS issue_year,
    CAST(year(issue_d) AS VARCHAR) || 'Q' || CAST(quarter(issue_d) AS VARCHAR) AS issue_q,
    date_diff('month', earliest_cr_line, issue_d)                          AS credit_hist_months,
    CASE WHEN annual_inc > 0 THEN loan_amnt / annual_inc END               AS loan_to_income,
    -- post-loan DTI: reported DTI (existing debts, ex-mortgage) plus the new instalment
    CASE WHEN annual_inc > 0 AND installment IS NOT NULL AND dti IS NOT NULL
         THEN dti + 100.0 * 12.0 * installment / annual_inc END            AS dti_post,
    -- A1 = 1 (lowest risk) ... G5 = 35 (highest risk)
    (ascii(substr(sub_grade, 1, 1)) - ascii('A')) * 5
        + TRY_CAST(substr(sub_grade, 2, 1) AS INTEGER)                     AS sub_grade_rank
FROM parsed AS p
WHERE issue_d BETWEEN DATE ':start' AND DATE ':end'
  AND term_months = :term
  AND application_type = 'INDIVIDUAL'
  AND (loan_status IN (:bad_list) OR loan_status IN (:good_list));
