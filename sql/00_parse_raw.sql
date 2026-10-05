-- Stage 0: parse the raw LendingClub file into a typed, loan-level table.
-- Every column is read as text first and then cast explicitly with TRY_CAST /
-- try_strptime, so a format surprise in a 2.2M-row file becomes a NULL that the
-- data-quality log counts, instead of a crash or a silent mis-parse.
-- :raw_path is substituted by load.py.

CREATE OR REPLACE TABLE parsed AS
WITH src AS (
    SELECT *
    FROM read_csv(':raw_path',
                  all_varchar = true,
                  header = true,
                  null_padding = true,
                  ignore_errors = true,
                  max_line_size = 20000000)
)
SELECT
    id,
    TRY_CAST(regexp_extract(term, '(\d+)', 1) AS INTEGER)                  AS term_months,
    CAST(try_strptime(trim(issue_d), '%b-%Y') AS DATE)                     AS issue_d,
    trim(loan_status)                                                      AS loan_status,
    upper(trim(application_type))                                          AS application_type,
    trim(grade)                                                            AS grade,
    trim(sub_grade)                                                        AS sub_grade,
    TRY_CAST(replace(trim(int_rate), '%', '') AS DOUBLE)                   AS int_rate,
    TRY_CAST(installment AS DOUBLE)                                        AS installment,
    TRY_CAST(loan_amnt AS DOUBLE)                                          AS loan_amnt,
    TRY_CAST(funded_amnt AS DOUBLE)                                        AS funded_amnt,
    TRY_CAST(annual_inc AS DOUBLE)                                         AS annual_inc,
    TRY_CAST(dti AS DOUBLE)                                                AS dti,
    CASE
        WHEN emp_length IS NULL OR trim(emp_length) IN ('', 'n/a') THEN NULL
        WHEN trim(emp_length) LIKE '<%' THEN 0
        ELSE TRY_CAST(regexp_extract(emp_length, '(\d+)', 1) AS INTEGER)
    END                                                                    AS emp_length_yrs,
    upper(trim(home_ownership))                                            AS home_ownership,
    trim(verification_status)                                              AS verification_status,
    lower(trim(purpose))                                                   AS purpose,
    trim(addr_state)                                                       AS addr_state,
    (TRY_CAST(fico_range_low AS DOUBLE) + TRY_CAST(fico_range_high AS DOUBLE)) / 2 AS fico,
    TRY_CAST(inq_last_6mths AS DOUBLE)                                     AS inq_last_6mths,
    TRY_CAST(delinq_2yrs AS DOUBLE)                                        AS delinq_2yrs,
    TRY_CAST(open_acc AS DOUBLE)                                           AS open_acc,
    TRY_CAST(pub_rec AS DOUBLE)                                            AS pub_rec,
    TRY_CAST(pub_rec_bankruptcies AS DOUBLE)                               AS pub_rec_bankruptcies,
    TRY_CAST(revol_bal AS DOUBLE)                                          AS revol_bal,
    TRY_CAST(replace(trim(revol_util), '%', '') AS DOUBLE)                 AS revol_util,
    TRY_CAST(total_acc AS DOUBLE)                                          AS total_acc,
    TRY_CAST(mort_acc AS DOUBLE)                                           AS mort_acc,
    CAST(try_strptime(trim(earliest_cr_line), '%b-%Y') AS DATE)            AS earliest_cr_line,
    -- outcome fields: monitoring and policy back-tests only, never model inputs
    TRY_CAST(total_pymnt AS DOUBLE)                                        AS total_pymnt,
    TRY_CAST(total_rec_prncp AS DOUBLE)                                    AS total_rec_prncp,
    TRY_CAST(total_rec_int AS DOUBLE)                                      AS total_rec_int,
    TRY_CAST(total_rec_late_fee AS DOUBLE)                                 AS total_rec_late_fee,
    TRY_CAST(recoveries AS DOUBLE)                                         AS recoveries,
    CAST(try_strptime(trim(last_pymnt_d), '%b-%Y') AS DATE)                AS last_pymnt_d
FROM src;
