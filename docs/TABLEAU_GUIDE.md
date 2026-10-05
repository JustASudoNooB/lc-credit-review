# Building the Tableau Public dashboard

Three views, built by hand from the CSVs in `outputs/tables/`. Allow about 2 hours the first time.
Tableau Public (Desktop) is free. Sign up at public.tableau.com and download Tableau Public.

Rule for every chart: one measure per axis. Never put two different measures on a dual axis. If you need
volume and bad rate together, make two charts stacked on top of each other.

## Connect the data

1. Open Tableau Public, then **Connect > Text file**. Pick `outputs/tables/mob_curves_by_quarter.csv`.
2. To add each further CSV, click **Add** next to Connections. Keep them as separate data sources;
   don't join them.

## View 1: Vintage monitor

**Sheet "Vintage curves"** (source: `mob_curves_by_quarter.csv`)
- Columns: `mob` (dimension, continuous). Rows: `cum_default_rate` (aggregation: SUM, since there is one row per quarter and MOB).
- Drag `issue_q` to Color, then filter `issue_q` to the quarters you want to compare (for example, the first
  quarter of each year). Or rebuild the chart with `issue_year` and a calculated field:
  `SUM([new_defaults]) / SUM([cohort_loans])` with a running total table calculation along `mob`.
- Format `cum_default_rate` as a percentage with 1 decimal.
- Caption: "Default month approximated as the month after the last payment received."

**Sheet "Quarterly bad rate"** (source: `portfolio_by_quarter.csv`)
- Columns: `issue_q`. Rows: `bad_rate` (line), with a second line for `bad_rate_rolling_4q` on the same axis.

**Sheet "Quarterly volume"** (same source)
- Columns: `issue_q`. Rows: `loans` (bars).

## View 2: Cut-off strategy

**Sheet "Bad rate by approval rate"** (source: `cutoff_strategy.csv`)
- Columns: `approval_rate` (continuous). Rows: `bad_rate`. Color: `model`.
- Add a reference line at 0.8 labelled "80% approval".

**Sheet "Return by approval rate"** (same source)
- Columns: `approval_rate`. Rows: `net_return`. Color: `model`.

**Sheet "Swap-set"** (source: `swap_set_grade_vs_scorecard.csv`)
- Columns: `cell`. Rows: `bad_rate` (bars). Label: `bad_rate` and `loans`.

## View 3: Root cause

**Sheet "Mix vs rate by grade"** (source: `rca_mix_rate_by_grade.csv`)
- Pivot `mix` and `rate` in the data source (select both columns, right-click, **Pivot**).
- Columns: `grade`. Rows: `Pivot Field Values`. Color: `Pivot Field Names`. Use side-by-side bars, not stacked.

**Sheet "Within-grade drivers"** (source: `rca_within_grade_drilldown.csv`)
- Filter: `dimension` (single value, shown as a dropdown).
- Rows: `segment`, sorted by `rate_contribution`. Columns: `rate_contribution` (bars).
- Tooltip: `avg_rate_change_pp`, `share_period_2`.

**Sheet "Early warning"** (source: `early_warning.csv`)
- Text table: `dimension`, `segment`, `baseline_bad_rate`, `first_breach_quarter`, `latest_bad_rate`.
  Sort by `first_breach_quarter`.

## Assemble and publish

1. **New Dashboard**, size Automatic. Make one dashboard per view, or one long dashboard with three sections.
2. Add a title text box: "LendingClub 2012-2015: vintage monitoring, model cut-offs and root cause".
3. **File > Save to Tableau Public**. Copy the public link and put it in the README and on your CV.

Every number on the dashboard comes from the CSVs, so it will match the reports exactly.
