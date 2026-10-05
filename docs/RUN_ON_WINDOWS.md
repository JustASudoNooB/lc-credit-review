# Running it on your laptop (Windows, PowerShell)

Expect 10-20 minutes for the full run on the real file. Most of that is reading the 400 MB gzip once.

## 1. Get the data

1. Log in to Kaggle and open the "Lending Club Loan Data" dataset by wordsforthewise.
2. Download `accepted_2007_to_2018Q4.csv.gz`. Kaggle may hand you a zip: extract it, but keep the `.csv.gz`
   inside as it is, without decompressing it.
3. Put it in `data\raw\` inside this folder.

## 2. Set up Python

```powershell
cd "C:\path with spaces\lc-credit-review"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

If PowerShell blocks the activate script, run this once:
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

## 3. Run

```powershell
python -m pytest              # 32 tests, about 5 seconds
python run_all.py             # full rebuild from the raw file
```

If the file sits somewhere else:
`python run_all.py --raw "D:\downloads\accepted_2007_to_2018Q4.csv.gz"`

## 4. Check before you trust anything

Open these in this order:

1. `outputs\tables\data_quality_funnel.csv`: every filter should drop a sensible number of rows. The
   final population should be several hundred thousand loans.
2. `outputs\tables\profile_status_counts.csv` and `profile_payment_identity.csv`. The second one shows whether
   `total_pymnt` already includes recoveries; on the real file the share should be close to 1.
3. `outputs\tables\iv_table.csv`: no feature above 0.5. FICO should be the strongest.
4. `reports\MODEL_REVIEW.md`, then `reports\RISK_MEMO.md`.

If any of these look wrong, stop and send me the file. Don't put a number on the CV until it passes.

## 5. Put it on GitHub

```powershell
git init
git add .
git commit -m "Retail credit model review and underwriting strategy on LendingClub 2012-2015"
```

Create an empty public repository on github.com called `lc-credit-review` (no README), then:

```powershell
git remote add origin https://github.com/JustASudoNooB/lc-credit-review.git
git branch -M main
git push -u origin main
```

The `.gitignore` keeps the data out of the repository. Commit `outputs\` and `reports\` so a reviewer can
read the results without running anything.
