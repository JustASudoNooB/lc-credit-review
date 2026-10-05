"""Rebuild every output from the raw LendingClub file.

    python run_all.py                       # uses data/raw/accepted_2007_to_2018Q4.csv.gz
    python run_all.py --raw "D:/path with spaces/accepted_2007_to_2018Q4.csv.gz"
    python run_all.py --reports-only        # re-write figures and documents from outputs/results.json
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from creditlab import config as C  # noqa: E402
from creditlab import figures, pipeline, report  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", type=Path, default=C.RAW_FILE, help="path to the raw LendingClub csv or csv.gz")
    ap.add_argument("--reports-only", action="store_true")
    args = ap.parse_args()

    if not args.reports_only:
        if not args.raw.exists():
            sys.exit(f"Raw file not found: {args.raw}\nDownload accepted_2007_to_2018Q4.csv.gz from Kaggle "
                     f"('Lending Club Loan Data', wordsforthewise) into {C.DATA_RAW}")
        pipeline.run(args.raw)
    print("[6] figures and reports ...")
    figures.all_figures()
    info = report.write_all()
    print(f"verdict: {info['verdict']}")
    if info["synthetic"]:
        print("\n*** These results come from SYNTHETIC test data. Do not quote them. ***")
    print(f"\nKey numbers: {C.RESULTS_JSON}\nReports: {C.REPORTS}\nCV bullets: {C.REPORTS / 'CV_BULLETS.md'}")


if __name__ == "__main__":
    main()
