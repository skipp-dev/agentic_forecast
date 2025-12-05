#!/usr/bin/env python
"""
Open the latest HTML system report in your default browser.

Assumes reports are stored as:
  results/reports/report_YYYYMMDD_HHMMSS.html
"""

import sys
import webbrowser
from pathlib import Path


def find_repo_root() -> Path:
    """
    Resolve repo root as the parent of this script's directory.
    Adjust if your layout is different.
    """
    return Path(__file__).resolve().parents[1]


def find_latest_report_html(report_dir: Path) -> Path:
    """
    Find the newest report_*.html in results/reports.
    """
    if not report_dir.exists():
        print(f"[ERROR] Reports directory does not exist: {report_dir}", file=sys.stderr)
        sys.exit(1)

    # You can switch to rglob if reports are nested in subfolders
    candidates = sorted(report_dir.glob("report_*.html"))

    if not candidates:
        print(f"[ERROR] No HTML reports found in: {report_dir}", file=sys.stderr)
        sys.exit(1)

    # Files are timestamped in filename, but sort by mtime to be safe
    latest = max(candidates, key=lambda p: p.stat().st_mtime)
    return latest


def main() -> None:
    repo_root = find_repo_root()
    report_dir = repo_root / "results" / "reports"

    latest = find_latest_report_html(report_dir)

    print(f"[INFO] Latest HTML report: {latest}")
    try:
        webbrowser.open(latest.as_uri())
        print("[INFO] Opened in default browser.")
    except Exception as e:
        print(f"[ERROR] Failed to open browser: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
