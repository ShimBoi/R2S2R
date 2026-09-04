#!/usr/bin/env python3
"""Assembles the publishable dashboard HTML: runs the data extractor, then substitutes its
output into index.html's __DASHBOARD_DATA__ placeholder, writing dashboard_built.html
(the file actually passed to the Artifact tool). Re-run this, then republish, to refresh.
"""
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent
TEMPLATE = HERE / "index.html"
OUT = HERE / "dashboard_built.html"
DATA_PATH = HERE / "dashboard_data.json"


def main() -> None:
    subprocess.run([sys.executable, str(HERE / "extract_dashboard_data.py")], check=True)
    data = json.loads(DATA_PATH.read_text())
    template = TEMPLATE.read_text()
    injected = template.replace("__DASHBOARD_DATA__", json.dumps(data))
    OUT.write_text(injected)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
