#!/usr/bin/env python3
"""Summarize saved three-seed results; never train, select or infer a model."""
import argparse
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from export_io import require_fresh_output, write_json
from result_summary import summarize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = require_fresh_output(args.output)
    report = summarize(json.loads(args.config.read_text()), args.config.resolve().parent)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "summary.json", report)
    print(json.dumps({"status": "pass", "ranking_rows": len(report["ranking_rows"]),
                      "paired_comparisons": len(report["paired_comparisons"]),
                      "coverage_rows": len(report["coverage"]), "new_training": False}))


if __name__ == "__main__":
    main()
