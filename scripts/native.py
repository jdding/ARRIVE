#!/usr/bin/env python3
"""Run an included native producer with its local dependency directory."""
import argparse
import os
import subprocess
import sys
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    native = root / "reproduction" / "native"
    modules = sorted(p.stem for p in native.glob("*.py") if p.stem.startswith(
        ("prepare_", "finalize_", "materialize_", "derive_", "run_", "evaluate_", "audit_", "arrive_train_fold", "reviewer_freeze")))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("module", choices=modules)
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    arguments = args.arguments[1:] if args.arguments[:1] == ["--"] else args.arguments
    environment = os.environ.copy()
    environment.update(ARRIVE_TOOLS=str(native), ARRIVE_CUE_RANKER_PATH=str(native / "run_arrive_recent_ranker.py"),
                       ARRIVE_CUE_DONOR_PATH=str(native / "arrive_cue_donors.py"))
    result = subprocess.run([sys.executable, str(native / (args.module + ".py")), *arguments], env=environment)
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
