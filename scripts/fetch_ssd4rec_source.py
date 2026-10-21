#!/usr/bin/env python3
"""Acquire the pinned official SSD4Rec source; no dependency install or training."""
import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

COMMIT = "bdbfe5193f3a6697bb6ee0699ab43386d80c6198"
URL = "https://raw.githubusercontent.com/ZhangYifeng1995/SSD4Rec/" + COMMIT + "/ssd4rec.py"
EXPECTED = "d11dd7087606c223d3630ae6884abb681115db08d9197f59bf7a76d520b58cb6"
VENDORED = "2b31ea7f5418391b49e3c76776d1c0dae75905729f65ddef4c8242ba2f1a43d1"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("vendor/ssd4rec_official/ssd4rec.py"))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    with urllib.request.urlopen(URL, timeout=60) as response:
        body = response.read()
    if hashlib.sha256(body).hexdigest() != EXPECTED:
        raise ValueError("pinned_upstream_source_hash_mismatch")
    # The recorded local source adds exactly one terminal newline.
    prepared = body + b"\n"
    if hashlib.sha256(prepared).hexdigest() != VENDORED:
        raise ValueError("recorded_newline_normalization_mismatch")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as stream:
        stream.write(prepared)
    print(json.dumps({"source": URL, "commit": COMMIT, "sha256": VENDORED}))


if __name__ == "__main__":
    main()
