#!/usr/bin/env python3
"""Convert the public Yambda (item_id, artist_id) parquet to native NPZ input."""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    import numpy as np
    import pyarrow.parquet as pq
    table = pq.read_table(args.input, columns=["item_id", "artist_id"])
    arrays = {name: table[name].to_numpy(zero_copy_only=False).astype(np.int64)
              for name in ("item_id", "artist_id")}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as stream:
        np.savez_compressed(stream, **arrays)


if __name__ == "__main__":
    main()
