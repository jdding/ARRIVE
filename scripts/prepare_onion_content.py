#!/usr/bin/env python3
"""Run the existing Onion content normalization, without neighbor experiments."""
import argparse
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "reproduction/native"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--musicnn", type=Path, required=True)
    parser.add_argument("--genre", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    import numpy as np
    from run_music4all_component_b2_attainability import prepare_features
    args.cache_dir = args.output
    prepare_features(args, len(np.load(args.data_dir / "train_frequencies.npy", allow_pickle=False)))


if __name__ == "__main__":
    main()
