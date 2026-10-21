#!/usr/bin/env python3
"""One reusable, complete Train-only transition cache for repaired MUSE."""
import argparse
from pathlib import Path
import numpy as np
from scipy import sparse
from arrive_muse_repaired import train_transition_counts
from run_arrive_main_table_baseline import digest
from run_lfm_timeconcat_control import atomic_json

p = argparse.ArgumentParser()
p.add_argument("--data", type=Path, default=Path("data/music4all_onion/prepared_v1"))
p.add_argument("--times", type=Path, default=Path("data/music4all_onion/baseline_features_v1/event_times.npy"))
p.add_argument("--output", type=Path, default=Path("data/music4all_onion/main_table_repair_v1"))
a = p.parse_args()
if a.output.exists():
    raise FileExistsError("muse_counts_output_exists")
events = np.load(a.data / "events.npy", mmap_mode="r")
times = np.load(a.times, mmap_mode="r")
sessions = np.load(a.data / "sessions.npy", mmap_mode="r")
vocab_size = len(np.load(a.data / "train_frequencies.npy", mmap_mode="r"))
if events.shape != times.shape:
    raise ValueError("event_time_shape_mismatch")
counts = train_transition_counts(events, times, sessions, vocab_size=vocab_size)
a.output.mkdir(parents=True, exist_ok=False)
path = a.output / "muse_transition_counts.npz"
sparse.save_npz(path, counts)
receipt = {"status": "pass", "test_accessed": False, "validation_accessed": False,
           "inputs": {"events": digest(a.data / "events.npy"), "sessions": digest(a.data / "sessions.npy"), "times": digest(a.times)},
           "counts_sha256": digest(path), "edges": int(counts.nnz),
           "builder_sha256": digest(Path(__file__)),
           "operators_sha256": digest(Path(__file__).with_name("arrive_muse_repaired.py"))}
atomic_json(a.output / "MUSE_COUNTS.json", receipt)
print(receipt, flush=True)
