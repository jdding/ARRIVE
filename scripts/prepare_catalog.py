#!/usr/bin/env python3
"""Build catalog inputs for matching controls using the native metadata loader."""
import argparse
import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path[:0] = [str(REPO / "reproduction/native"), str(REPO / "tools")]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data-dir", "audio", "genre", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("tags", "information", "artist-mapping"):
        parser.add_argument("--" + name, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.artist_mapping is None and not (args.tags and args.information):
        parser.error("Onion requires --tags and --information; Yambda requires --artist-mapping")
    import numpy as np
    from scipy import sparse
    from run_music4all_artist_component_b import load_metadata
    from export_io import CATALOG_FILES, digest
    frequencies = np.load(args.data_dir / "train_frequencies.npy", allow_pickle=False)
    metadata_args = SimpleNamespace(data_dir=args.data_dir, tags=args.tags, information=args.information,
        artist_mapping=args.artist_mapping, audio_array=args.audio if args.artist_mapping else None)
    tags, artists, _ = load_metadata(metadata_args, len(frequencies))
    args.output.mkdir(parents=True, exist_ok=False)
    np.save(args.output / "artists.npy", artists, allow_pickle=False)
    sparse.save_npz(args.output / "tags.npz", sparse.csr_matrix(tags, dtype=np.float32))
    for source, name in ((args.audio, "audio.npy"), (args.genre, "genre.npz"),
                         (args.data_dir / "train_frequencies.npy", "train_frequencies.npy")):
        shutil.copyfile(source, args.output / name)
    binding = {name: digest(args.output / name) for name in CATALOG_FILES}
    record = dict(schema="arrive.reviewer_catalog.v1", catalog_id="catalog-" + binding["artists.npy"][:16],
        counts_scope="train_only", uses_future_labels=False, sha256=binding,
        track_identity_sha256=digest(args.data_dir / "track_identity.tsv"))
    (args.output / "CATALOG_MANIFEST.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({"status": "complete", "catalog_id": record["catalog_id"]}))


if __name__ == "__main__":
    main()
