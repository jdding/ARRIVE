#!/usr/bin/env python3
"""Check native export -> catalog -> three ranker arms on synthetic data only."""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(REPO / "reproduction/native"))


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def run(*arguments):
    result = subprocess.run([sys.executable, *map(str, arguments)], cwd=REPO,
                            capture_output=True, text=True, timeout=90,
                            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
    if result.returncode:
        raise RuntimeError(f"{arguments}\n{result.stdout}\n{result.stderr}")


def fixture(root):
    import numpy as np
    import torch
    from scipy import sparse
    from lfm_timeconcat_control import LFMTimeConcatData
    from run_music4all_baselines import create_model
    from run_music4all_candidate_support_a1 import CROSSFIT_SCHEMA

    torch.set_num_threads(1)
    torch.manual_seed(2024)
    data = root / "data/music4all_onion/prepared_v1"
    content = root / "data/music4all_onion/component_b2_attainability_v1"
    raw = root / "data/music4all_onion/raw"
    index = root / "index"
    for path in (data, content, raw, index):
        path.mkdir(parents=True)
    vocab, rows = 259, 16
    events, sessions = [], []
    for at in range(rows):
        begin = len(events)
        events += [3 + at % 8, 20 + at % 8, 40 + at % 8, 60 + at % 8]
        sessions.append([100 + at % 8, begin, begin + 1, begin + 4,
                         35 * 86400, 0 if at < 8 else 1])
    events, sessions = np.asarray(events, np.int32), np.asarray(sessions, np.int64)
    np.save(data / "events.npy", events)
    np.save(data / "sessions.npy", sessions)
    frequency = np.ones(vocab, np.int64)
    frequency[:3] = 0
    np.save(data / "train_frequencies.npy", frequency)
    np.savez(data / "metric_labels.npz", row_ids=np.arange(rows),
             offsets=np.arange(0, rows * 2 + 1, 2),
             values=np.asarray([[40 + i % 8, 60 + i % 8] for i in range(rows)]).ravel())
    save_json(data / "PREP_STATS.json", {"synthetic": True, "rows": rows})
    (data / "track_identity.tsv").write_text("raw_index\tmodel_id\ttrack_id\ttrain_events\n" +
        "".join(f"{i-3}\t{i}\tsong-{i}\t1\n" for i in range(3, vocab)))
    (raw / "id_tags.csv").write_text("id\ttags\n" +
        "".join(f"song-{i}\ttag-{i % 7}, shared\n" for i in range(3, vocab)))
    (raw / "id_information.csv").write_text("id\tartist\tsong\talbum_name\n" +
        "".join(f"song-{i}\tartist-{i % 9}\tsong-{i}\talbum\n" for i in range(3, vocab)))
    rng = np.random.default_rng(2024)
    audio = rng.normal(size=(vocab, 8)).astype(np.float32)
    audio[:3] = 0
    audio /= np.maximum(np.linalg.norm(audio, axis=1, keepdims=True), 1e-8)
    np.save(content / "musicnn_normalized.npy", audio)
    category = sparse.csr_matrix((np.ones(vocab - 3, np.float32),
        (np.arange(3, vocab), np.arange(3, vocab) % 5)), shape=(vocab, 5))
    sparse.save_npz(content / "genre_normalized.npz", category)
    transition = sparse.csr_matrix((np.ones(vocab - 3),
        (np.arange(3, vocab), 3 + (np.arange(3, vocab) - 2) % (vocab - 3))), shape=(vocab, vocab))
    for suffix in ("fit", "full"):
        for channel in ("x", "h"):
            sparse.save_npz(index / f"{channel}_{suffix}.npz", transition)
        np.save(index / f"{suffix}_target_frequency.npy", frequency)
    np.save(index / "cv_rows.npy", np.arange(8))
    names = ("events.npy", "sessions.npy", "train_frequencies.npy", "track_identity.tsv",
             "metric_labels.npz", "PREP_STATS.json")
    prepared = {name: sha(data / name) for name in names}
    save_json(index / "index_manifest.json", dict(status="pass", test_accessed=False,
        synthetic=True, prepared_artifacts=prepared,
        artifacts={path.name: {"sha256": sha(path)} for path in sorted(index.iterdir())}))
    configuration = dict(baseline="sasrec_standard", embed_dim=8, num_heads=2,
        num_layers=1, dropout=0., max_context=129, horizon=64, seed=2024)
    model = create_model(SimpleNamespace(**configuration), LFMTimeConcatData(data))
    checkpoints = {}
    for surface in ("train_cv", "validation"):
        directory = root / surface
        directory.mkdir()
        binding = dict(configuration=configuration, prepared_artifacts=prepared)
        record = dict(status="pass", test_accessed=False, synthetic=True, binding=binding)
        if surface == "train_cv":
            record.update(schema=CROSSFIT_SCHEMA, excluded_cv_rows=8)
            binding.update(base_method="sasrec", cv_rows_sha256=sha(index / "cv_rows.npy"),
                           index_manifest_sha256=sha(index / "index_manifest.json"))
            hash_key = "final_model_sha256"
        else:
            record["baseline"] = "sasrec_standard"
            hash_key = "best_model_sha256"
        checkpoint = directory / "initialized_only.pt"
        torch.save({"binding": binding, "model": model.state_dict()}, checkpoint)
        record[hash_key] = sha(checkpoint)
        save_json(directory / "training_manifest.json", record)
        checkpoints[surface] = checkpoint
    return data, content, raw, index, checkpoints


def main():
    with tempfile.TemporaryDirectory(prefix="arrive-native-smoke-") as directory:
        root = Path(directory)
        data, content, raw, index, checkpoints = fixture(root)
        catalog = root / "catalog"
        run(REPO / "scripts/prepare_catalog.py", "--data-dir", data,
            "--audio", content / "musicnn_normalized.npy", "--genre", content / "genre_normalized.npz",
            "--tags", raw / "id_tags.csv", "--information", raw / "id_information.csv", "--output", catalog)
        for surface in ("train_cv", "validation"):
            run(REPO / "scripts/native.py", "run_arrive_full_rcce_control", "framework_export",
                "--project", root, "--dataset", "music4all_onion", "--seed", "2024", "--surface", surface,
                "--index-dir", index, "--base-checkpoint", checkpoints[surface],
                "--base-manifest", checkpoints[surface].parent / "training_manifest.json",
                "--output", root / ("native_" + surface), "--device", "cpu", "--threads", "1")
            run(REPO / "scripts/convert_native_export.py", "--input", root / ("native_" + surface),
                "--output", root / ("export_" + surface), "--catalog", catalog)
        # Exercise the same native fold merger with two toy folds. The paper
        # uses five Yambda folds; this is only an interface/coverage check.
        import numpy as np
        import torch
        from arrive_train_fold_coverage import fold_of_rows
        train_rows, _, assignments = fold_of_rows(data, 2, 20260927)
        folds = root / "folds"
        for fold in range(2):
            directory = folds / f"fold{fold}"
            directory.mkdir(parents=True)
            fold_index = directory / "index"
            shutil.copytree(index, fold_index)
            excluded = train_rows[assignments == fold]
            np.save(fold_index / "cv_rows.npy", excluded)
            im = json.loads((fold_index / "index_manifest.json").read_text())
            im["artifacts"]["cv_rows.npy"]["sha256"] = sha(fold_index / "cv_rows.npy")
            save_json(fold_index / "index_manifest.json", im)
            base = directory / "base"
            base.mkdir()
            bm = json.loads((checkpoints["train_cv"].parent / "training_manifest.json").read_text())
            bm.update(excluded_cv_rows=len(excluded), train_rows=len(train_rows)-len(excluded))
            bm["binding"].update(cv_rows_sha256=sha(fold_index / "cv_rows.npy"),
                index_manifest_sha256=sha(fold_index / "index_manifest.json"),
                full_base_manifest_sha256=sha(checkpoints["validation"].parent / "training_manifest.json"))
            state = torch.load(checkpoints["train_cv"], map_location="cpu", weights_only=False)
            state["binding"] = bm["binding"]
            torch.save(state, base / "final_model.pt")
            bm["final_model_sha256"] = sha(base / "final_model.pt")
            save_json(base / "training_manifest.json", bm)
            run(REPO / "scripts/native.py", "run_arrive_full_rcce_control", "framework_export",
                "--project", root, "--dataset", "music4all_onion", "--seed", "2024", "--surface", "train_cv",
                "--index-dir", fold_index, "--base-checkpoint", base / "final_model.pt",
                "--base-manifest", base / "training_manifest.json", "--output", directory / "export",
                "--device", "cpu", "--threads", "1")
        run(REPO / "scripts/native.py", "arrive_train_fold_coverage", "merge", "--project", root,
            "--dataset", "music4all_onion", "--data-dir", data, "--seed", "2024", "--folds", "2",
            "--assignment-seed", "20260927", "--fold-root", folds, "--storage", "physical",
            "--output", root / "native_fold_merge")
        run(REPO / "scripts/convert_native_export.py", "--input", root / "native_fold_merge",
            "--output", root / "export_fold_merge", "--catalog", catalog, "--prepared-data", data)
        for arm in ("true", "none", "wrong"):
            treatment = ["--catalog", catalog] if arm == "wrong" else []
            run(REPO / "scripts/train_ranker.py", "--train-export", root / "export_fold_merge",
                "--validation-export", root / "export_validation", "--output", root / ("model_" + arm),
                "--cue-match-mode", arm, "--threads", "1", "--smoke", *treatment)
            run(REPO / "scripts/evaluate_ranker.py", "--export", root / "export_validation",
                "--model-dir", root / ("model_" + arm), "--output", root / ("evaluation_" + arm),
                "--threads", "1", *treatment)
        run(REPO / "scripts/replay_coverage.py", "--export", root / "export_validation",
            "--output", root / "coverage")
    print(json.dumps(dict(status="pass", synthetic=True, dataset="toy_onion_layout",
        backbone="randomly_initialized_sasrec_no_optimizer", native_export_surfaces=["train_cv", "validation"],
        ranker_arms=["true", "none", "wrong"], source_replay=True,
        toy_fold_merge=True,
        real_data_accessed=False, gpu_used=False, paper_metric_reproduction=False)))


if __name__ == "__main__":
    main()
