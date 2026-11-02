"""Bind the portable ranker/export interface before any Test feature generation."""
import argparse
import json
from pathlib import Path

from arrive_framework_datasets import (TEST_SCHEMAS, bundle, code_bindings,
    feature_input_paths, feature_schema, hashes, source_policy)
from run_music4all_candidate_support_a1 import digest


def serving_inputs(args):
    ds = bundle(args.project, args.dataset)
    index = args.index_dir or Path(ds["index"])
    values = hashes(feature_input_paths(ds, index, "full"))
    values.update(base_checkpoint=digest(args.base_checkpoint), base_manifest=digest(args.base_manifest))
    return values


def model_identity(args):
    record = json.loads((args.model_dir / "training_manifest.json").read_text())
    expected = dict(schema="arrive.reviewer_ranker.v1", status="pass", dataset=args.dataset,
        seed=args.seed, feature_count=31, feature_schema=feature_schema(), test_accessed=False,
        smoke=False, synthetic=False, cue_match_mode="true")
    if any(record.get(key) != value for key, value in expected.items()):
        raise ValueError("reviewer_freeze_correct_ranker_identity")
    if source_policy(record) != source_policy(vars(args)):
        raise ValueError("reviewer_freeze_ranker_source_policy")
    if digest(args.model_dir / "yeti_ndcg.cbm") != record.get("model_sha256"):
        raise ValueError("reviewer_freeze_model_hash")
    if record.get("selected_trees") not in (220, 440):
        raise ValueError("reviewer_freeze_selected_trees")
    return record


def validate_before_export(args):
    model = model_identity(args)
    record = json.loads(args.freeze_record.read_text())
    expected = dict(schema="arrive.reviewer_freeze.v1", status="frozen", dataset=args.dataset,
        seed=args.seed, model_sha256=model["model_sha256"],
        training_manifest_sha256=digest(args.model_dir / "training_manifest.json"),
        freeze_helper_sha256=digest(Path(__file__)),
        code_sha256=code_bindings(), serving_inputs=serving_inputs(args),
        test_manifest_sha256=digest(args.test_manifest), test_data=str(args.test_data.resolve()),
        retrieval_contract="legacy20k_64_128_512" if args.legacy20k_retrieval else "direct")
    if any(record.get(key) != value for key, value in expected.items()):
        raise ValueError("reviewer_freeze_binding_changed")
    if source_policy(record) != source_policy(vars(args)):
        raise ValueError("reviewer_freeze_source_policy")
    if json.loads(args.test_manifest.read_text()).get("schema") != TEST_SCHEMAS[args.dataset]:
        raise ValueError("reviewer_freeze_test_schema")
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path("."))
    parser.add_argument("--dataset", choices=tuple(TEST_SCHEMAS), required=True)
    parser.add_argument("--seed", type=int, choices=(2024, 2025, 2026), required=True)
    for name in ("model-dir", "validation-export", "base-checkpoint", "base-manifest", "test-data", "test-manifest", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--index-dir", type=Path)
    parser.add_argument("--legacy20k-retrieval", action="store_true")
    parser.add_argument("--source-policy", choices=("full", "joint-removal"), default="full")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    model = model_identity(args)
    validation = json.loads((args.validation_export / "EXPORT_MANIFEST.json").read_text())
    actual = serving_inputs(args)
    if (model["validation_export_sha256"] != digest(args.validation_export / "EXPORT_MANIFEST.json")
            or validation.get("surface") != "validation"
            or source_policy(validation) != source_policy(vars(args))
            or validation.get("native_serving_inputs") != actual
            or validation.get("native_code_sha256") != code_bindings()):
        raise ValueError("reviewer_freeze_validation_serving_binding")
    if json.loads(args.test_manifest.read_text()).get("schema") != TEST_SCHEMAS[args.dataset]:
        raise ValueError("reviewer_freeze_test_schema")
    record = dict(schema="arrive.reviewer_freeze.v1", status="frozen", dataset=args.dataset,
        seed=args.seed, source_policy=source_policy(vars(args)), model_sha256=model["model_sha256"],
        training_manifest_sha256=digest(args.model_dir / "training_manifest.json"),
        freeze_helper_sha256=digest(Path(__file__)),
        code_sha256=code_bindings(), serving_inputs=actual,
        test_manifest_sha256=digest(args.test_manifest), test_data=str(args.test_data.resolve()),
        retrieval_contract="legacy20k_64_128_512" if args.legacy20k_retrieval else "direct",
        test_arrays_accessed=False, scope="reproduction_interface_not_historical_freeze")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(record, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"status": "frozen", "test_arrays_accessed": False}))


if __name__ == "__main__":
    main()
