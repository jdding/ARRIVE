#!/usr/bin/env python3
"""Plan, or explicitly execute, a backbone run with its recorded settings."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("music4all_onion", "yambda500m"), required=True)
    parser.add_argument("--backbone", choices=("sasrec", "ssd4rec"), required=True)
    parser.add_argument("--seed", type=int, choices=(2024, 2025, 2026), required=True)
    for name in ("data-dir", "prepared-manifest", "labels-manifest", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("event-times", "timestamp-manifest"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--ssd4rec-source", type=Path, default=Path("vendor/ssd4rec_official/ssd4rec.py"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--execute", action="store_true", help="Without this flag only print the command.")
    args = parser.parse_args()
    records = json.loads((REPO / "reproduction/configs/backbone_training.json").read_text())["records"]
    record = next(r for r in records if (r["dataset"], r["backbone"], r["seed"]) == (args.dataset, args.backbone, args.seed))
    config = record["configuration"]
    module = "run_music4all_baselines" if args.backbone == "sasrec" else "run_arrive_main_table_baseline"
    command = [sys.executable, str(REPO / "scripts/native.py"), module]
    if args.backbone == "sasrec":
        command.append("train")
    command += [
        "--data-dir", str(args.data_dir), "--manifest", str(args.prepared_manifest),
        "--labels-manifest", str(args.labels_manifest), "--output", str(args.output),
        "--device", args.device, "--threads", str(args.threads)]
    for name in ("seed", "batch_size", "dropout", "embed_dim", "epochs", "horizon", "learning_rate",
                 "max_context", "negative_samples", "num_heads", "num_layers", "objective"):
        command.extend(["--" + name.replace("_", "-"), str(config[name])])
    policy = record.get("convergence_policy") or {}
    for name in ("min_epochs", "patience", "min_delta"):
        if name in policy:
            command.extend(["--" + name.replace("_", "-"), str(policy[name])])
    if args.backbone == "sasrec":
        command.extend(["--baseline", "sasrec_standard"])
    else:
        if args.event_times is None or args.timestamp_manifest is None:
            parser.error("SSD4Rec requires --event-times and --timestamp-manifest")
        command.extend(["--method", "ssd4rec", "--dataset-config", str(REPO / "reproduction/configs/music4all_repair_20260912.json")
            if args.dataset == "music4all_onion" else str(REPO / "reproduction/configs/yambda_500m_v1.json"),
            "--event-times", str(args.event_times), "--timestamp-manifest", str(args.timestamp_manifest),
            "--ssd4rec-source", str(args.ssd4rec_source)])
        optimizer = record.get("optimizer") or {}
        for key in ("optimizer", "beta2", "weight_decay"):
            if key in optimizer:
                command.extend(["--" + key.replace("_", "-"), str(optimizer[key])])
    print(json.dumps({"dataset": args.dataset, "backbone": args.backbone, "seed": args.seed,
                      "argv": command, "execute": args.execute}, indent=2), flush=True)
    if args.execute:
        if args.output.exists():
            raise FileExistsError(args.output)
        if args.threads < 1:
            parser.error("positive --threads required")
        raise SystemExit(subprocess.run(command).returncode)


if __name__ == "__main__":
    main()
