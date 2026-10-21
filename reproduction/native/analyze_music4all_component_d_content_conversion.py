#!/usr/bin/env python3
"""Train-only OOF gate for content-aware conversion inside frozen Top-200."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
from scipy import sparse

from analyze_music4all_component_d_selective_exchange import (
    SCOPES, atomic_json, candidate_lookup, paired_ci, row_metrics, user_macro,
)
from run_music4all_candidate_support_a0 import recent_anchors


SCHEMA = "genrec.music4all_onion.component_d_content_conversion_oof.v1"


def split_calibration(users: np.ndarray, fold: int) -> np.ndarray:
    return np.asarray([
        int.from_bytes(hashlib.sha256(f"d-content:{fold}:{int(user)}".encode()).digest()[:8], "big") % 5 == 0
        for user in users
    ], dtype=bool)


def source_contexts(pred: dict[str, np.ndarray], data_dir: Path, audio: np.ndarray, genre: sparse.csr_matrix):
    events = np.load(data_dir / "events.npy", mmap_mode="r")
    sessions = np.load(data_dir / "sessions.npy", mmap_mode="r")
    audio_obs = np.zeros((len(pred["row_ids"]), audio.shape[1]), dtype=np.float32)
    audio_hist = np.zeros_like(audio_obs)
    genre_obs: list[sparse.csr_matrix] = []
    genre_hist: list[sparse.csr_matrix] = []
    for at, row_id in enumerate(pred["row_ids"]):
        session = sessions[int(row_id)]
        observed = int(events[int(session[2])])
        anchors = recent_anchors(events, session, 5)
        audio_obs[at] = audio[observed]
        audio_hist[at] = np.asarray(audio[anchors]).mean(axis=0) if anchors else 0
        genre_obs.append(genre.getrow(observed))
        genre_hist.append(sparse.csr_matrix(genre[anchors].mean(axis=0)) if anchors else sparse.csr_matrix((1, genre.shape[1])))
    return audio_obs, audio_hist, genre_obs, genre_hist


def derangement(size: int) -> np.ndarray:
    order = np.argsort(np.asarray([int.from_bytes(hashlib.sha256(f"d-content-shuffle:{i}".encode()).digest()[:8], "big") for i in range(size)], dtype=np.uint64))
    source = np.empty(size, dtype=np.int64)
    source[order] = np.roll(order, 1)
    if np.any(source == np.arange(size)):
        raise ValueError("content_derangement_failed")
    return source


def build_features(
    pred: dict[str, np.ndarray], c2: dict[str, np.ndarray], raw: dict[str, np.ndarray],
    audio: np.ndarray, genre: sparse.csr_matrix, contexts, content_source: np.ndarray,
    depth: int,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    audio_obs, audio_hist, genre_obs, genre_hist = contexts
    pools, matrices, labels = [], [], []
    for row in range(len(pred["row_ids"])):
        order = np.lexsort((pred["candidates"][row], -pred["scores"][row]))[:depth]
        pool = pred["candidates"][row, order].astype(np.int32)
        source = int(content_source[row])
        candidates_audio = np.asarray(audio[pool])
        a_obs = candidates_audio @ audio_obs[source]
        a_hist = candidates_audio @ audio_hist[source]
        g_obs = np.asarray(genre[pool].dot(genre_obs[source].T).toarray()).ravel()
        g_hist = np.asarray(genre[pool].dot(genre_hist[source].T).toarray()).ravel()
        score_values = pred["scores"][row, order]
        score_mean, score_std = float(score_values.mean()), max(float(score_values.std()), 1e-6)
        c2_list = list(map(int, c2["method_top10"][row]))
        truth = set(map(int, pred["targets"][row, : int(pred["target_counts"][row])]))
        matrix = np.zeros((depth, 17), dtype=np.float32)
        for at, item in enumerate(map(int, pool)):
            base_rank, x_rank, h_rank, popularity = candidate_lookup(raw, row, item)
            c2_rank = c2_list.index(item) + 1 if item in c2_list else 11
            matrix[at] = (
                (float(score_values[at]) - score_mean) / score_std,
                1.0 / math.log2(at + 2.0),
                1.0 / math.log2(base_rank + 2.0),
                1.0 / math.log2(x_rank + 2.0),
                1.0 / math.log2(h_rank + 2.0),
                popularity,
                math.log1p(float(pred["gaps"][row])) / 8.0,
                float(pred["bucket_ids"][row] == 0),
                float(pred["bucket_ids"][row] == 1),
                float(pred["bucket_ids"][row] == 2),
                float(pred["bucket_ids"][row] == 3),
                float(item in c2_list),
                1.0 / c2_rank,
                float(a_obs[at]), float(a_hist[at]), float(g_obs[at]), float(g_hist[at]),
            )
        pools.append(pool)
        matrices.append(matrix)
        labels.append(np.asarray([int(int(item) in truth) for item in pool], dtype=np.uint8))
    return pools, matrices, labels


def table(matrices: list[np.ndarray], labels: list[np.ndarray], selected: np.ndarray, columns: slice | list[int]):
    rows = np.flatnonzero(selected)
    x = np.concatenate([matrices[row][:, columns] for row in rows])
    y = np.concatenate([labels[row] for row in rows])
    weights = np.concatenate([np.full(len(labels[row]), 1.0 / len(labels[row]), dtype=np.float32) for row in rows])
    positives = int(y.sum())
    weights[y == 1] *= min(max((len(y) - positives) / max(positives, 1), 1.0), 40.0)
    return x, y, weights


def train(x: np.ndarray, y: np.ndarray, w: np.ndarray, seed: int) -> lgb.LGBMClassifier:
    model = lgb.LGBMClassifier(
        objective="binary", n_estimators=180, num_leaves=31, learning_rate=.04,
        min_child_samples=100, subsample=.8, subsample_freq=1, colsample_bytree=.9,
        reg_lambda=1.0, random_state=seed, n_jobs=8, verbosity=-1,
    )
    model.fit(x, y, sample_weight=w)
    return model


def probabilities(model: lgb.LGBMClassifier, matrices: list[np.ndarray], selected: np.ndarray, columns: slice | list[int]):
    output = [np.empty(0, dtype=np.float32) for _ in matrices]
    for row in np.flatnonzero(selected):
        output[row] = model.booster_.predict(matrices[row][:, columns], num_threads=8).astype(np.float32)
    return output


def exchange(pools, probabilities, c2_top10, selected, quota: int, margin: float):
    output = c2_top10.copy()
    for row in np.flatnonzero(selected):
        value = {int(item): float(probabilities[row][at]) for at, item in enumerate(pools[row])}
        current = list(map(int, c2_top10[row]))
        incoming = [int(item) for item in pools[row] if int(item) not in current]
        incoming.sort(key=lambda item: (-value[item], item))
        outgoing = sorted(current, key=lambda item: (value.get(item, -1.0), item))
        used = 0
        for add, remove in zip(incoming, outgoing):
            if used >= quota or value[add] - value.get(remove, -1.0) <= margin:
                continue
            current[current.index(remove)] = add
            used += 1
        output[row] = np.asarray(current, dtype=np.int32)
    return output


def choose(model, matrices, pools, c2, pred, selected, columns):
    probs = probabilities(model, matrices, selected, columns)
    base_r, base_n = row_metrics(c2["method_top10"], pred["targets"], pred["target_counts"])
    choices = []
    for quota in (0, 1, 2, 5, 10):
        for margin in (0.0, .02, .05, .10, .20):
            out = exchange(pools, probs, c2["method_top10"], selected, quota, margin)
            r, n = row_metrics(out, pred["targets"], pred["target_counts"])
            long = selected & (pred["bucket_ids"] == 3)
            values = {
                "overall_recall_delta": user_macro(r - base_r, pred["users"], selected),
                "overall_ndcg_delta": user_macro(n - base_n, pred["users"], selected),
                "long_recall_delta": user_macro(r - base_r, pred["users"], long),
                "long_ndcg_delta": user_macro(n - base_n, pred["users"], long),
            }
            choices.append({"quota": quota, "margin": margin, "objective": sum(values.values()), **values})
    safe = [row for row in choices if row["long_recall_delta"] >= 0 and row["long_ndcg_delta"] >= 0]
    return max(safe, key=lambda row: (row["objective"], -row["quota"], row["margin"]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predecessor", type=Path, default=Path("runs/music4all_component_c/oof_predecessor/oof_predecessor.npz"))
    parser.add_argument("--c2", type=Path, default=Path("refine-logs/music4all_onion/component_c2/remote_evidence/runs/music4all_component_c2/oof_set_contextual_formal_s2024/oof_predictions.npz"))
    parser.add_argument("--features", type=Path, default=Path("runs/music4all_artist_b/features/train_cv_features.npz"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/music4all_onion/prepared_v1"))
    parser.add_argument("--content-dir", type=Path, default=Path("data/music4all_onion/component_b2_attainability_v1"))
    parser.add_argument("--output", type=Path, default=Path("refine-logs/music4all_onion/component_d/CONTENT_CONVERSION_OOF_RESULT.json"))
    parser.add_argument("--depth", type=int, default=50)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("content_conversion_output_exists")
    pred = dict(np.load(args.predecessor, allow_pickle=False))
    c2 = dict(np.load(args.c2, allow_pickle=False))
    raw = dict(np.load(args.features, allow_pickle=False))
    audio = np.load(args.content_dir / "musicnn_normalized.npy", mmap_mode="r")
    genre = sparse.load_npz(args.content_dir / "genre_normalized.npz").tocsr()
    contexts = source_contexts(pred, args.data_dir, audio, genre)
    real_source = np.arange(len(pred["row_ids"]), dtype=np.int64)
    shuffled_source = derangement(len(real_source))
    real = build_features(pred, c2, raw, audio, genre, contexts, real_source, args.depth)
    shuffled = build_features(pred, c2, raw, audio, genre, contexts, shuffled_source, args.depth)
    arms = {
        "behavior_only": (real, list(range(13))),
        "real_content": (real, slice(None)),
        "shuffled_content": (shuffled, slice(None)),
    }
    outputs = {name: c2["method_top10"].copy() for name in arms}
    records = {name: [] for name in arms}
    for fold in sorted(map(int, np.unique(pred["fold_ids"]))):
        evaluation = pred["fold_ids"] == fold
        cal = (~evaluation) & split_calibration(pred["users"], fold)
        fit = (~evaluation) & (~cal)
        for arm_index, (name, ((pools, matrices, labels), columns)) in enumerate(arms.items()):
            x, y, w = table(matrices, labels, fit, columns)
            model = train(x, y, w, 15000 + fold * 100 + arm_index)
            selection = choose(model, matrices, pools, c2, pred, cal, columns)
            probs = probabilities(model, matrices, evaluation, columns)
            prediction = exchange(pools, probs, c2["method_top10"], evaluation, int(selection["quota"]), float(selection["margin"]))
            outputs[name][evaluation] = prediction[evaluation]
            records[name].append({"fold": fold, "selection": selection})
            print(json.dumps({"event": "content_conversion_fold", "fold": fold, "arm": name, "selection": selection}), flush=True)
    base_r, base_n = row_metrics(c2["method_top10"], pred["targets"], pred["target_counts"])
    results: dict[str, Any] = {}
    for arm_index, (name, output) in enumerate(outputs.items()):
        r, n = row_metrics(output, pred["targets"], pred["target_counts"])
        metrics = {}
        for scope_index, scope in enumerate(SCOPES):
            selected = np.ones(len(r), dtype=bool) if scope == "overall" else pred["bucket_ids"] == scope_index - 1
            metrics[scope] = {
                "recall_paired": paired_ci(r - base_r, pred["users"], selected, 16000 + arm_index * 100 + scope_index),
                "ndcg_paired": paired_ci(n - base_n, pred["users"], selected, 17000 + arm_index * 100 + scope_index),
            }
        results[name] = {"metrics": metrics, "fold_records": records[name], "better_rows": int((r > base_r).sum()), "worse_rows": int((r < base_r).sum())}
    proposed, plain, shuffled_control = results["real_content"], results["behavior_only"], results["shuffled_content"]
    p_overall, p_long = proposed["metrics"]["overall"], proposed["metrics"][">365d"]
    content_increment = p_overall["recall_paired"]["delta"] > plain["metrics"]["overall"]["recall_paired"]["delta"] and p_overall["recall_paired"]["delta"] > shuffled_control["metrics"]["overall"]["recall_paired"]["delta"]
    passed = p_overall["recall_paired"]["ci95_low"] > 0 and p_overall["ndcg_paired"]["ci95_low"] > 0 and p_long["recall_paired"]["delta"] >= 0 and p_long["ndcg_paired"]["delta"] >= 0 and proposed["better_rows"] > proposed["worse_rows"] and content_increment
    result = {"schema": SCHEMA, "status": "pass", "decision": "CONTENT_CONVERSION_OOF_PASS" if passed else "CONTENT_CONVERSION_OOF_STOP", "surface": "Train-only nested OOF", "depth": args.depth, "results": results, "content_increment_present": content_increment, "bindings": {"validation_accessed": False, "test_accessed": False}}
    atomic_json(args.output, result)
    print(json.dumps({"decision": result["decision"], "content_increment_present": content_increment, "results": results}), flush=True)


if __name__ == "__main__":
    main()
