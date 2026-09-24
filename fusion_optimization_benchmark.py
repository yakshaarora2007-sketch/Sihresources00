"""Bit-exact Fusion comparison and controlled 271-frame benchmark.

The repository contains all 271 saved cell inputs and prediction outputs, but
not the full 271-frame raw scan/pose dataset used by the historical pipeline
timing report.  This harness therefore uses the saved ``cells`` arrays with an
identity relative transform.  It is intended to gate implementation changes
and compare the old and optimized Fusion call chains on identical inputs; the
historical end-to-end timing remains a separate reference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fovmap import dynamics_fusion as fusion  # noqa: E402
from fovmap.dynamics_fusion import (  # noqa: E402
    FUSED_CELL_DTYPE,
    carry_log_odds_through_rebin,
    detect_dynamic_cells,
    fuse_log_odds,
    process_frame,
)
from fovmap.grid_engine import (  # noqa: E402
    CELL_DTYPE,
    RING_RESOLUTIONS,
    assign_rings,
    compute_cell_keys,
    transform_points,
)


DEFAULT_MAP_ROOT = ROOT / "SalsaNext-Fork" / "predictions" / "uncertainty_valid" / "maps" / "sequences" / "08"


def baseline_rebin_stored_map(stored_cells, T_rel):
    """Pre-S5 rebin implementation used as the paired benchmark baseline."""
    if len(stored_cells) == 0:
        return np.array([], dtype=CELL_DTYPE)
    ring_ids = stored_cells["ring_id"].astype(np.int32)
    rows = stored_cells["row"].astype(np.int32)
    cols = stored_cells["col"].astype(np.int32)
    resolutions = RING_RESOLUTIONS[ring_ids]
    centers = np.column_stack((
        (cols.astype(np.float32) + 0.5) * resolutions,
        (rows.astype(np.float32) + 0.5) * resolutions,
        stored_cells["z_mean"],
    ))
    transformed = transform_points(centers, T_rel)
    new_ring_ids = assign_rings(
        np.sqrt(transformed[:, 0] ** 2 + transformed[:, 1] ** 2).astype(np.float32)
    )
    new_keys, _, _ = compute_cell_keys(transformed[:, :2], new_ring_ids)
    unique_keys, inverse_indices = np.unique(new_keys, return_inverse=True)
    point_counts = np.bincount(inverse_indices, weights=stored_cells["point_count"])
    z_sums = np.bincount(
        inverse_indices,
        weights=stored_cells["z_mean"] * stored_cells["point_count"],
    )
    z_means = z_sums / point_counts
    z_mins = np.full_like(unique_keys, np.inf, dtype=np.float32)
    z_maxs = np.full_like(unique_keys, -np.inf, dtype=np.float32)
    np.minimum.at(z_mins, inverse_indices, stored_cells["z_min"])
    np.maximum.at(z_maxs, inverse_indices, stored_cells["z_max"])
    label_sums = np.bincount(
        inverse_indices,
        weights=stored_cells["semantic_label"].astype(np.float32)
        * stored_cells["point_count"],
    )
    cell_semantics = np.round(label_sums / point_counts).astype(np.uint8)
    conf_sums = np.bincount(
        inverse_indices,
        weights=stored_cells["confidence"] * stored_cells["point_count"],
    )
    cell_confidences = conf_sums / point_counts
    timestamps = np.bincount(
        inverse_indices,
        weights=stored_cells["timestamp"].astype(np.float32)
        * stored_cells["point_count"],
    )
    cell_timestamps = np.round(timestamps / point_counts).astype(np.uint32)
    dynamic_flags = np.bincount(
        inverse_indices,
        weights=stored_cells["dynamic_flag"].astype(np.float32)
        * stored_cells["point_count"],
    )
    cell_dynamic = dynamic_flags / point_counts > 0.5
    rebinned = np.empty(len(unique_keys), dtype=CELL_DTYPE)
    rebinned["cell_key"] = unique_keys
    rebinned["ring_id"] = ((unique_keys >> 48) & 0xFFFF).astype(np.uint8)
    rebinned["row"] = (((unique_keys >> 24) & 0xFFFFFF) - (1 << 23)).astype(np.int32)
    rebinned["col"] = ((unique_keys & 0xFFFFFF) - (1 << 23)).astype(np.int32)
    rebinned["point_count"] = point_counts.astype(np.uint32)
    rebinned["z_mean"] = z_means.astype(np.float32)
    rebinned["z_min"] = z_mins
    rebinned["z_max"] = z_maxs
    rebinned["semantic_label"] = cell_semantics
    rebinned["confidence"] = cell_confidences.astype(np.float32)
    rebinned["dynamic_flag"] = cell_dynamic
    rebinned["timestamp"] = cell_timestamps
    return rebinned


def baseline_process_frame(current_cells, stored_cells, transform):
    """The pre-S5/S7 candidate call chain, kept local for gating."""
    rebinned = baseline_rebin_stored_map(stored_cells, transform)
    rebinned = carry_log_odds_through_rebin(stored_cells, rebinned, transform)
    dynamic = detect_dynamic_cells(current_cells, rebinned)
    fused = fuse_log_odds(current_cells, rebinned, dynamic)
    return fused, dynamic, rebinned


def frame_paths(map_root: Path):
    paths = sorted(map_root.glob("*.npz"))
    paths = [path for path in paths if path.name != "metadata.json"]
    if not paths:
        raise FileNotFoundError(f"No .npz snapshots found under {map_root}")
    return paths


def load_cells(path: Path):
    with np.load(path, allow_pickle=False) as snapshot:
        return snapshot["cells"]


def load_frames(paths):
    return [load_cells(path) for path in paths]


def _rigid_transform(frame: int) -> np.ndarray:
    """Deterministic two-step composed non-identity transform."""
    def step(yaw, tx, ty):
        c, s = np.cos(yaw), np.sin(yaw)
        result = np.eye(4, dtype=np.float32)
        result[:2, :2] = np.array([[c, -s], [s, c]], dtype=np.float32)
        result[0, 3] = tx
        result[1, 3] = ty
        return result

    first = step(
        0.0025 * np.sin((frame + 1) * 0.07),
        0.31 + 0.02 * np.sin(frame * 0.11),
        -0.18 + 0.015 * np.cos(frame * 0.09),
    )
    second = step(
        -0.0014 * np.cos((frame + 1) * 0.05),
        -0.09 + 0.01 * np.cos(frame * 0.13),
        0.06 + 0.008 * np.sin(frame * 0.08),
    )
    return (second @ first).astype(np.float32)


def transforms_for_regime(frame_count: int, regime: str):
    if regime == "identity":
        return [np.eye(4, dtype=np.float32) for _ in range(frame_count)]
    if regime == "nonidentity":
        return [_rigid_transform(frame) for frame in range(frame_count)]
    raise ValueError(f"unknown regime: {regime}")


def digest(array: np.ndarray) -> str:
    hasher = hashlib.sha256()
    hasher.update(str(array.dtype).encode())
    hasher.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    hasher.update(array.tobytes(order="C"))
    return hasher.hexdigest()


def compare_array(left: np.ndarray, right: np.ndarray, name: str, frame: int):
    if left.dtype != right.dtype:
        return f"frame={frame} output={name} dtype {left.dtype!r} != {right.dtype!r}"
    if left.shape != right.shape:
        return f"frame={frame} output={name} shape {left.shape} != {right.shape}"
    if left.dtype.names:
        for field in left.dtype.names:
            if not np.array_equal(left[field], right[field], equal_nan=True):
                mismatch = np.flatnonzero(left[field] != right[field])
                location = int(mismatch[0]) if len(mismatch) else "nan"
                return (
                    f"frame={frame} output={name} field={field} "
                    f"mismatch_count={int(np.count_nonzero(left[field] != right[field]))} "
                    f"first={location}"
                )
    elif not np.array_equal(left, right, equal_nan=True):
        mismatch = np.flatnonzero(left != right)
        location = int(mismatch[0]) if len(mismatch) else "nan"
        return (
            f"frame={frame} output={name} mismatch_count={int(np.count_nonzero(left != right))} "
            f"first={location}"
        )
    return None


def compare_triplet(baseline, candidate, frame: int):
    names = ("fused_cells", "dynamic_mask", "rebinned_prior")
    for name, left, right in zip(names, baseline, candidate):
        mismatch = compare_array(left, right, name, frame)
        if mismatch:
            return mismatch
    return None


def stats(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean_ms": float(values.mean()),
        "min_ms": float(values.min()),
        "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
        "max_ms": float(values.max()),
        "total_ms": float(values.sum()),
    }


def process_memory_mb():
    try:
        import psutil

        return float(psutil.Process(os.getpid()).memory_info().rss / 1048576.0)
    except Exception:
        return None


def _matching_for_outputs(current_cells, rebinned):
    current_keys = current_cells["cell_key"]
    prior_keys = rebinned["cell_key"]
    if fusion._is_sorted_keys(current_keys) and fusion._is_sorted_keys(prior_keys):
        return fusion._matching_indices_sorted(current_keys, prior_keys)
    return fusion._matching_indices(current_keys, prior_keys)


def frame_size_record(current_cells, stored_cells, transform, rebinned):
    H = len(stored_cells)
    N = len(current_cells)
    record = {"N": N, "H": H, "R": 0, "M": 0, "collisions": 0, "R_over_H": None}
    if H == 0:
        return record
    ring_ids = stored_cells["ring_id"].astype(np.int32)
    resolutions = RING_RESOLUTIONS[ring_ids]
    centers = np.column_stack((
        (stored_cells["col"].astype(np.float32) + 0.5) * resolutions,
        (stored_cells["row"].astype(np.float32) + 0.5) * resolutions,
        stored_cells["z_mean"],
    ))
    transformed = transform_points(centers, transform)
    new_ring_ids = assign_rings(
        np.hypot(transformed[:, 0], transformed[:, 1]).astype(np.float32)
    )
    new_keys, _, _ = compute_cell_keys(transformed[:, :2], new_ring_ids)
    unique_keys = np.unique(new_keys)
    record["R"] = len(unique_keys)
    record["collisions"] = H - len(unique_keys)
    record["M"] = int(np.count_nonzero(_matching_for_outputs(current_cells, rebinned) >= 0))
    record["R_over_H"] = float(len(unique_keys) / H)
    return record


def key_stability_record(stored_cells, transform):
    if len(stored_cells) == 0:
        return {str(ring): None for ring in range(4)}
    ring_ids = stored_cells["ring_id"].astype(np.int32)
    resolutions = RING_RESOLUTIONS[ring_ids]
    centers = np.column_stack((
        (stored_cells["col"].astype(np.float32) + 0.5) * resolutions,
        (stored_cells["row"].astype(np.float32) + 0.5) * resolutions,
        stored_cells["z_mean"],
    ))
    transformed = transform_points(centers, transform)
    new_ring_ids = assign_rings(
        np.hypot(transformed[:, 0], transformed[:, 1]).astype(np.float32)
    )
    new_keys, _, _ = compute_cell_keys(transformed[:, :2], new_ring_ids)
    stable = new_keys == stored_cells["cell_key"]
    result = {}
    for ring in range(4):
        mask = ring_ids == ring
        count = int(np.count_nonzero(mask))
        if count == 0:
            result[str(ring)] = None
            continue
        values = stable[mask]
        runs = []
        if len(values):
            starts = np.r_[0, np.flatnonzero(values[1:] != values[:-1]) + 1]
            ends = np.r_[starts[1:], len(values)]
            runs = (ends - starts).astype(int).tolist()
        result[str(ring)] = {
            "count": count,
            "stable_count": int(np.count_nonzero(values)),
            "stable_fraction": float(np.mean(values)),
            "all_stable": bool(np.all(values)),
            "run_lengths": {
                "count": len(runs),
                "values": runs,
                "min": int(min(runs)) if runs else 0,
                "p50": float(np.percentile(runs, 50)) if runs else 0.0,
                "p95": float(np.percentile(runs, 95)) if runs else 0.0,
                "max": int(max(runs)) if runs else 0,
            },
        }
    return result


def run_open_loop(frames, transforms):
    """Compare each candidate step with the baseline state for that step."""
    stored = np.empty(0, dtype=CELL_DTYPE)
    first_mismatch = None
    checksums = []
    trajectory = []
    for frame, (cells, transform) in enumerate(zip(frames, transforms)):
        baseline = baseline_process_frame(cells, stored, transform)
        candidate = process_frame(cells, stored, transform)
        if first_mismatch is None:
            first_mismatch = compare_triplet(baseline, candidate, frame)
        checksums.append({name: digest(value) for name, value in zip(
            ("fused_cells", "dynamic_mask", "rebinned_prior"), candidate
        )})
        stored = baseline[0]
        trajectory.append(len(stored))
    return {
        "pass": first_mismatch is None,
        "first_mismatch": first_mismatch,
        "checksums": checksums,
        "stored_cells": {
            "min": int(min(trajectory)),
            "max": int(max(trajectory)),
            "mean": float(np.mean(trajectory)),
            "last": int(trajectory[-1]),
        },
    }


def run_closed_loop(frames, transforms):
    baseline_stored = np.empty(0, dtype=CELL_DTYPE)
    candidate_stored = np.empty(0, dtype=CELL_DTYPE)
    first_mismatch = None
    for frame, (cells, transform) in enumerate(zip(frames, transforms)):
        baseline = baseline_process_frame(cells, baseline_stored, transform)
        candidate = process_frame(cells, candidate_stored, transform)
        if first_mismatch is None:
            first_mismatch = compare_triplet(baseline, candidate, frame)
        baseline_stored = baseline[0]
        candidate_stored = candidate[0]
    return {"pass": first_mismatch is None, "first_mismatch": first_mismatch}


def benchmark(frames, transforms, repeats):
    results = {}
    timings_by_arm = {}
    for label, function in (("baseline", baseline_process_frame), ("candidate", process_frame)):
        repeat_stats = []
        repeat_timings = []
        last_trajectory = []
        for _ in range(repeats):
            stored = np.empty(0, dtype=CELL_DTYPE)
            timings = []
            trajectory = []
            for cells, transform in zip(frames, transforms):
                start = time.perf_counter()
                fused, _, _ = function(cells, stored, transform)
                timings.append((time.perf_counter() - start) * 1000.0)
                stored = fused
                trajectory.append(len(stored))
            repeat_stats.append(stats(timings))
            repeat_timings.append(timings)
            last_trajectory = trajectory
        timings_by_arm[label] = repeat_timings
        results[label] = {
            "repeats": repeat_stats,
            "mean_of_repeat_means_ms": float(np.mean([item["mean_ms"] for item in repeat_stats])),
            "best_mean_ms": float(min(item["mean_ms"] for item in repeat_stats)),
            "trajectory": {
                "min": int(min(last_trajectory)),
                "max": int(max(last_trajectory)),
                "mean": float(np.mean(last_trajectory)),
                "last": int(last_trajectory[-1]),
            },
            "rss_after_mb": process_memory_mb(),
        }
    paired = np.concatenate([
        np.asarray(base) - np.asarray(candidate)
        for base, candidate in zip(
            timings_by_arm["baseline"], timings_by_arm["candidate"]
        )
    ])
    results["paired_delta_baseline_minus_candidate"] = {
        "mean_ms": float(np.mean(paired)),
        "p50_ms": float(np.percentile(paired, 50)),
        "p95_ms": float(np.percentile(paired, 95)),
        "positive_fraction": float(np.mean(paired > 0)),
    }
    return results


def profile_candidate_frame(current_cells, stored_cells, transform):
    """Profile the candidate's semantic phases without changing its algorithm."""
    timings = {}
    start = time.perf_counter()
    rebinned = fusion._rebin_with_log_odds(stored_cells, transform)
    timings["rebin"] = (time.perf_counter() - start) * 1000.0

    start = time.perf_counter()
    prior_indices = _matching_for_outputs(current_cells, rebinned)
    timings["matching"] = (time.perf_counter() - start) * 1000.0

    start = time.perf_counter()
    dynamic_mask = detect_dynamic_cells(
        current_cells, rebinned, prior_indices=prior_indices
    )
    timings["dynamic_detection"] = (time.perf_counter() - start) * 1000.0

    start = time.perf_counter()
    prior = fusion._as_fused_dtype(rebinned)
    current = fusion._as_fused_dtype(current_cells)
    current["dynamic_flag"] = dynamic_mask
    timings["output_conversion"] = (time.perf_counter() - start) * 1000.0

    matched = prior_indices >= 0
    safe_indices = np.maximum(prior_indices, 0)
    current_log_odds = current["log_odds"]
    current_log_odds[:] = 0.0
    static_observations = ~dynamic_mask
    start = time.perf_counter()
    current_log_odds[static_observations & matched] = (
        np.float32(0.98) * prior["log_odds"][safe_indices[static_observations & matched]]
        + fusion._measurement_log_odds(current)[static_observations & matched]
    )
    current_log_odds[static_observations & ~matched] = fusion._measurement_log_odds(current)[
        static_observations & ~matched
    ]
    current_log_odds[dynamic_mask & matched] = prior["log_odds"][
        safe_indices[dynamic_mask & matched]
    ]
    timings["measurement_log_odds"] = (time.perf_counter() - start) * 1000.0

    start = time.perf_counter()
    matched_prior = np.zeros(len(prior), dtype=bool)
    matched_prior[prior_indices[matched]] = True
    carried = prior[~matched_prior].copy()
    carried["log_odds"] *= np.float32(0.98)
    fused = np.empty(len(current) + len(carried), dtype=FUSED_CELL_DTYPE)
    fused[:len(current)] = current
    fused[len(current):] = carried
    resolutions = np.choose(
        np.minimum(fused["ring_id"].astype(np.int64), 3),
        fusion._PRUNE_RESOLUTIONS,
    )
    x_centers = (fused["col"].astype(np.float32) + 0.5) * resolutions
    y_centers = (fused["row"].astype(np.float32) + 0.5) * resolutions
    distances = np.hypot(x_centers, y_centers)
    fused = fused[distances <= np.float32(100.0)]
    timings["assembly_prune"] = (time.perf_counter() - start) * 1000.0
    timings["total"] = sum(timings.values())
    return fused, dynamic_mask, rebinned, timings


def profile_candidate(frames, transforms):
    stored = np.empty(0, dtype=CELL_DTYPE)
    names = (
        "rebin", "matching", "dynamic_detection", "output_conversion",
        "measurement_log_odds", "assembly_prune", "total",
    )
    totals = {name: [] for name in names}
    sizes = []
    stability = {str(ring): [] for ring in range(4)}
    for cells, transform in zip(frames, transforms):
        fused, _, rebinned, timings = profile_candidate_frame(cells, stored, transform)
        for name in names:
            totals[name].append(timings[name])
        sizes.append(frame_size_record(cells, stored, transform, rebinned))
        record = key_stability_record(stored, transform)
        for ring in range(4):
            if record[str(ring)] is not None:
                stability[str(ring)].append(record[str(ring)])
        stored = fused

    attribution = {}
    total_mean = float(np.mean(totals["total"]))
    for name in names[:-1]:
        mean_ms = float(np.mean(totals[name]))
        attribution[name] = {
            "mean_ms": mean_ms,
            "p50_ms": float(np.percentile(totals[name], 50)),
            "p95_ms": float(np.percentile(totals[name], 95)),
            "share_of_profiled_total": float(mean_ms / total_mean) if total_mean else 0.0,
        }
    stability_summary = {}
    for ring, records in stability.items():
        fractions = [item["stable_fraction"] for item in records]
        all_stable = [item["all_stable"] for item in records]
        run_counts = [item["run_lengths"]["count"] for item in records]
        run_lengths = [
            length
            for item in records
            for length in item["run_lengths"]["values"]
        ]
        stability_summary[ring] = {
            "eligible_frames": len(records),
            "stable_fraction_mean": float(np.mean(fractions)) if fractions else None,
            "stable_fraction_p50": float(np.percentile(fractions, 50)) if fractions else None,
            "all_stable_event_rate": float(np.mean(all_stable)) if all_stable else None,
            "run_count_mean": float(np.mean(run_counts)) if run_counts else None,
            "run_length_distribution": {
                "count": len(run_lengths),
                "min": int(min(run_lengths)) if run_lengths else 0,
                "p50": float(np.percentile(run_lengths, 50)) if run_lengths else 0.0,
                "p95": float(np.percentile(run_lengths, 95)) if run_lengths else 0.0,
                "max": int(max(run_lengths)) if run_lengths else 0,
            },
        }
    return {
        "attribution": attribution,
        "frame_sizes": sizes,
        "size_summary": {
            "N_min": int(min(item["N"] for item in sizes)),
            "N_max": int(max(item["N"] for item in sizes)),
            "H_min": int(min(item["H"] for item in sizes)),
            "H_max": int(max(item["H"] for item in sizes)),
            "R_min": int(min(item["R"] for item in sizes)),
            "R_max": int(max(item["R"] for item in sizes)),
            "M_min": int(min(item["M"] for item in sizes)),
            "M_max": int(max(item["M"] for item in sizes)),
            "collision_max": int(max(item["collisions"] for item in sizes)),
            "collision_mean": float(np.mean([item["collisions"] for item in sizes])),
            "R_over_H_mean": float(np.mean([
                item["R_over_H"] for item in sizes if item["R_over_H"] is not None
            ])),
        },
        "key_stability": stability_summary,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--map-root", type=Path, default=DEFAULT_MAP_ROOT)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, default=ROOT / "fusion_optimization_report.json")
    parser.add_argument("--benchmark-only", action="store_true")
    parser.add_argument("--regime", choices=("identity", "nonidentity", "both"), default="both")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--profile-only", action="store_true")
    args = parser.parse_args()
    paths = frame_paths(args.map_root)
    if args.max_frames is not None:
        paths = paths[:args.max_frames]
    frames = load_frames(paths)
    if args.repeats < 1:
        raise ValueError("--repeats must be positive")

    if args.profile_only:
        regimes = ("nonidentity",)
    else:
        regimes = ("identity", "nonidentity") if args.regime == "both" else (args.regime,)
    report = {
        "scope": {
            "frames": len(paths),
            "regimes": list(regimes),
            "inputs": str(args.map_root),
            "note": "Controlled Fusion benchmark; historical full-pipeline report used raw scans and poses unavailable in this checkout. Identity is correctness/comparative only; absolute timing claims use nonidentity.",
            "numpy": np.__version__,
        },
        "regimes": {},
    }
    for regime in regimes:
        transforms = transforms_for_regime(len(frames), regime)
        item = {
            "transform": regime,
            "open_loop": None if (args.benchmark_only or args.profile_only) else run_open_loop(frames, transforms),
            "closed_loop": None if (args.benchmark_only or args.profile_only) else run_closed_loop(frames, transforms),
            "benchmark": None if args.profile_only else benchmark(frames, transforms, args.repeats),
        }
        if regime == "nonidentity":
            item["g0_profile"] = profile_candidate(frames, transforms)
        report["regimes"][regime] = item
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    summary = {"frames": report["scope"]["frames"], "report": str(args.output)}
    for regime in regimes:
        item = report["regimes"][regime]
        summary[regime] = {}
        if not args.profile_only:
            summary[regime]["baseline"] = item["benchmark"]["baseline"]["mean_of_repeat_means_ms"]
            summary[regime]["candidate"] = item["benchmark"]["candidate"]["mean_of_repeat_means_ms"]
        if not args.benchmark_only and not args.profile_only:
            summary[regime]["open_loop"] = item["open_loop"]["pass"]
            summary[regime]["closed_loop"] = item["closed_loop"]["pass"]
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
