"""Benchmark the experimental PyTorch CUDA Fusion path against NumPy Fusion."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
SIH_ROOT = ROOT
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from fovmap.dynamics_fusion import process_frame  # noqa: E402
from fovmap.gpu_dynamics_fusion import process_frame_cuda  # noqa: E402
from fovmap.grid_engine import CELL_DTYPE  # noqa: E402
from fusion_optimization_benchmark import (  # noqa: E402
    _rigid_transform,
    frame_paths,
    load_frames,
)


MAP_ROOT = ROOT / "SalsaNext-Fork" / "predictions" / "uncertainty_valid" / "maps" / "sequences" / "08"


def transforms(count, regime):
    if regime == "identity":
        return [np.eye(4, dtype=np.float32) for _ in range(count)]
    return [_rigid_transform(index) for index in range(count)]


def compare_array(left, right):
    result = {
        "dtype_equal": str(left.dtype) == str(right.dtype),
        "shape_equal": left.shape == right.shape,
        "exact": False,
        "mismatch_count": None,
        "max_abs_float_delta": 0.0,
    }
    if not result["dtype_equal"] or not result["shape_equal"]:
        return result
    if left.dtype.names:
        mismatches = 0
        max_delta = 0.0
        for field in left.dtype.names:
            equal = np.array_equal(left[field], right[field], equal_nan=True)
            if not equal:
                mismatches += int(np.count_nonzero(left[field] != right[field]))
                if np.issubdtype(left[field].dtype, np.floating):
                    delta = np.nanmax(np.abs(left[field].astype(np.float64) - right[field].astype(np.float64)))
                    max_delta = max(max_delta, float(delta))
        result["mismatch_count"] = mismatches
        result["max_abs_float_delta"] = max_delta
        result["exact"] = mismatches == 0
    else:
        result["mismatch_count"] = int(np.count_nonzero(left != right))
        result["exact"] = np.array_equal(left, right, equal_nan=True)
    return result


def run_regime(frames, transforms_for_frames, repeats):
    correctness = {"fused_cells": None, "dynamic_mask": None, "rebinned_prior": None}
    cpu_times = []
    gpu_times = []
    for repeat in range(repeats):
        cpu_stored = np.empty(0, dtype=CELL_DTYPE)
        gpu_stored = np.empty(0, dtype=CELL_DTYPE)
        cpu_frame_times = []
        gpu_frame_times = []
        for frame, (cells, transform) in enumerate(zip(frames, transforms_for_frames)):
            start = time.perf_counter()
            cpu_result = process_frame(cells, cpu_stored, transform)
            cpu_frame_times.append((time.perf_counter() - start) * 1000.0)
            start = time.perf_counter()
            gpu_result = process_frame_cuda(cells, gpu_stored, transform)
            gpu_frame_times.append((time.perf_counter() - start) * 1000.0)
            if repeat == 0 and correctness["fused_cells"] is None:
                correctness["fused_cells"] = compare_array(cpu_result[0], gpu_result[0])
                correctness["dynamic_mask"] = compare_array(cpu_result[1], gpu_result[1])
                correctness["rebinned_prior"] = compare_array(cpu_result[2], gpu_result[2])
            cpu_stored = cpu_result[0]
            gpu_stored = gpu_result[0]
        cpu_times.append(cpu_frame_times)
        gpu_times.append(gpu_frame_times)

    def summarize(values):
        return {
            "mean_ms": float(np.mean(values)),
            "p50_ms": float(np.percentile(values, 50)),
            "p95_ms": float(np.percentile(values, 95)),
            "max_ms": float(np.max(values)),
            "repeat_mean_sigma_ms": float(np.std([np.mean(item) for item in values])),
        }

    paired = np.concatenate([
        np.asarray(cpu) - np.asarray(gpu)
        for cpu, gpu in zip(cpu_times, gpu_times)
    ])
    return {
        "correctness": correctness,
        "cpu": summarize(cpu_times),
        "gpu": summarize(gpu_times),
        "paired_cpu_minus_gpu": {
            "mean_ms": float(np.mean(paired)),
            "positive_fraction": float(np.mean(paired > 0)),
        },
        "stored_cells_last": {
            "cpu": int(len(cpu_stored)),
            "gpu": int(len(gpu_stored)),
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--output", type=Path, default=ROOT / "gpu_fusion_benchmark.json")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    paths = frame_paths(MAP_ROOT)
    if args.max_frames is not None:
        paths = paths[:args.max_frames]
    frames = load_frames(paths)
    report = {
        "environment": {
            "python": sys.version,
            "torch": torch.__version__,
            "cuda_build": torch.version.cuda,
            "device": torch.cuda.get_device_name(0),
            "numpy": np.__version__,
            "frames": len(frames),
            "repeats": args.repeats,
        },
        "regimes": {},
    }
    for regime in ("identity", "nonidentity"):
        report["regimes"][regime] = run_regime(
            frames, transforms(len(frames), regime), args.repeats
        )
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "device": report["environment"]["device"],
        "frames": len(frames),
        "identity": report["regimes"]["identity"]["gpu"],
        "nonidentity": report["regimes"]["nonidentity"]["gpu"],
        "output": str(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()
