"""Benchmark baseline, NumPy S4, and compiled Rating on sequence 08."""

from __future__ import annotations

import hashlib
import os
import time
from pathlib import Path

import numpy as np

from visualize_grid_matplotlib import (
    MatplotlibGridVisualizer,
    resolve_dataset,
    resolve_predictions,
)
from fovmap.rating import rate_cells


def _stats(values):
    return (
        float(np.min(values)),
        float(np.mean(values)),
        float(np.percentile(values, 50)),
        float(np.percentile(values, 95)),
        float(np.max(values)),
    )


def run(mode: str, frames):
    os.environ["FOVMAP_RATING_IMPL"] = mode
    first_start = time.perf_counter()
    rate_cells(frames[0])
    first_ms = (time.perf_counter() - first_start) * 1000.0
    timings = []
    hashes = []
    for cells in frames:
        start = time.perf_counter()
        ratings = rate_cells(cells)
        timings.append((time.perf_counter() - start) * 1000.0)
        hashes.append(hashlib.sha256(ratings.tobytes()).hexdigest())
    return np.asarray(timings), hashes, first_ms


def main():
    sequence = "08"
    data_dir = resolve_dataset(None, sequence)
    pred_dir = resolve_predictions(None, data_dir, sequence)
    visualizer = MatplotlibGridVisualizer(data_dir, pred_dir=pred_dir)
    frames = [visualizer.load_frame_data(i)["cells"] for i in range(visualizer.num_frames)]
    results = {}
    for iteration in range(3):
        for mode in ("baseline", "compiled"):
            results.setdefault(mode, []).append(run(mode, frames))
        baseline = results["baseline"][-1][0]
        compiled = results["compiled"][-1][0]
        print(
            f"run {iteration + 1}: baseline {_stats(baseline)}; "
            f"compiled {_stats(compiled)}; "
            f"hash_equal={results['baseline'][-1][1] == results['compiled'][-1][1]}"
        )
    for mode, runs in results.items():
        aggregate = np.concatenate([run_result[0] for run_result in runs])
        print(f"{mode} aggregate: {_stats(aggregate)}")
    print(
        "compiled deterministic:",
        all(run_result[1] == results["compiled"][0][1] for run_result in results["compiled"][1:]),
    )


if __name__ == "__main__":
    main()
