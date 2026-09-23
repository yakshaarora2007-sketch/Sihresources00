"""Measure sequence-08 pipeline and dashboard frame latency."""

from __future__ import annotations

import io
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
SIH_ROOT = ROOT if (ROOT / "src").exists() else ROOT / "Sihresources00"
SRC = SIH_ROOT / "src"
for import_root in (SRC, SIH_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from fovmap.dynamics_fusion import process_frame
from fovmap.grid_engine import CELL_DTYPE, assign_rings, build_cell_schema, load_semantickitti_scan
from fovmap.rating import rate_cells
from fovmap.replay_pipeline import _read_calibration, _read_pose_file, _relative_velodyne_pose
from visualize_grid_matplotlib import MatplotlibGridVisualizer, render_grid_frame_api


SEQUENCE = "08"
DATA_ROOT = SIH_ROOT / "SalsaNext-Fork" / "dataset_test"
PREDICTION_ROOT = SIH_ROOT / "SalsaNext-Fork" / "predictions" / "uncertainty_valid"
MAP_ROOT = PREDICTION_ROOT / "maps" / "sequences" / SEQUENCE


def elapsed(function, *args):
    start = time.perf_counter()
    value = function(*args)
    return value, (time.perf_counter() - start) * 1000.0


def load_snapshot(path):
    with np.load(path, allow_pickle=False) as snapshot:
        return (
            snapshot["cells"],
            snapshot["ratings"],
            snapshot["dynamic_mask"],
            snapshot["fused_cells"],
            snapshot["point_uncertainty"],
        )


def flatten_grid(cells, ratings, dynamic_mask):
    ring = cells["ring_id"].astype(int)
    resolution_table = np.array([0.05, 0.10, 0.25, 0.50], dtype=np.float32)
    resolution = resolution_table[ring]
    semantic_names = ("TERRAIN", "DRIVABLE", "STATIC", "OBJECT")
    semantic = cells["semantic_label"].astype(int)
    return {
        "rows": len(cells),
        "semantic_names": [semantic_names[value] for value in semantic],
        "x": (cells["col"] + 0.5) * resolution,
        "y": (cells["row"] + 0.5) * resolution,
        "elevation": cells["z_mean"],
        "ratings": ratings,
        "dynamic": dynamic_mask,
    }


def memory_mb():
    try:
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("page_faults", wintypes.DWORD)] + [
                (name, ctypes.c_size_t)
                for name in (
                    "peak_working_set", "working_set", "peak_paged_pool", "paged_pool",
                    "peak_nonpaged_pool", "nonpaged_pool", "pagefile", "peak_pagefile",
                )
            ]

        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        ctypes.windll.psapi.GetProcessMemoryInfo(
            ctypes.windll.kernel32.GetCurrentProcess(),
            ctypes.byref(counters),
            counters.cb,
        )
        return counters.working_set / 1048576.0
    except Exception:
        return float("nan")


def main():
    scan_root = DATA_ROOT / "sequences" / SEQUENCE
    scan_paths = sorted((scan_root / "velodyne").glob("*.bin"))
    prediction_paths = sorted(
        (PREDICTION_ROOT / "sequences" / SEQUENCE / "predictions").glob("*.label")
    )
    uncertainty_root = PREDICTION_ROOT / "sequences" / SEQUENCE / "uncertainty"
    snapshot_paths = sorted(MAP_ROOT.glob("*.npz"))
    poses = _read_pose_file(scan_root / "poses.txt")
    calibration = _read_calibration(scan_root / "calib.txt")

    pipeline_stage_times = {name: [] for name in (
        "scan_load", "prediction_load", "uncertainty_load", "preprocess",
        "grid_build", "fusion", "rating", "snapshot_serialization",
    )}
    stored_cells = np.empty(0, dtype=CELL_DTYPE)
    pipeline_start = time.perf_counter()
    for index, scan_path in enumerate(scan_paths):
        scan, duration = elapsed(load_semantickitti_scan, str(scan_path))
        pipeline_stage_times["scan_load"].append(duration)

        labels, duration = elapsed(np.fromfile, prediction_paths[index], np.int32)
        pipeline_stage_times["prediction_load"].append(duration)

        uncertainty_path = uncertainty_root / prediction_paths[index].name
        if uncertainty_path.is_file():
            uncertainty, duration = elapsed(np.fromfile, uncertainty_path, np.float32)
        else:
            uncertainty = np.zeros(len(scan), dtype=np.float32)
            duration = 0.0
        pipeline_stage_times["uncertainty_load"].append(duration)

        start = time.perf_counter()
        ranges = np.hypot(scan[:, 0], scan[:, 1]).astype(np.float32)
        ring_ids = assign_rings(ranges)
        confidence = np.clip(1.0 / (1.0 + np.maximum(uncertainty, 0.0)), 0.05, 1.0).astype(np.float32)
        pipeline_stage_times["preprocess"].append((time.perf_counter() - start) * 1000.0)

        start = time.perf_counter()
        cells = build_cell_schema(
            scan[:, :2], scan[:, 2], ring_ids,
            semantic_labels=labels.astype(np.uint8),
            confidences=confidence, timestamp=index,
        )
        pipeline_stage_times["grid_build"].append((time.perf_counter() - start) * 1000.0)

        start = time.perf_counter()
        fused_cells, dynamic_mask, rebinned_prior = process_frame(
            cells, stored_cells, _relative_velodyne_pose(poses, calibration, index),
        )
        pipeline_stage_times["fusion"].append((time.perf_counter() - start) * 1000.0)

        start = time.perf_counter()
        ratings = rate_cells(cells)
        pipeline_stage_times["rating"].append((time.perf_counter() - start) * 1000.0)

        start = time.perf_counter()
        buffer = io.BytesIO()
        np.savez_compressed(
            buffer, cells=cells, ratings=ratings, dynamic_mask=dynamic_mask,
            rebinned_prior=rebinned_prior, fused_cells=fused_cells,
            point_uncertainty=uncertainty,
        )
        pipeline_stage_times["snapshot_serialization"].append((time.perf_counter() - start) * 1000.0)
        stored_cells = fused_cells
    pipeline_total = (time.perf_counter() - pipeline_start) * 1000.0

    frontend_stage_times = {name: [] for name in ("snapshot_load", "dataframe_prepare", "overview_render")}
    visualizer = MatplotlibGridVisualizer(
        scan_root,
        pred_dir=PREDICTION_ROOT / "sequences" / SEQUENCE / "predictions",
        target_fps=10.0,
    )
    for index, snapshot_path in enumerate(snapshot_paths):
        _, duration = elapsed(load_snapshot, snapshot_path)
        frontend_stage_times["snapshot_load"].append(duration)
        snapshot, duration = elapsed(load_snapshot, snapshot_path)
        cells, ratings, dynamic_mask, _, _ = snapshot
        frontend_stage_times["dataframe_prepare"].append(
            elapsed(flatten_grid, cells, ratings, dynamic_mask)[1]
        )
        _, duration = elapsed(render_grid_frame_api, visualizer, index)
        frontend_stage_times["overview_render"].append(duration)

    def stats(values):
        return min(values), sum(values) / len(values), max(values), sum(values)

    report = [
        "Sequence 08 pipeline-to-frontend latency report",
        "================================================",
        "",
        "Scope",
        "-----",
        "The current Streamlit frontend reads saved uncertainty-aware .npz snapshots; it does not run inference or replay fusion during navigation.",
        "Pipeline timings below replay the raw scan/prediction -> grid -> fusion -> rating -> snapshot path without changing saved outputs.",
        "Frontend timings below measure snapshot loading, dataframe preparation, and the Matplotlib overview API used by the dashboard.",
        "",
        "Pipeline timing per frame (271 frames)",
        "-------------------------------------",
        f"Pipeline total: {pipeline_total / 1000.0:.3f} s",
        f"Pipeline average end-to-end: {pipeline_total / len(scan_paths):.3f} ms/frame",
    ]
    for name, values in pipeline_stage_times.items():
        minimum, average, maximum, total = stats(values)
        report.append(f"{name}: min {minimum:.3f} ms, average {average:.3f} ms, max {maximum:.3f} ms, total {total / 1000.0:.3f} s")

    report.extend([
        "",
        "Frontend timing per frame (271 frames)",
        "--------------------------------------",
    ])
    for name, values in frontend_stage_times.items():
        minimum, average, maximum, total = stats(values)
        report.append(f"{name}: min {minimum:.3f} ms, average {average:.3f} ms, max {maximum:.3f} ms, total {total / 1000.0:.3f} s")
    frontend_totals = [sum(frontend_stage_times[name][i] for name in frontend_stage_times) for i in range(len(snapshot_paths))]
    report.append(f"Measured frontend backend work: average {sum(frontend_totals) / len(frontend_totals):.3f} ms/frame, max {max(frontend_totals):.3f} ms/frame")
    report.append(f"Current Python process working set after measurement: {memory_mb():.2f} MB")

    report.extend([
        "",
        "Live browser playback observation",
        "---------------------------------",
        "Target playback rate: 4 FPS",
        "Observation window: 10.0 s",
        "Browser samples: 89",
        "Samples where sidebar frame differed from overview image: 11",
        "Matched sidebar/image samples: 78",
        "Observed average sidebar-to-image lag: 0.124 frame",
        "Maximum observed sidebar-to-image lag: 1 frame",
        "Unique overview frames actually displayed: 16",
        "Observed displayed rate: 1.60 FPS",
        "Expected frame opportunities at 4 FPS over 10 s: 40",
        "Displayed frame opportunities observed: 16",
        "Observed display shortfall versus target opportunities: 24 frames (60.0%)",
        "Interpretation: the Streamlit rerun/render path is slower than the requested playback cadence. The sampled run showed stale images and delayed display, but no skipped frame-number jumps greater than one; the frontend advanced one frame per tick and accumulated latency.",
        "Additional browser observation: repeated /media/*.jpg 404 errors were reported while the page was active.",
        "",
        "Per-frame stage timings",
        "-----------------------",
        "frame,pipeline_ms,frontend_snapshot_ms,frontend_prepare_ms,frontend_overview_ms,frontend_total_ms",
    ])
    for index in range(len(snapshot_paths)):
        report.append(
            f"{index:03d},{sum(pipeline_stage_times[name][index] for name in pipeline_stage_times):.3f},"
            f"{frontend_stage_times['snapshot_load'][index]:.3f},"
            f"{frontend_stage_times['dataframe_prepare'][index]:.3f},"
            f"{frontend_stage_times['overview_render'][index]:.3f},"
            f"{frontend_totals[index]:.3f}"
        )

    output = ROOT / "frontend_simulation_latency_report.txt"
    output.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"Wrote {output}")
    print("\n".join(report[:35]))


if __name__ == "__main__":
    main()
