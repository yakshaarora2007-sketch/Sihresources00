"""Headless SemanticKITTI replay through grid, rating, dynamics, and fusion.

This adapter consumes SalsaNext's four-class per-point prediction files and,
when present, aligned per-point uncertainty files. Pose files are treated as
cam0 poses and are conjugated by the velo-to-cam0 calibration before being
passed to robot-centric fusion.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .dynamics_fusion import process_frame
from .grid_engine import (
    CELL_DTYPE,
    assign_rings,
    build_cell_schema,
    load_semantickitti_scan,
)
from .rating import rate_cells


def _read_pose_file(path: Path) -> np.ndarray:
    rows = np.loadtxt(path, dtype=np.float64)
    rows = np.atleast_2d(rows)
    if rows.shape[1] != 12:
        raise ValueError(f"Expected 12 pose values per row in {path}")
    transforms = np.tile(np.eye(4, dtype=np.float64), (len(rows), 1, 1))
    transforms[:, :3, :] = rows.reshape(-1, 3, 4)
    return transforms


def _read_calibration(path: Path) -> np.ndarray:
    for line in path.read_text().splitlines():
        if line.startswith("Tr:") or line.startswith("Tr_velo_to_cam:"):
            values = np.fromstring(line.split(":", 1)[1], sep=" ")
            if values.size != 12:
                raise ValueError(f"Invalid velo-to-cam calibration in {path}")
            transform = np.eye(4, dtype=np.float64)
            transform[:3, :] = values.reshape(3, 4)
            return transform
    raise ValueError(f"No velo-to-cam calibration found in {path}")


def _relative_velodyne_pose(
    poses: np.ndarray,
    calibration: np.ndarray,
    index: int,
) -> np.ndarray:
    if index == 0:
        return np.eye(4, dtype=np.float32)
    # T_rel maps previous-frame velodyne coordinates into current-frame
    # velodyne coordinates when poses are cam0-frame absolute poses.
    relative_cam = np.linalg.inv(poses[index]) @ poses[index - 1]
    relative_velo = np.linalg.inv(calibration) @ relative_cam @ calibration
    return relative_velo.astype(np.float32)


def replay_prediction_sequence(
    dataset_root: str | Path,
    prediction_root: str | Path,
    output_root: str | Path,
    sequence: str = "08",
    max_frames: int | None = None,
) -> dict[str, int | str]:
    """Replay prediction files and save rated/fused map snapshots.

    ``prediction_root`` must contain ``sequences/<seq>/predictions/*.label``
    produced by ``infer.py``. Each output ``.npz`` contains cells, ratings,
    dynamic mask, and the aligned rebinned prior for that frame.
    """
    dataset_root = Path(dataset_root)
    prediction_root = Path(prediction_root)
    output_root = Path(output_root)
    sequence = f"{int(sequence):02d}"
    scan_dir = dataset_root / "sequences" / sequence / "velodyne"
    prediction_dir = prediction_root / "sequences" / sequence / "predictions"
    uncertainty_dir = prediction_root / "sequences" / sequence / "uncertainty"
    sequence_root = dataset_root / "sequences" / sequence
    if not scan_dir.is_dir() or not prediction_dir.is_dir():
        raise FileNotFoundError(
            f"Missing scan or prediction directory for sequence {sequence}"
        )

    scan_paths = sorted(scan_dir.glob("*.bin"))
    prediction_paths = sorted(prediction_dir.glob("*.label"))
    if len(scan_paths) != len(prediction_paths):
        raise ValueError(
            f"Scan/prediction count mismatch: {len(scan_paths)} vs "
            f"{len(prediction_paths)}"
        )
    poses = _read_pose_file(sequence_root / "poses.txt")
    calibration = _read_calibration(sequence_root / "calib.txt")
    frame_count = len(scan_paths) if max_frames is None else min(max_frames, len(scan_paths))

    output_sequence = output_root / "sequences" / sequence
    output_sequence.mkdir(parents=True, exist_ok=True)
    stored_cells = np.empty(0, dtype=CELL_DTYPE)
    for index in range(frame_count):
        scan = load_semantickitti_scan(str(scan_paths[index]))
        labels = np.fromfile(prediction_paths[index], dtype=np.int32)
        if len(labels) != len(scan):
            raise ValueError(
                f"Point/prediction mismatch at frame {index}: "
                f"{len(scan)} vs {len(labels)}"
            )
        uncertainty_path = uncertainty_dir / prediction_paths[index].name
        if uncertainty_path.is_file():
            uncertainty = np.fromfile(uncertainty_path, dtype=np.float32)
            if len(uncertainty) != len(scan):
                raise ValueError(
                    f"Point/uncertainty count mismatch at frame {index}: "
                    f"{len(scan)} vs {len(uncertainty)}"
                )
            confidence = np.clip(
                1.0 / (1.0 + np.maximum(uncertainty, 0.0)),
                0.05,
                1.0,
            ).astype(np.float32)
        else:
            uncertainty = np.zeros(len(scan), dtype=np.float32)
            confidence = np.ones(len(scan), dtype=np.float32)
        ranges = np.hypot(scan[:, 0], scan[:, 1]).astype(np.float32)
        cells = build_cell_schema(
            scan[:, :2],
            scan[:, 2],
            assign_rings(ranges),
            semantic_labels=labels.astype(np.uint8),
            confidences=confidence,
            timestamp=index,
        )
        fused_cells, dynamic_mask, rebinned_prior = process_frame(
            cells,
            stored_cells,
            _relative_velodyne_pose(poses, calibration, index),
        )
        ratings = rate_cells(cells)
        np.savez_compressed(
            output_sequence / f"{scan_paths[index].stem}.npz",
            cells=cells,
            ratings=ratings,
            dynamic_mask=dynamic_mask,
            rebinned_prior=rebinned_prior,
            fused_cells=fused_cells,
            point_uncertainty=uncertainty,
        )
        stored_cells = fused_cells

    metadata = {
        "sequence": sequence,
        "frames": frame_count,
        "confidence_source": (
            "SalsaNext ADF predictive variance mapped as 1/(1+variance)"
            if uncertainty_dir.is_dir() and any(uncertainty_dir.iterdir())
            else "1.0 fallback; uncertainty files not present"
        ),
        "pipeline": "scan -> prediction -> grid -> rating -> rebin -> dynamics -> log_odds_fusion",
    }
    (output_sequence / "metadata.json").write_text(json.dumps(metadata, indent=2))
    return metadata


__all__ = ["replay_prediction_sequence"]
