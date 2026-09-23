"""Vectorized MVP traversability rating for foveated grid cells.

The numeric weights in this rating formula (0.7, 0.3, 0.8, H_max) are an MVP engineering convention selected for this POC, not a value derived from the source specification or published research. Sensitivity analysis is expected before treating these as final.

Local ground is the median elevation of nearby DRIVABLE cells in the same
ring.  A scan-wide DRIVABLE median is used when a cell has no local support;
when the scan has no DRIVABLE cells at all, that cell's ``z_min`` is used as a
conservative, finite fallback.
"""

from __future__ import annotations

import os
import warnings

import numpy as np

from .grid_engine import CELL_DTYPE, pack_cell_key


H_MAX_DEFAULT = 0.5
HEIGHT_WEIGHT = 0.7  # MVP engineering convention, not a scientific constant — see Step 7 spec for sensitivity analysis requirements.
SEMANTIC_WEIGHT = 0.3  # MVP engineering convention, not a scientific constant — see Step 7 spec for sensitivity analysis requirements.
DYNAMIC_WEIGHT = 0.8  # MVP engineering convention, not a scientific constant — see Step 7 spec for sensitivity analysis requirements.

TERRAIN = np.uint8(0)
DRIVABLE = np.uint8(1)
STATIC = np.uint8(2)
OBJECT = np.uint8(3)


def _validate_cells(cells: np.ndarray) -> None:
    if not isinstance(cells, np.ndarray) or cells.dtype.names is None:
        raise TypeError("cells must be a structured NumPy array")
    missing = set(CELL_DTYPE.names) - set(cells.dtype.names)
    if missing:
        raise ValueError(f"cells is missing required fields: {sorted(missing)}")


def estimate_local_ground(
    cells: np.ndarray,
    ground_neighborhood_radius: int = 3,
) -> np.ndarray:
    """Estimate per-cell ground elevation from nearby DRIVABLE cells.

    Neighborhoods are square windows in ``(row, col)`` grid units and never
    cross ring boundaries.  The implementation creates one vectorized lookup
    for each fixed window offset, rather than iterating over cells.  Missing
    local support falls back to the scan-wide DRIVABLE median.  If no
    DRIVABLE cells exist, ``z_min`` for each cell is used so the result is
    finite and does not grant an unsupported flat-ground bonus.
    """
    _validate_cells(cells)
    if not isinstance(ground_neighborhood_radius, (int, np.integer)):
        raise TypeError("ground_neighborhood_radius must be an integer")
    if ground_neighborhood_radius < 0:
        raise ValueError("ground_neighborhood_radius must be non-negative")

    n_cells = len(cells)
    if n_cells == 0:
        return np.empty(0, dtype=np.float32)

    ring_ids = cells["ring_id"].astype(np.int64, copy=False)
    rows = cells["row"].astype(np.int64, copy=False)
    cols = cells["col"].astype(np.int64, copy=False)

    order = np.argsort(cells["cell_key"], kind="mergesort")
    sorted_keys = cells["cell_key"][order]
    sorted_labels = cells["semantic_label"][order]
    sorted_z = cells["z_mean"][order].astype(np.float32, copy=False)

    window_size = 2 * int(ground_neighborhood_radius) + 1
    candidate_z = np.full((n_cells, window_size * window_size), np.nan, dtype=np.float32)
    offset_index = 0
    for row_offset in range(-ground_neighborhood_radius, ground_neighborhood_radius + 1):
        for col_offset in range(-ground_neighborhood_radius, ground_neighborhood_radius + 1):
            candidate_keys = pack_cell_key(
                ring_ids,
                rows + row_offset,
                cols + col_offset,
            )
            positions = np.searchsorted(sorted_keys, candidate_keys, side="left")
            valid_positions = positions < n_cells
            safe_positions = np.minimum(positions, n_cells - 1)
            valid_positions &= sorted_keys[safe_positions] == candidate_keys
            valid_positions &= sorted_labels[safe_positions] == DRIVABLE
            candidate_z[valid_positions, offset_index] = sorted_z[safe_positions[valid_positions]]
            offset_index += 1

    local_support = np.isfinite(candidate_z).any(axis=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        local_ground = np.nanmedian(candidate_z, axis=1).astype(np.float32)

    drivable = (cells["semantic_label"] == DRIVABLE) & np.isfinite(cells["z_mean"])
    if np.any(drivable):
        fallback_ground = np.float32(np.median(cells["z_mean"][drivable]))
        local_ground[~local_support] = fallback_ground
    else:
        local_ground[~local_support] = cells["z_min"][~local_support]

    # A malformed input should not turn a valid rating into NaN.
    invalid_ground = ~np.isfinite(local_ground)
    local_ground[invalid_ground] = np.nan_to_num(
        cells["z_min"][invalid_ground], nan=0.0
    )
    return local_ground.astype(np.float32, copy=False)


def rate_cells(
    cells: np.ndarray,
    h_max: float = H_MAX_DEFAULT,
    confidence_threshold: float = 0.5,
    ground_neighborhood_radius: int = 3,
) -> np.ndarray:
    """Return float32 traversability ratings in ``[0, 100]``.

    DRIVABLE cells below ``confidence_threshold`` receive the TERRAIN
    semantic penalty (0.5), while confident DRIVABLE cells receive no
    semantic penalty.  STATIC is treated like OBJECT (penalty 1.0) because it
    is considered non-traversable for this MVP.
    """
    _validate_cells(cells)
    if not np.isfinite(h_max) or h_max <= 0:
        raise ValueError("h_max must be a finite positive number")
    if not np.isfinite(confidence_threshold) or not 0 <= confidence_threshold <= 1:
        raise ValueError("confidence_threshold must be between 0 and 1")

    if os.environ.get("FOVMAP_RATING_IMPL", "baseline").lower() == "fast":
        from .rating_fast import estimate_local_ground_fast

        ground = estimate_local_ground_fast(cells, ground_neighborhood_radius)
    else:
        ground = estimate_local_ground(cells, ground_neighborhood_radius)
    obstacle_height = np.maximum(
        cells["z_max"].astype(np.float32) - ground,
        np.float32(0.0),
    )
    height_penalty = np.clip(obstacle_height / np.float32(h_max), 0.0, 1.0)

    labels = cells["semantic_label"]
    semantic_penalty = np.ones(len(cells), dtype=np.float32)
    semantic_penalty[labels == TERRAIN] = 0.5
    confident_drivable = (labels == DRIVABLE) & (
        cells["confidence"] >= confidence_threshold
    )
    low_confidence_drivable = (labels == DRIVABLE) & ~confident_drivable
    semantic_penalty[confident_drivable] = 0.0
    semantic_penalty[low_confidence_drivable] = 0.5

    base_score = 100.0 * (
        1.0 - HEIGHT_WEIGHT * height_penalty - SEMANTIC_WEIGHT * semantic_penalty
    )
    score = base_score * (
        1.0 - DYNAMIC_WEIGHT * cells["dynamic_flag"].astype(np.float32)
    )
    return np.clip(score, 0.0, 100.0).astype(np.float32)


__all__ = ["estimate_local_ground", "rate_cells"]
