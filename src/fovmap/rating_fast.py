"""Optional Tier-1 dense per-ring implementation of Rating ground estimation."""

from __future__ import annotations

import warnings

import numpy as np

from .grid_engine import CELL_DTYPE
from .rating import DRIVABLE


_MAX_RING_SLOTS = 2_000_000


def _validate_cells(cells: np.ndarray) -> None:
    if not isinstance(cells, np.ndarray) or cells.dtype.names is None:
        raise TypeError("cells must be a structured NumPy array")
    missing = set(CELL_DTYPE.names) - set(cells.dtype.names)
    if missing:
        raise ValueError(f"cells is missing required fields: {sorted(missing)}")


def _ring_maps(
    cells: np.ndarray,
) -> list[tuple[int, int, int, np.ndarray, np.ndarray]]:
    maps = []
    for ring in np.unique(cells["ring_id"]):
        selected = cells["ring_id"] == ring
        rows = cells["row"][selected].astype(np.int64, copy=False)
        cols = cells["col"][selected].astype(np.int64, copy=False)
        row_min, row_max = int(rows.min()), int(rows.max())
        col_min, col_max = int(cols.min()), int(cols.max())
        height = row_max - row_min + 1
        width = col_max - col_min + 1
        slots = height * width
        if slots > _MAX_RING_SLOTS:
            raise ValueError(
                f"ring {int(ring)} lattice is too large: {slots} slots "
                f"(maximum {_MAX_RING_SLOTS})"
            )

        z_map = np.full((height, width), np.nan, dtype=np.float32)
        label_map = np.full((height, width), -1, dtype=np.int16)
        z_map[rows - row_min, cols - col_min] = cells["z_mean"][selected]
        label_map[rows - row_min, cols - col_min] = (
            cells["semantic_label"][selected].astype(np.int16, copy=False)
        )
        maps.append((int(ring), row_min, col_min, z_map, label_map))
    return maps


def estimate_local_ground_fast(
    cells: np.ndarray,
    ground_neighborhood_radius: int = 3,
) -> np.ndarray:
    """Estimate local ground without sorting or packed-key searches.

    The candidate matrix, median, finite-support rule, and fallback expressions
    intentionally match :func:`rating.estimate_local_ground`.
    """
    _validate_cells(cells)
    if not isinstance(ground_neighborhood_radius, (int, np.integer)):
        raise TypeError("ground_neighborhood_radius must be an integer")
    if ground_neighborhood_radius < 0:
        raise ValueError("ground_neighborhood_radius must be non-negative")

    n_cells = len(cells)
    if n_cells == 0:
        return np.empty(0, dtype=np.float32)

    radius = int(ground_neighborhood_radius)
    window_size = 2 * radius + 1
    candidate_z = np.full(
        (n_cells, window_size * window_size), np.nan, dtype=np.float32
    )
    ring_maps = {ring: (row_min, col_min, z_map, label_map)
                 for ring, row_min, col_min, z_map, label_map in _ring_maps(cells)}
    rows = cells["row"].astype(np.int64, copy=False)
    cols = cells["col"].astype(np.int64, copy=False)
    rings = cells["ring_id"]

    offset_index = 0
    for row_offset in range(-radius, radius + 1):
        for col_offset in range(-radius, radius + 1):
            target_rows = rows + row_offset
            target_cols = cols + col_offset
            for ring, (row_min, col_min, z_map, label_map) in ring_maps.items():
                selected = rings == ring
                map_rows = target_rows[selected] - row_min
                map_cols = target_cols[selected] - col_min
                valid = (
                    (map_rows >= 0)
                    & (map_rows < z_map.shape[0])
                    & (map_cols >= 0)
                    & (map_cols < z_map.shape[1])
                )
                selected_indices = np.flatnonzero(selected)
                valid_indices = selected_indices[valid]
                values = z_map[map_rows[valid], map_cols[valid]]
                labels = label_map[map_rows[valid], map_cols[valid]]
                valid_indices = valid_indices[labels == int(DRIVABLE)]
                values = values[labels == int(DRIVABLE)]
                candidate_z[valid_indices, offset_index] = values
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

    invalid_ground = ~np.isfinite(local_ground)
    local_ground[invalid_ground] = np.nan_to_num(
        cells["z_min"][invalid_ground], nan=0.0
    )
    return local_ground.astype(np.float32, copy=False)


__all__ = ["estimate_local_ground_fast"]
