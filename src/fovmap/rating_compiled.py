"""Opt-in compiled Tier-1 ground estimator with baseline-compatible fallbacks."""

from __future__ import annotations

import logging

import numpy as np

from .grid_engine import CELL_DTYPE
from .rating import DRIVABLE, estimate_local_ground

_LOGGER = logging.getLogger(__name__)
_MISSING_EXTENSION_WARNED = False
_DUPLICATE_WARNED = False


def _load_kernel():
    try:
        from . import _rating_kernel
    except ImportError:
        global _MISSING_EXTENSION_WARNED
        if not _MISSING_EXTENSION_WARNED:
            _LOGGER.warning(
                "compiled Rating extension is unavailable; using baseline ground estimator"
            )
            _MISSING_EXTENSION_WARNED = True
        return None
    return _rating_kernel


def _strictly_ascending(keys: np.ndarray) -> bool:
    return len(keys) < 2 or bool(np.all(keys[1:] > keys[:-1]))


def _check_map_bounds(cells: np.ndarray) -> None:
    total_slots = 0
    for ring in np.unique(cells["ring_id"]):
        selected = cells["ring_id"] == ring
        rows = cells["row"][selected].astype(np.int64, copy=False)
        cols = cells["col"][selected].astype(np.int64, copy=False)
        height = int(rows.max() - rows.min() + 1)
        width = int(cols.max() - cols.min() + 1)
        if height > 2048 or width > 2048:
            raise RuntimeError(
                f"ring {int(ring)} lattice box exceeds 2048: {height}x{width}"
            )
        total_slots += height * width
    if total_slots > 8_000_000:
        raise RuntimeError(
            f"Rating lattice allocation exceeds 8000000 slots: {total_slots}"
        )


def estimate_local_ground_compiled(
    cells: np.ndarray,
    ground_neighborhood_radius: int = 3,
) -> np.ndarray:
    from .rating import _validate_cells

    _validate_cells(cells)
    if not isinstance(ground_neighborhood_radius, (int, np.integer)):
        raise TypeError("ground_neighborhood_radius must be an integer")
    if ground_neighborhood_radius < 0:
        raise ValueError("ground_neighborhood_radius must be non-negative")
    if len(cells) == 0:
        return np.empty(0, dtype=np.float32)
    if not _strictly_ascending(cells["cell_key"]):
        global _DUPLICATE_WARNED
        if not _DUPLICATE_WARNED:
            _LOGGER.warning(
                "Rating cell keys are not strictly ascending; using baseline estimator"
            )
            _DUPLICATE_WARNED = True
        return estimate_local_ground(cells, ground_neighborhood_radius)

    kernel = _load_kernel()
    if kernel is None:
        return estimate_local_ground(cells, ground_neighborhood_radius)
    _check_map_bounds(cells)
    ground, local_support = kernel.estimate(
        cells["row"],
        cells["col"],
        cells["ring_id"],
        cells["z_mean"],
        cells["semantic_label"],
        int(ground_neighborhood_radius),
    )
    drivable = (cells["semantic_label"] == DRIVABLE) & np.isfinite(cells["z_mean"])
    if np.any(drivable):
        fallback_ground = np.float32(np.median(cells["z_mean"][drivable]))
        ground[~local_support.astype(bool)] = fallback_ground
    else:
        ground[~local_support.astype(bool)] = cells["z_min"][~local_support.astype(bool)]
    invalid_ground = ~np.isfinite(ground)
    ground[invalid_ground] = np.nan_to_num(
        cells["z_min"][invalid_ground], nan=0.0
    )
    return ground.astype(np.float32, copy=False)


def warmup() -> None:
    cells = np.zeros(1, dtype=CELL_DTYPE)
    cells["cell_key"] = 2**23 * (2**24 + 1)
    cells["semantic_label"] = DRIVABLE
    cells["z_mean"] = 0.0
    cells["z_min"] = 0.0
    kernel = _load_kernel()
    if kernel is not None:
        estimate_local_ground_compiled(cells, 0)


__all__ = ["estimate_local_ground_compiled", "warmup"]
