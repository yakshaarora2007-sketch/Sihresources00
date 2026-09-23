"""Dynamics detection and robot-centric log-odds fusion.

Dynamics is a confidence- and evidence-gated semantic disagreement between a
current cell and a matching cell returned by ``rebin_stored_map``.  The
auditable material transitions are DRIVABLE <-> OBJECT and STATIC <-> DRIVABLE
or OBJECT; TERRAIN <-> DRIVABLE is deliberately ignored as boundary noise.

``CELL_DTYPE`` has no log-odds field, so this module uses ``FUSED_CELL_DTYPE``,
which appends a float32 ``log_odds`` field.  A binary occupancy convention is
used: OBJECT and STATIC are occupied, DRIVABLE and TERRAIN are free, and the
semantic confidence is the probability of that observed state.  Dynamic cells
are carried through with their prior log-odds unchanged, but never contribute
an observation update.  Unobserved stored cells receive decay so repeated
frames move their probability toward 0.5.

Robot-centric transform and re-quantization provide bounded memory and a stable
foveation center, but repeated transform/re-key operations can introduce
positional blur over many frames.  ``rebin_stored_map`` remains the source of
aligned prior cells; this module does not reimplement it.
"""

from __future__ import annotations

import numpy as np

from .grid_engine import (
    CELL_DTYPE,
    RING_RESOLUTIONS,
    assign_rings,
    compute_cell_keys,
    rebin_stored_map,
    transform_points,
)


LOG_ODDS_FIELD = "log_odds"
FUSED_CELL_DTYPE = np.dtype(CELL_DTYPE.descr + [(LOG_ODDS_FIELD, np.float32)])
PRUNE_RADIUS_M = 100.0
DEFAULT_LAMBDA_DECAY = 0.98
DEFAULT_C_MIN = 0.6
DEFAULT_N_MIN = 3
LOGIT_EPSILON = 1e-6

# Auditable semantic transitions; absence is represented by no matching prior.
# Design choice: TERRAIN <-> DRIVABLE is treated as segmentation noise and is
# never flagged; only DRIVABLE/OBJECT/STATIC transitions can trigger dynamics.
DEFAULT_MATERIAL_TRANSITION_PAIRS = frozenset({
    (1, 3), (3, 1),
    (1, 2), (2, 1),
    (2, 3), (3, 2),
})


def _validate_cells(cells: np.ndarray, name: str) -> None:
    if not isinstance(cells, np.ndarray) or cells.dtype.names is None:
        raise TypeError(f"{name} must be a structured NumPy array")
    missing = set(CELL_DTYPE.names) - set(cells.dtype.names)
    if missing:
        raise ValueError(f"{name} is missing required fields: {sorted(missing)}")


def _matching_indices(current_keys: np.ndarray, prior_keys: np.ndarray) -> np.ndarray:
    """Return prior indices for matching keys, or -1, using sorted search."""
    result = np.full(len(current_keys), -1, dtype=np.int64)
    if len(prior_keys) == 0 or len(current_keys) == 0:
        return result
    order = np.argsort(prior_keys, kind="mergesort")
    sorted_keys = prior_keys[order]
    positions = np.searchsorted(sorted_keys, current_keys, side="left")
    valid = positions < len(sorted_keys)
    safe_positions = np.minimum(positions, len(sorted_keys) - 1)
    valid &= sorted_keys[safe_positions] == current_keys
    result[valid] = order[positions[valid]]
    return result


def detect_dynamic_cells(
    current_cells: np.ndarray,
    rebinned_stored_cells: np.ndarray,
    c_min: float = DEFAULT_C_MIN,
    n_min: int = DEFAULT_N_MIN,
    material_transition_pairs: frozenset = None,
) -> np.ndarray:
    """Return a current-order dynamic mask without mutating either input.

    Only matched cells with current confidence >= ``c_min`` and point count >=
    ``n_min`` can be dynamic.  A newly observed current cell has no prior and
    is therefore never flagged.  The default transition table is intentionally
    stricter than generic label inequality.
    """
    _validate_cells(current_cells, "current_cells")
    _validate_cells(rebinned_stored_cells, "rebinned_stored_cells")
    if not np.isfinite(c_min) or not 0 <= c_min <= 1:
        raise ValueError("c_min must be between 0 and 1")
    if not isinstance(n_min, (int, np.integer)) or n_min < 1:
        raise ValueError("n_min must be a positive integer")
    pairs = DEFAULT_MATERIAL_TRANSITION_PAIRS if material_transition_pairs is None else frozenset(material_transition_pairs)
    if any(len(pair) != 2 for pair in pairs):
        raise ValueError("material_transition_pairs must contain 2-item pairs")

    prior_indices = _matching_indices(
        current_cells["cell_key"],
        rebinned_stored_cells["cell_key"],
    )
    matched = prior_indices >= 0
    safe_indices = np.maximum(prior_indices, 0)
    previous_labels = np.zeros(len(current_cells), dtype=np.uint8)
    if len(rebinned_stored_cells):
        previous_labels = rebinned_stored_cells["semantic_label"][safe_indices]
    transitions = np.column_stack((
        current_cells["semantic_label"],
        previous_labels,
    ))
    material = np.zeros(len(current_cells), dtype=bool)
    for pair in pairs:
        material |= np.all(transitions == np.asarray(pair, dtype=np.uint8), axis=1)
    return (
        matched
        & (current_cells["confidence"] >= c_min)
        & (current_cells["point_count"] >= n_min)
        & material
    )


def _as_fused_dtype(cells: np.ndarray) -> np.ndarray:
    result = np.empty(len(cells), dtype=FUSED_CELL_DTYPE)
    for field in CELL_DTYPE.names:
        result[field] = cells[field]
    if LOG_ODDS_FIELD in cells.dtype.names:
        result[LOG_ODDS_FIELD] = cells[LOG_ODDS_FIELD]
    else:
        result[LOG_ODDS_FIELD] = 0.0
    return result


def carry_log_odds_through_rebin(
    stored_cells: np.ndarray,
    rebinned_cells: np.ndarray,
    T_rel: np.ndarray,
) -> np.ndarray:
    """Attach prior log-odds to P4's rebinned cells without changing its output.

    Old-to-new keys are recomputed with P4's exact cell-center transform and
    floor quantization (option (a)); changing ``grid_engine.py`` to expose a
    mapping is unnecessary.  When multiple fine cells coarsen into one output
    cell, their log-odds are averaged weighted by old ``point_count``, matching
    P4's evidence-weighted aggregation spirit.  A refined output inherits its
    single coarse parent's log-odds directly, without scaling; this is carried
    belief rather than independent evidence.  If a re-keyed output has no
    traceable prior, its log-odds is initialized to zero.
    """
    _validate_cells(stored_cells, "stored_cells")
    _validate_cells(rebinned_cells, "rebinned_cells")
    if LOG_ODDS_FIELD not in stored_cells.dtype.names:
        return _as_fused_dtype(rebinned_cells)
    if len(rebinned_cells) == 0:
        return _as_fused_dtype(rebinned_cells)
    if len(stored_cells) == 0:
        return _as_fused_dtype(rebinned_cells)

    ring_ids = stored_cells["ring_id"].astype(np.int32)
    resolutions = RING_RESOLUTIONS[ring_ids]
    centers = np.column_stack((
        (stored_cells["col"].astype(np.float32) + 0.5) * resolutions,
        (stored_cells["row"].astype(np.float32) + 0.5) * resolutions,
        stored_cells["z_mean"],
    ))
    transformed = transform_points(centers, np.asarray(T_rel))
    new_ring_ids = assign_rings(
        np.hypot(transformed[:, 0], transformed[:, 1]).astype(np.float32)
    )
    new_keys, _, _ = compute_cell_keys(transformed[:, :2], new_ring_ids)

    output = _as_fused_dtype(rebinned_cells)
    output["log_odds"] = 0.0
    output_positions = _matching_indices(new_keys, output["cell_key"])
    valid = output_positions >= 0
    source_weights = stored_cells["point_count"].astype(np.float64)
    source_weights = np.maximum(source_weights, 1.0)
    contributions = np.bincount(
        output_positions[valid],
        weights=source_weights[valid] * stored_cells["log_odds"][valid],
        minlength=len(output),
    )
    weights = np.bincount(
        output_positions[valid],
        weights=source_weights[valid],
        minlength=len(output),
    )
    supported = weights > 0
    output["log_odds"][supported] = (
        contributions[supported] / weights[supported]
    ).astype(np.float32)
    return output


def _logit(probability: np.ndarray) -> np.ndarray:
    clipped = np.clip(probability.astype(np.float32), LOGIT_EPSILON, 1.0 - LOGIT_EPSILON)
    return np.log(clipped / (1.0 - clipped)).astype(np.float32)


def _measurement_log_odds(cells: np.ndarray) -> np.ndarray:
    occupied = (cells["semantic_label"] == 2) | (cells["semantic_label"] == 3)
    magnitude = _logit(cells["confidence"])
    return np.where(occupied, magnitude, -magnitude).astype(np.float32)


def fuse_log_odds(
    current_cells: np.ndarray,
    rebinned_stored_cells: np.ndarray,
    dynamic_mask: np.ndarray,
    lambda_decay: float = DEFAULT_LAMBDA_DECAY,
    prune_radius_m: float = PRUNE_RADIUS_M,
) -> np.ndarray:
    """Fuse static evidence and return an extended array with ``log_odds``.

    Current dynamic cells are retained for the next dynamics comparison, but
    retain their prior log-odds unchanged and do not contribute measurements.
    Matched static observations use ``lambda_decay * prior + measurement``;
    unmatched stored cells are carried forward with decay.  Cells outside the
    robot-centric pruning radius are removed.
    """
    _validate_cells(current_cells, "current_cells")
    _validate_cells(rebinned_stored_cells, "rebinned_stored_cells")
    if not np.isfinite(lambda_decay) or not 0 < lambda_decay <= 1:
        raise ValueError("lambda_decay must be in (0, 1]")
    if not np.isfinite(prune_radius_m) or prune_radius_m < 0:
        raise ValueError("prune_radius_m must be non-negative")
    dynamic_mask = np.asarray(dynamic_mask, dtype=bool)
    if dynamic_mask.shape != (len(current_cells),):
        raise ValueError("dynamic_mask must align with current_cells")

    prior = _as_fused_dtype(rebinned_stored_cells)
    current = _as_fused_dtype(current_cells)
    current["dynamic_flag"] = dynamic_mask
    prior_indices = _matching_indices(current["cell_key"], prior["cell_key"])
    matched = prior_indices >= 0
    safe_indices = np.maximum(prior_indices, 0)

    current_log_odds = current[LOG_ODDS_FIELD]
    current_log_odds[:] = 0.0
    static_observations = ~dynamic_mask
    current_log_odds[static_observations & matched] = (
        np.float32(lambda_decay) * prior[LOG_ODDS_FIELD][safe_indices[static_observations & matched]]
        + _measurement_log_odds(current)[static_observations & matched]
    )
    current_log_odds[static_observations & ~matched] = _measurement_log_odds(current)[static_observations & ~matched]
    current_log_odds[dynamic_mask & matched] = prior[LOG_ODDS_FIELD][safe_indices[dynamic_mask & matched]]

    unmatched_prior = ~np.isin(prior["cell_key"], current["cell_key"])
    carried = prior[unmatched_prior].copy()
    carried[LOG_ODDS_FIELD] *= np.float32(lambda_decay)
    fused = np.concatenate((current, carried))

    resolutions = np.choose(
        np.minimum(fused["ring_id"].astype(np.int64), 3),
        np.array([0.05, 0.10, 0.25, 0.50], dtype=np.float32),
    )
    x_centers = (fused["col"].astype(np.float32) + 0.5) * resolutions
    y_centers = (fused["row"].astype(np.float32) + 0.5) * resolutions
    distances = np.hypot(x_centers, y_centers)
    return fused[distances <= np.float32(prune_radius_m)]


def process_frame(
    current_cells: np.ndarray,
    stored_cells: np.ndarray,
    T_rel: np.ndarray,
    c_min: float = DEFAULT_C_MIN,
    n_min: int = DEFAULT_N_MIN,
    lambda_decay: float = DEFAULT_LAMBDA_DECAY,
    prune_radius_m: float = PRUNE_RADIUS_M,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run one complete re-bin -> dynamics -> fusion frame step.

    ``stored_cells`` is the previous fused map (or an empty ``CELL_DTYPE``
    array for the first frame).  The returned tuple is
    ``(fused_map, dynamic_mask, rebinned_prior)``.  The helper deliberately
    calls P4's ``rebin_stored_map`` before either comparison or fusion.

    P4's rebin output is ``CELL_DTYPE`` and therefore cannot carry the
    extended ``log_odds`` field through a non-identity transform.  For exact
    unchanged keys this helper restores the sidecar log-odds; transformed or
    coarsened keys start that prior at zero until P4 exposes log-odds
    aggregation.  This limitation is explicit rather than silently treating
    re-quantized log-odds as preserved.
    """
    _validate_cells(current_cells, "current_cells")
    _validate_cells(stored_cells, "stored_cells")
    T_rel = np.asarray(T_rel)
    if T_rel.shape != (4, 4) or not np.issubdtype(T_rel.dtype, np.number):
        raise ValueError("T_rel must be a numeric 4x4 transform")

    rebinned_prior = rebin_stored_map(stored_cells, T_rel)
    rebinned_prior = carry_log_odds_through_rebin(
        stored_cells,
        rebinned_prior,
        T_rel,
    )

    dynamic_mask = detect_dynamic_cells(
        current_cells,
        rebinned_prior,
        c_min=c_min,
        n_min=n_min,
    )
    fused_map = fuse_log_odds(
        current_cells,
        rebinned_prior,
        dynamic_mask,
        lambda_decay=lambda_decay,
        prune_radius_m=prune_radius_m,
    )
    return fused_map, dynamic_mask, rebinned_prior


__all__ = [
    "FUSED_CELL_DTYPE",
    "carry_log_odds_through_rebin",
    "detect_dynamic_cells",
    "fuse_log_odds",
    "process_frame",
]
