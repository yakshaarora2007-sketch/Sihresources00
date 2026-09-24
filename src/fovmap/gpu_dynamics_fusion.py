"""CUDA/PyTorch Fusion prototype.

This module is intentionally separate from ``dynamics_fusion.py``.  PyTorch
CUDA reductions and matrix operations are not expected to preserve NumPy's
floating-point accumulation bits, so this path is benchmarked and compared
explicitly before any production replacement is considered.
"""

from __future__ import annotations

import numpy as np
import torch

from .dynamics_fusion import (
    DEFAULT_C_MIN,
    DEFAULT_LAMBDA_DECAY,
    DEFAULT_N_MIN,
    FUSED_CELL_DTYPE,
    LOGIT_EPSILON,
    PRUNE_RADIUS_M,
)
from .grid_engine import CELL_DTYPE, RING_BOUNDARIES, RING_RESOLUTIONS


CUDA_DEVICE = torch.device("cuda")


def _require_cuda() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")


def _to_device(array: np.ndarray, dtype: torch.dtype) -> torch.Tensor:
    return torch.as_tensor(np.ascontiguousarray(array), dtype=dtype, device=CUDA_DEVICE)


def _field_dtype(name: str):
    if name in ("cell_key", "row", "col"):
        return torch.int64
    if name == "ring_id" or name == "semantic_label":
        return torch.int64
    if name == "point_count" or name == "timestamp":
        return torch.float64
    if name == "dynamic_flag":
        return torch.bool
    return torch.float32


def _structured_to_torch(cells: np.ndarray) -> dict[str, torch.Tensor]:
    return {
        field: _to_device(cells[field], _field_dtype(field))
        for field in CELL_DTYPE.names
        if field in cells.dtype.names
    } | ({
        "log_odds": _to_device(cells["log_odds"], torch.float32)
    } if "log_odds" in cells.dtype.names else {})


def _transform_centers(stored: dict[str, torch.Tensor], T_rel: np.ndarray):
    ring = stored["ring_id"]
    resolutions = _to_device(RING_RESOLUTIONS, torch.float32)[ring]
    centers = torch.stack((
        (stored["col"].to(torch.float32) + 0.5) * resolutions,
        (stored["row"].to(torch.float32) + 0.5) * resolutions,
        stored["z_mean"],
        torch.ones_like(stored["z_mean"]),
    ), dim=1)
    transform = _to_device(np.asarray(T_rel, dtype=np.float32), torch.float32)
    return (centers @ transform.T)[:, :3]


def _assign_rings(ranges: torch.Tensor) -> torch.Tensor:
    boundaries = _to_device(RING_BOUNDARIES, torch.float32)
    return torch.bucketize(ranges, boundaries, right=True).to(torch.int64)


def _compute_keys(points_xy: torch.Tensor, rings: torch.Tensor):
    resolutions = _to_device(RING_RESOLUTIONS, torch.float32)[rings]
    cols = torch.floor(points_xy[:, 0] / resolutions).to(torch.int64)
    rows = torch.floor(points_xy[:, 1] / resolutions).to(torch.int64)
    offset = 1 << 23
    keys = (rings << 48) | ((rows + offset) << 24) | (cols + offset)
    return keys, rows, cols


def _group_min_max(values_min, values_max, inverse, group_count):
    minimum = torch.full(
        (group_count,), float("inf"), dtype=torch.float32, device=CUDA_DEVICE
    )
    maximum = torch.full(
        (group_count,), float("-inf"), dtype=torch.float32, device=CUDA_DEVICE
    )
    minimum.scatter_reduce_(0, inverse, values_min, reduce="amin", include_self=True)
    maximum.scatter_reduce_(0, inverse, values_max, reduce="amax", include_self=True)
    return minimum, maximum


def _rebin(stored: dict[str, torch.Tensor], T_rel: np.ndarray) -> dict[str, torch.Tensor]:
    count = len(stored["cell_key"])
    if count == 0:
        return {
            field: torch.empty(0, dtype=_field_dtype(field), device=CUDA_DEVICE)
            for field in FUSED_CELL_DTYPE.names
        }

    transformed = _transform_centers(stored, T_rel)
    rings = _assign_rings(torch.hypot(transformed[:, 0], transformed[:, 1]))
    keys, _, _ = _compute_keys(transformed[:, :2], rings)
    unique_keys, inverse = torch.unique(keys, sorted=True, return_inverse=True)
    groups = len(unique_keys)
    point_count = stored["point_count"]
    point_counts = torch.bincount(inverse, weights=point_count, minlength=groups)
    z_sums = torch.bincount(
        inverse,
        weights=stored["z_mean"].to(torch.float64) * point_count,
        minlength=groups,
    )
    z_mean = z_sums / point_counts
    z_min, z_max = _group_min_max(
        stored["z_min"], stored["z_max"], inverse, groups
    )
    label_sums = torch.bincount(
        inverse,
        weights=stored["semantic_label"].to(torch.float64) * point_count,
        minlength=groups,
    )
    confidence_sums = torch.bincount(
        inverse,
        weights=stored["confidence"].to(torch.float64) * point_count,
        minlength=groups,
    )
    timestamp_sums = torch.bincount(
        inverse,
        weights=stored["timestamp"] * point_count,
        minlength=groups,
    )
    dynamic_sums = torch.bincount(
        inverse,
        weights=stored["dynamic_flag"].to(torch.float64) * point_count,
        minlength=groups,
    )

    output = {
        "cell_key": unique_keys,
        "ring_id": (unique_keys >> 48) & 0xFFFF,
        "row": ((unique_keys >> 24) & 0xFFFFFF) - (1 << 23),
        "col": (unique_keys & 0xFFFFFF) - (1 << 23),
        "point_count": point_counts,
        "z_mean": (z_mean).to(torch.float32),
        "z_min": z_min,
        "z_max": z_max,
        "semantic_label": torch.round(label_sums / point_counts).to(torch.int64),
        "confidence": (confidence_sums / point_counts).to(torch.float32),
        "dynamic_flag": (dynamic_sums / point_counts > 0.5),
        "timestamp": torch.round(timestamp_sums / point_counts),
    }
    if "log_odds" in stored:
        source_weights = torch.clamp(point_count, min=1.0)
        contributions = torch.bincount(
            inverse,
            weights=source_weights * stored["log_odds"].to(torch.float64),
            minlength=groups,
        )
        weights = torch.bincount(
            inverse, weights=source_weights, minlength=groups
        )
        output["log_odds"] = torch.where(
            weights > 0, contributions / weights, torch.zeros_like(weights)
        ).to(torch.float32)
    else:
        output["log_odds"] = torch.zeros(groups, dtype=torch.float32, device=CUDA_DEVICE)
    return output


def _matching(current_keys, prior_keys):
    result = torch.full(
        (len(current_keys),), -1, dtype=torch.int64, device=CUDA_DEVICE
    )
    if len(current_keys) == 0 or len(prior_keys) == 0:
        return result
    positions = torch.searchsorted(prior_keys, current_keys)
    valid = positions < len(prior_keys)
    safe = torch.clamp(positions, max=len(prior_keys) - 1)
    valid &= prior_keys[safe] == current_keys
    result[valid] = positions[valid]
    return result


def _measurement_log_odds(current):
    occupied = (current["semantic_label"] == 2) | (current["semantic_label"] == 3)
    confidence = torch.clamp(current["confidence"], LOGIT_EPSILON, 1.0 - LOGIT_EPSILON)
    magnitude = torch.log(confidence / (1.0 - confidence))
    return torch.where(occupied, magnitude, -magnitude).to(torch.float32)


def _to_numpy(values: dict[str, torch.Tensor], dtype: np.dtype):
    result = np.empty(len(values["cell_key"]), dtype=dtype)
    for field in dtype.names:
        value = values[field].detach().cpu().numpy()
        result[field] = value
    return result


@torch.inference_mode()
def process_frame_cuda(
    current_cells: np.ndarray,
    stored_cells: np.ndarray,
    T_rel: np.ndarray,
    c_min: float = DEFAULT_C_MIN,
    n_min: int = DEFAULT_N_MIN,
    lambda_decay: float = DEFAULT_LAMBDA_DECAY,
    prune_radius_m: float = PRUNE_RADIUS_M,
):
    """Run the experimental CUDA Fusion path and return NumPy API outputs."""
    _require_cuda()
    current = _structured_to_torch(current_cells)
    stored = _structured_to_torch(stored_cells)
    rebinned = _rebin(stored, T_rel)
    prior_indices = _matching(current["cell_key"], rebinned["cell_key"])
    matched = prior_indices >= 0
    safe = torch.clamp(prior_indices, min=0)

    previous_labels = torch.zeros(len(current_cells), dtype=torch.int64, device=CUDA_DEVICE)
    if len(rebinned["cell_key"]):
        previous_labels = rebinned["semantic_label"][safe]
    material = (
        ((current["semantic_label"] == 1) & (previous_labels == 2))
        | ((current["semantic_label"] == 2) & (previous_labels == 1))
        | ((current["semantic_label"] == 1) & (previous_labels == 3))
        | ((current["semantic_label"] == 3) & (previous_labels == 1))
        | ((current["semantic_label"] == 2) & (previous_labels == 3))
        | ((current["semantic_label"] == 3) & (previous_labels == 2))
    )
    dynamic = (
        matched
        & (current["confidence"] >= c_min)
        & (current["point_count"] >= n_min)
        & material
    )

    current["dynamic_flag"] = dynamic
    current["log_odds"] = torch.zeros(
        len(current_cells), dtype=torch.float32, device=CUDA_DEVICE
    )
    static = ~dynamic
    measurement = _measurement_log_odds(current)
    current["log_odds"][static & matched] = (
        torch.tensor(lambda_decay, dtype=torch.float32, device=CUDA_DEVICE)
        * rebinned["log_odds"][safe[static & matched]]
        + measurement[static & matched]
    )
    current["log_odds"][static & ~matched] = measurement[static & ~matched]
    current["log_odds"][dynamic & matched] = rebinned["log_odds"][safe[dynamic & matched]]

    matched_prior = torch.zeros(len(rebinned["cell_key"]), dtype=torch.bool, device=CUDA_DEVICE)
    matched_prior[prior_indices[matched]] = True
    keep_prior = ~matched_prior
    carried = {field: value[keep_prior].clone() for field, value in rebinned.items()}
    carried["log_odds"] *= torch.tensor(lambda_decay, dtype=torch.float32, device=CUDA_DEVICE)

    fused = {
        field: torch.cat((current[field], carried[field]))
        for field in FUSED_CELL_DTYPE.names
    }
    resolutions = _to_device(RING_RESOLUTIONS, torch.float32)[
        torch.clamp(fused["ring_id"], max=3)
    ]
    x_centers = (fused["col"].to(torch.float32) + 0.5) * resolutions
    y_centers = (fused["row"].to(torch.float32) + 0.5) * resolutions
    keep = torch.hypot(x_centers, y_centers) <= prune_radius_m
    fused = {field: value[keep] for field, value in fused.items()}
    torch.cuda.synchronize()
    return (
        _to_numpy(fused, FUSED_CELL_DTYPE),
        dynamic.detach().cpu().numpy().astype(bool),
        _to_numpy(rebinned, FUSED_CELL_DTYPE),
    )


__all__ = ["process_frame_cuda"]
