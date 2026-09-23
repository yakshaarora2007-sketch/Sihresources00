import numpy as np

from .grid_engine import CELL_DTYPE, pack_cell_key
from .dynamics_fusion import detect_dynamic_cells


def cell(label, confidence=1.0, point_count=3, row=0, col=0):
    result = np.zeros(1, dtype=CELL_DTYPE)
    result["cell_key"] = pack_cell_key(np.array([0]), np.array([row]), np.array([col]))
    result["row"], result["col"] = row, col
    result["point_count"] = point_count
    result["semantic_label"] = label
    result["confidence"] = confidence
    return result


def test_confident_high_evidence_flip_is_dynamic():
    assert detect_dynamic_cells(cell(3), cell(1))[0]


def test_confidence_and_point_gates_block_flip():
    assert not detect_dynamic_cells(cell(3, confidence=0.5), cell(1))[0]
    assert not detect_dynamic_cells(cell(3, point_count=2), cell(1))[0]


def test_terrain_drivable_boundary_is_ignored():
    assert not detect_dynamic_cells(cell(0), cell(1))[0]


def test_unmatched_new_cell_is_never_dynamic():
    assert not detect_dynamic_cells(cell(3, row=5), cell(1))[0]
