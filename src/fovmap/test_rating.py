import numpy as np

from .grid_engine import CELL_DTYPE, pack_cell_key
from .rating import estimate_local_ground, rate_cells
from .rating_fast import estimate_local_ground_fast
from .rating_compiled import estimate_local_ground_compiled


def make_cells(*specs):
    cells = np.zeros(len(specs), dtype=CELL_DTYPE)
    for index, spec in enumerate(specs):
        row, col, z_mean, z_max, label, confidence = spec[:6]
        cells[index]["cell_key"] = (
            (np.int64(0) << 48)
            | (np.int64(row + (1 << 23)) << 24)
            | np.int64(col + (1 << 23))
        )
        cells[index]["row"] = row
        cells[index]["col"] = col
        cells[index]["point_count"] = 1
        cells[index]["z_mean"] = z_mean
        cells[index]["z_min"] = z_mean
        cells[index]["z_max"] = z_max
        cells[index]["semantic_label"] = label
        cells[index]["confidence"] = confidence
        cells[index]["dynamic_flag"] = bool(spec[6]) if len(spec) > 6 else False
    return cells


def test_geometry_is_monotonic():
    cells = make_cells(
        (0, 0, 0.0, 0.0, 1, 1.0),
        (0, 1, 0.0, 0.05, 1, 1.0),
        (0, 2, 0.0, 0.20, 1, 1.0),
        (0, 3, 0.0, 0.40, 1, 1.0),
    )
    scores = rate_cells(cells)
    assert scores[0] > 99
    assert scores[3] < scores[2] < scores[1] < scores[0]


def test_low_confidence_drivable_is_more_conservative():
    cells = make_cells((0, 0, 0.0, 0.0, 1, 1.0), (0, 1, 0.0, 0.0, 1, 0.2))
    scores = rate_cells(cells)
    assert scores[0] > scores[1]


def test_estimate_local_ground_returns_expected_median():
    cells = make_cells(
        (0, 0, 1.0, 1.0, 0, 1.0),
        (0, 1, 0.00, 0.00, 1, 1.0),
        (1, 0, 0.02, 0.02, 1, 1.0),
        (1, 1, -0.01, -0.01, 1, 1.0),
        (-1, 0, 0.01, 0.01, 1, 1.0),
        (-1, -1, 0.00, 0.00, 1, 1.0),
    )
    ground = estimate_local_ground(cells, ground_neighborhood_radius=1)
    assert np.isclose(ground[0], 0.00, atol=1e-6)


def test_terrain_bump_is_reduced_but_still_finite():
    cells = make_cells(
        (0, 0, 0.0, 0.0, 1, 1.0),
        (0, 1, 0.0, 0.05, 1, 1.0),
    )
    score = rate_cells(cells)[1]
    assert 0 < score < 100


def test_object_is_low_score():
    cells = make_cells((0, 0, 0.0, 0.0, 1, 1.0), (0, 1, 0.0, 0.0, 3, 1.0))
    scores = rate_cells(cells)
    assert scores[1] < scores[0]
    assert np.isclose(scores[1], 70.0)


def test_static_uses_non_traversable_penalty():
    cells = make_cells((0, 0, 0.0, 0.0, 1, 1.0), (0, 1, 0.0, 0.0, 2, 1.0))
    assert np.isclose(rate_cells(cells)[1], 70.0)


def test_dynamic_penalty_reduces_same_cell_score():
    cells = make_cells((0, 0, 0.0, 0.0, 1, 1.0), (0, 1, 0.0, 0.0, 1, 1.0, True))
    scores = rate_cells(cells)
    assert np.isclose(scores[1], scores[0] * 0.2)


def test_no_local_drivable_uses_finite_conservative_fallback():
    cells = make_cells((0, 0, 1.0, 1.2, 0, 1.0))
    ground = estimate_local_ground(cells)
    score = rate_cells(cells)
    assert np.isfinite(ground[0])
    assert np.isfinite(score[0])
    assert score[0] < 100


def test_fast_ground_matches_baseline_for_mixed_rings_and_edges():
    cells = make_cells(
        (-2, -2, 1.0, 1.1, 1, 1.0),
        (-2, -1, 1.1, 1.2, 1, 1.0),
        (-1, -2, np.nan, 1.3, 1, 1.0),
        (0, 0, 2.0, 2.1, 1, 1.0),
        (0, 1, 2.1, 2.2, 3, 1.0),
        (1, 0, 2.2, 2.3, 1, 0.2),
    )
    cells["ring_id"] = [0, 0, 0, 2, 2, 2]
    cells["cell_key"] = pack_cell_key(
        cells["ring_id"], cells["row"], cells["col"]
    )
    baseline = estimate_local_ground(cells)
    fast = estimate_local_ground_fast(cells)
    np.testing.assert_array_equal(fast, baseline)


def test_fast_rate_path_matches_baseline(monkeypatch):
    cells = make_cells(
        (0, 0, 0.0, 0.0, 1, 1.0),
        (0, 1, 0.1, 0.4, 1, 1.0),
        (1, 0, 0.0, 0.0, 2, 1.0, True),
    )
    baseline = rate_cells(cells)
    monkeypatch.setenv("FOVMAP_RATING_IMPL", "fast")
    fast = rate_cells(cells)
    np.testing.assert_array_equal(fast, baseline)


def test_compiled_ground_matches_baseline(monkeypatch):
    cells = make_cells(
        (-2, -2, 1.0, 1.1, 1, 1.0),
        (-2, -1, 1.1, 1.2, 1, 1.0),
        (-1, -2, np.nan, 1.3, 1, 1.0),
        (0, 0, 2.0, 2.1, 1, 1.0),
        (0, 1, np.inf, 2.2, 1, 1.0),
        (1, 0, 2.2, 2.3, 2, 0.2, True),
    )
    baseline = estimate_local_ground(cells)
    baseline_ratings = rate_cells(cells)
    compiled = estimate_local_ground_compiled(cells)
    np.testing.assert_array_equal(compiled, baseline)
    monkeypatch.setenv("FOVMAP_RATING_IMPL", "compiled")
    np.testing.assert_array_equal(rate_cells(cells), baseline_ratings)


def test_compiled_duplicate_keys_route_to_baseline():
    cells = make_cells(
        (0, 0, 0.0, 0.0, 1, 1.0),
        (0, 0, 1.0, 1.0, 1, 1.0),
    )
    np.testing.assert_array_equal(
        estimate_local_ground_compiled(cells),
        estimate_local_ground(cells),
    )
