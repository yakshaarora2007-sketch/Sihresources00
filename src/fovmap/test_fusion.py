import numpy as np

from .grid_engine import CELL_DTYPE, pack_cell_key, rebin_stored_map
from .dynamics_fusion import (
    FUSED_CELL_DTYPE,
    carry_log_odds_through_rebin,
    fuse_log_odds,
    process_frame,
)


def cell(label=3, confidence=0.9, row=0, col=0, dynamic=False):
    result = np.zeros(1, dtype=CELL_DTYPE)
    result["cell_key"] = pack_cell_key(np.array([0]), np.array([row]), np.array([col]))
    result["row"], result["col"] = row, col
    result["point_count"] = 3
    result["semantic_label"] = label
    result["confidence"] = confidence
    result["dynamic_flag"] = dynamic
    return result


def probability(log_odds):
    return 1.0 / (1.0 + np.exp(-log_odds))


def test_consistent_observations_increase_confidence():
    first = fuse_log_odds(cell(), np.array([], dtype=CELL_DTYPE), np.array([False]))
    second = fuse_log_odds(cell(), first, np.array([False]))
    assert probability(second["log_odds"][0]) > probability(first["log_odds"][0])


def test_dynamic_cell_keeps_prior_without_update():
    prior = fuse_log_odds(cell(), np.array([], dtype=CELL_DTYPE), np.array([False]))
    result = fuse_log_odds(cell(label=0, dynamic=True), prior, np.array([True]))
    assert result["log_odds"][0] == prior["log_odds"][0]


def test_unobserved_cell_decays_toward_neutral():
    prior = fuse_log_odds(cell(), np.array([], dtype=CELL_DTYPE), np.array([False]))
    result = fuse_log_odds(
        np.array([], dtype=CELL_DTYPE),
        prior,
        np.array([], dtype=bool),
        lambda_decay=0.5,
    )
    assert abs(result["log_odds"][0]) < abs(prior["log_odds"][0])


def test_pruning_keeps_99m_and_drops_over_100m():
    current = np.concatenate((cell(row=0, col=1980), cell(row=0, col=2021)))
    result = fuse_log_odds(current, np.array([], dtype=CELL_DTYPE), np.array([False, False]))
    assert len(result) == 1
    assert result["col"][0] == 1980


def test_extreme_confidence_stays_finite():
    result = fuse_log_odds(cell(confidence=1.0), np.array([], dtype=CELL_DTYPE), np.array([False]))
    assert np.isfinite(result["log_odds"]).all()
    assert np.all((probability(result["log_odds"]) > 0) & (probability(result["log_odds"]) < 1))


def test_process_frame_rebins_then_detects_then_fuses():
    transform = np.eye(4, dtype=np.float32)
    first, first_mask, _ = process_frame(
        cell(label=1),
        np.array([], dtype=CELL_DTYPE),
        transform,
    )
    second, second_mask, rebinned = process_frame(
        cell(label=3),
        first,
        transform,
    )
    assert not first_mask[0]
    assert second_mask[0]
    assert rebinned["cell_key"][0] == cell(label=1)["cell_key"][0]
    assert second["dynamic_flag"][0]
    assert second["log_odds"][0] == first["log_odds"][0]


def fused_cell(label=3, confidence=0.9, row=0, col=0, point_count=1, log_odds=0):
    result = np.zeros(1, dtype=FUSED_CELL_DTYPE)
    result["cell_key"] = pack_cell_key(np.array([0]), np.array([row]), np.array([col]))
    result["row"], result["col"] = row, col
    result["ring_id"] = 0
    result["point_count"] = point_count
    result["z_mean"] = result["z_min"] = result["z_max"] = 0
    result["semantic_label"] = label
    result["confidence"] = confidence
    result["log_odds"] = log_odds
    return result


def test_coarsening_uses_point_count_weighted_log_odds_average():
    fine = np.concatenate((
        fused_cell(row=0, col=198, point_count=1, log_odds=1),
        fused_cell(row=0, col=199, point_count=2, log_odds=2),
        fused_cell(row=1, col=198, point_count=3, log_odds=3),
        fused_cell(row=1, col=199, point_count=4, log_odds=4),
    ))
    transform = np.eye(4, dtype=np.float32)
    transform[0, 3] = 1.0
    rebinned = rebin_stored_map(fine, transform)
    carried = carry_log_odds_through_rebin(fine, rebinned, transform)
    expected = np.average([1, 2, 3, 4], weights=[1, 2, 3, 4])
    assert len(carried) == 1
    assert np.isclose(carried["log_odds"][0], expected)


def test_refined_output_inherits_single_coarse_prior_exactly():
    coarse = fused_cell(row=0, col=1, point_count=7, log_odds=4.25)
    coarse["ring_id"] = 3
    coarse["cell_key"] = pack_cell_key(np.array([3]), np.array([0]), np.array([1]))
    transform = np.eye(4, dtype=np.float32)
    rebinned = rebin_stored_map(coarse, transform)
    carried = carry_log_odds_through_rebin(coarse, rebinned, transform)
    assert len(carried) == 1
    assert carried["ring_id"][0] < 3
    assert carried["log_odds"][0] == np.float32(4.25)


def test_confident_cell_survives_ring_shift_without_reset():
    fine = np.concatenate((
        fused_cell(row=0, col=198, point_count=3, log_odds=5),
        fused_cell(row=0, col=199, point_count=3, log_odds=5),
        fused_cell(row=1, col=198, point_count=3, log_odds=5),
        fused_cell(row=1, col=199, point_count=3, log_odds=5),
    ))
    transform = np.eye(4, dtype=np.float32)
    transform[0, 3] = 1.0
    fused, dynamic_mask, rebinned = process_frame(
        np.array([], dtype=CELL_DTYPE),
        fine,
        transform,
    )
    assert dynamic_mask.size == 0
    assert len(rebinned) == 1
    assert probability(fused["log_odds"][0]) > 0.99
