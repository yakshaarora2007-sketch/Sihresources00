import numpy as np
from pathlib import Path


RING_BOUNDARIES = np.array([10.0, 25.0, 50.0], dtype=np.float32)
RING_RESOLUTIONS = np.array([0.05, 0.10, 0.25, 0.50], dtype=np.float32)

CELL_DTYPE = np.dtype([
    ('cell_key', np.int64),
    ('ring_id', np.uint8),
    ('row', np.int32),
    ('col', np.int32),
    ('point_count', np.uint32),
    ('z_mean', np.float32),
    ('z_min', np.float32),
    ('z_max', np.float32),
    ('semantic_label', np.uint8),
    ('confidence', np.float32),
    ('dynamic_flag', np.bool_),
    ('timestamp', np.uint32),
])

SEMANTIC_KITTI_REMAP = np.array([
    0, 0, 3, 3, 3, 3, 3, 3, 3, 3, 2, 1, 1, 2, 0, 0, 0, 0, 0, 0, 0,
], dtype=np.uint8)


def assign_rings(ranges: np.ndarray) -> np.ndarray:
    """Assign each point to a foveated ring using half-open intervals [low, high)."""
    return np.searchsorted(RING_BOUNDARIES, ranges, side='right').astype(np.int32)


def compute_cell_keys(points_xy: np.ndarray, ring_ids: np.ndarray):
    """Compute integer cell keys for each point using floor-division quantization."""
    x = points_xy[:, 0]
    y = points_xy[:, 1]
    res = RING_RESOLUTIONS[ring_ids]

    col = np.floor(x / res).astype(np.int64)
    row = np.floor(y / res).astype(np.int64)

    OFFSET = 1 << 23
    ring_shifted = (ring_ids.astype(np.int64) << 48)
    row_shifted = ((row + OFFSET).astype(np.int64) << 24)
    col_shifted = (col + OFFSET).astype(np.int64)

    cell_keys = ring_shifted | row_shifted | col_shifted
    return cell_keys, row, col


def build_cell_schema(points_xy: np.ndarray, points_z: np.ndarray, ring_ids: np.ndarray,
                       semantic_labels: np.ndarray = None, confidences: np.ndarray = None,
                       timestamp: int = 0) -> np.ndarray:
    """
    Build the full cell schema array with all fields Person 5 needs.

    Args:
        points_xy: (N, 2) array of (x, y) in meters
        points_z: (N,) array of z in meters
        ring_ids: (N,) array of ring indices (0-3)
        semantic_labels: (N,) optional, 4-class labels (0=TERRAIN,1=DRIVABLE,2=STATIC,3=OBJECT)
        confidences: (N,) optional, per-point confidence scores [0,1]
        timestamp: current scan timestamp/frame index

    Returns:
        Structured array with dtype CELL_DTYPE, one row per unique cell
    """
    n_points = len(points_xy)
    if semantic_labels is None:
        semantic_labels = np.zeros(n_points, dtype=np.uint8)
    if confidences is None:
        confidences = np.ones(n_points, dtype=np.float32)

    cell_keys, rows, cols = compute_cell_keys(points_xy, ring_ids)

    unique_keys, inverse_indices, counts = np.unique(
        cell_keys, return_inverse=True, return_counts=True
    )

    z_sum = np.bincount(inverse_indices, weights=points_z)
    z_mean = z_sum / counts

    z_min = np.full_like(unique_keys, np.inf, dtype=np.float32)
    z_max = np.full_like(unique_keys, -np.inf, dtype=np.float32)
    np.minimum.at(z_min, inverse_indices, points_z)
    np.maximum.at(z_max, inverse_indices, points_z)

    label_sum = np.bincount(inverse_indices, weights=semantic_labels.astype(np.float32))
    cell_semantic = np.round(label_sum / counts).astype(np.uint8)

    conf_sum = np.bincount(inverse_indices, weights=confidences)
    cell_confidence = conf_sum / counts

    ring_ids_per_cell = (unique_keys >> 48) & 0xFFFF
    rows_per_cell = ((unique_keys >> 24) & 0xFFFFFF) - (1 << 23)
    cols_per_cell = (unique_keys & 0xFFFFFF) - (1 << 23)

    cells = np.empty(len(unique_keys), dtype=CELL_DTYPE)
    cells['cell_key'] = unique_keys
    cells['ring_id'] = ring_ids_per_cell.astype(np.uint8)
    cells['row'] = rows_per_cell.astype(np.int32)
    cells['col'] = cols_per_cell.astype(np.int32)
    cells['point_count'] = counts.astype(np.uint32)
    cells['z_mean'] = z_mean.astype(np.float32)
    cells['z_min'] = z_min.astype(np.float32)
    cells['z_max'] = z_max.astype(np.float32)
    cells['semantic_label'] = cell_semantic
    cells['confidence'] = cell_confidence.astype(np.float32)
    cells['dynamic_flag'] = False
    cells['timestamp'] = np.full(len(unique_keys), timestamp, dtype=np.uint32)

    return cells


def unpack_cell_key(cell_keys: np.ndarray):
    """Unpack cell_key -> (ring_id, row, col)."""
    ring_ids = (cell_keys >> 48) & 0xFFFF
    rows = ((cell_keys >> 24) & 0xFFFFFF) - (1 << 23)
    cols = (cell_keys & 0xFFFFFF) - (1 << 23)
    return ring_ids.astype(np.int32), rows.astype(np.int32), cols.astype(np.int32)


def pack_cell_key(ring_ids: np.ndarray, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    """Pack (ring_id, row, col) -> cell_key."""
    OFFSET = 1 << 23
    ring_shifted = (ring_ids.astype(np.int64) << 48)
    row_shifted = ((rows + OFFSET).astype(np.int64) << 24)
    col_shifted = (cols + OFFSET).astype(np.int64)
    return ring_shifted | row_shifted | col_shifted


def transform_points(points_xyz: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Apply 4x4 transform T to (N,3) points."""
    homogeneous = np.hstack([points_xyz, np.ones((len(points_xyz), 1), dtype=np.float32)])
    transformed = (T @ homogeneous.T).T[:, :3]
    return transformed.astype(np.float32)


def rebin_stored_map(stored_cells: np.ndarray, T_rel: np.ndarray) -> np.ndarray:
    """
    Re-bin stored map cells to current robot-centric frame (Step 8 fusion).

    1. Transform stored cell centers by relative pose T_rel
    2. Re-quantize to ring/row/col using SAME half-open/floor logic
    3. Handle nested-lattice coarsen/refine via exact integer multiples

    Args:
        stored_cells: structured array with CELL_DTYPE from previous frame
        T_rel: 4x4 relative pose (current <- previous), float32

    Returns:
        Re-binned cells in current frame, same CELL_DTYPE
    """
    if len(stored_cells) == 0:
        return np.array([], dtype=CELL_DTYPE)

    ring_ids = stored_cells['ring_id'].astype(np.int32)
    rows = stored_cells['row'].astype(np.int32)
    cols = stored_cells['col'].astype(np.int32)

    res = RING_RESOLUTIONS[ring_ids]
    OFFSET = 1 << 23

    x_centers = (cols.astype(np.float32) + 0.5) * res
    y_centers = (rows.astype(np.float32) + 0.5) * res
    z_centers = stored_cells['z_mean']

    centers = np.column_stack([x_centers, y_centers, z_centers])
    centers_transformed = transform_points(centers, T_rel)

    new_ranges = np.sqrt(centers_transformed[:, 0]**2 + centers_transformed[:, 1]**2).astype(np.float32)
    new_ring_ids = assign_rings(new_ranges)

    new_points_xy = centers_transformed[:, :2]
    new_cell_keys, new_rows, new_cols = compute_cell_keys(new_points_xy, new_ring_ids)

    unique_keys, inverse_indices = np.unique(new_cell_keys, return_inverse=True)

    point_counts = np.bincount(inverse_indices, weights=stored_cells['point_count'])
    z_sums = np.bincount(inverse_indices, weights=stored_cells['z_mean'] * stored_cells['point_count'])
    z_means = z_sums / point_counts

    z_mins = np.full_like(unique_keys, np.inf, dtype=np.float32)
    z_maxs = np.full_like(unique_keys, -np.inf, dtype=np.float32)
    np.minimum.at(z_mins, inverse_indices, stored_cells['z_min'])
    np.maximum.at(z_maxs, inverse_indices, stored_cells['z_max'])

    label_sums = np.bincount(inverse_indices, weights=stored_cells['semantic_label'].astype(np.float32) * stored_cells['point_count'])
    cell_semantics = np.round(label_sums / point_counts).astype(np.uint8)

    conf_sums = np.bincount(inverse_indices, weights=stored_cells['confidence'] * stored_cells['point_count'])
    cell_confidences = conf_sums / point_counts

    timestamps = np.bincount(inverse_indices, weights=stored_cells['timestamp'].astype(np.float32) * stored_cells['point_count'])
    cell_timestamps = np.round(timestamps / point_counts).astype(np.uint32)

    dynamic_flags = np.bincount(inverse_indices, weights=stored_cells['dynamic_flag'].astype(np.float32) * stored_cells['point_count'])
    cell_dynamic = (dynamic_flags / point_counts > 0.5)

    new_ring_ids_per_cell = (unique_keys >> 48) & 0xFFFF
    new_rows_per_cell = ((unique_keys >> 24) & 0xFFFFFF) - (1 << 23)
    new_cols_per_cell = (unique_keys & 0xFFFFFF) - (1 << 23)

    rebinned = np.empty(len(unique_keys), dtype=CELL_DTYPE)
    rebinned['cell_key'] = unique_keys
    rebinned['ring_id'] = new_ring_ids_per_cell.astype(np.uint8)
    rebinned['row'] = new_rows_per_cell.astype(np.int32)
    rebinned['col'] = new_cols_per_cell.astype(np.int32)
    rebinned['point_count'] = point_counts.astype(np.uint32)
    rebinned['z_mean'] = z_means.astype(np.float32)
    rebinned['z_min'] = z_mins.astype(np.float32)
    rebinned['z_max'] = z_maxs.astype(np.float32)
    rebinned['semantic_label'] = cell_semantics
    rebinned['confidence'] = cell_confidences.astype(np.float32)
    rebinned['dynamic_flag'] = cell_dynamic
    rebinned['timestamp'] = cell_timestamps

    return rebinned


def load_semantickitti_scan(bin_path: str) -> np.ndarray:
    """Load .bin scan -> (N, 4) float32 [x, y, z, intensity]."""
    scan = np.fromfile(bin_path, dtype=np.float32)
    if scan.size % 4 != 0:
        raise ValueError(f"Invalid scan size: {scan.size} not divisible by 4")
    return scan.reshape(-1, 4)


def load_semantickitti_labels(label_path: str) -> np.ndarray:
    """Load .label file -> (N,) uint32, lower 16 bits = semantic class."""
    labels = np.fromfile(label_path, dtype=np.uint32)
    return labels & 0xFFFF


def remap_labels(labels_19: np.ndarray) -> np.ndarray:
    """Map 19-class SemanticKITTI labels to 4-class groups."""
    return SEMANTIC_KITTI_REMAP[labels_19]


def process_scan(bin_path: str, label_path: str, timestamp: int = 0):
    """Full pipeline: load scan + labels -> build cell schema."""
    scan = load_semantickitti_scan(bin_path)
    labels_19 = load_semantickitti_labels(label_path)

    if len(scan) != len(labels_19):
        raise ValueError(f"Point/label count mismatch: {len(scan)} vs {len(labels_19)}")

    x, y, z = scan[:, 0], scan[:, 1], scan[:, 2]
    ranges = np.sqrt(x * x + y * y).astype(np.float32)

    ring_ids = assign_rings(ranges)
    labels_4 = remap_labels(labels_19)
    confidences = np.ones_like(ranges, dtype=np.float32)
    points_xy = np.column_stack([x, y])

    cells = build_cell_schema(points_xy, z, ring_ids, labels_4, confidences, timestamp)
    return cells, scan, labels_4, ring_ids


def compute_memory_table(points_xy: np.ndarray, points_z: np.ndarray, ring_ids: np.ndarray):
    """Compute cell counts for foveated vs uniform 5cm vs uniform 50cm."""
    foveated_cells = build_cell_schema(points_xy, points_z, ring_ids)

    uniform_5cm_ids = np.zeros(len(ring_ids), dtype=np.int32)
    uniform_5cm_cells = build_cell_schema(points_xy, points_z, uniform_5cm_ids)

    uniform_50cm_ids = np.full(len(ring_ids), 3, dtype=np.int32)
    uniform_50cm_cells = build_cell_schema(points_xy, points_z, uniform_50cm_ids)

    return {
        'foveated': len(foveated_cells),
        'uniform_5cm': len(uniform_5cm_cells),
        'uniform_50cm': len(uniform_50cm_cells),
        'foveated_mem_kb': foveated_cells.nbytes / 1024,
        'uniform_5cm_mem_kb': uniform_5cm_cells.nbytes / 1024,
        'uniform_50cm_mem_kb': uniform_50cm_cells.nbytes / 1024,
    }


def find_semantickitti_dirs():
    """Try to locate SemanticKITTI sequence 08 directories."""
    possible_roots = [
        Path.home() / "data" / "SemanticKITTI" / "dataset" / "sequences" / "08",
        Path.home() / "SemanticKITTI" / "dataset" / "sequences" / "08",
        Path("/data/SemanticKITTI/dataset/sequences/08"),
        Path("C:/data/SemanticKITTI/dataset/sequences/08"),
        Path("D:/SemanticKITTI/dataset/sequences/08"),
        Path("data/sequences/08"),
        Path("sequences/08"),
    ]

    for root in possible_roots:
        velodyne = root / "velodyne"
        labels = root / "labels"
        if velodyne.exists() and labels.exists():
            return str(velodyne), str(labels)

    return None, None


if __name__ == "__main__":
    # Quick synthetic validation
    rng = np.random.default_rng(42)
    n = 50000
    ranges = rng.uniform(0, 80, n).astype(np.float32)
    angles = rng.uniform(-np.pi, np.pi, n).astype(np.float32)
    x = ranges * np.cos(angles)
    y = ranges * np.sin(angles)
    z = rng.uniform(-2, 3, n).astype(np.float32)

    ring_ids = assign_rings(ranges)
    points_xy = np.column_stack([x, y])

    cells = build_cell_schema(points_xy, z, ring_ids, timestamp=0)

    # Gate 1: zero-loss
    total_counted = cells['point_count'].sum()
    assert total_counted == n, f"Point loss: {total_counted} != {n}"

    # Gate 2: no duplicate keys
    unique_keys = np.unique(cells['cell_key'])
    assert len(unique_keys) == len(cells), "Duplicate cell keys found"

    # Gate 3: boundary determinism
    boundary_ranges = np.array([10.0, 25.0, 50.0], dtype=np.float32)
    boundary_rings = assign_rings(boundary_ranges)
    assert boundary_rings[0] == 1 and boundary_rings[1] == 2 and boundary_rings[2] == 3

    # Gate 4: all schema fields present and valid
    for field in CELL_DTYPE.names:
        data = cells[field]
        if data.dtype.kind == 'f':
            assert not np.any(np.isnan(data)), f"NaN in {field}"
    assert np.all(cells['timestamp'] == 0)
    assert np.all(cells['dynamic_flag'] == False)
    assert np.all(cells['semantic_label'] < 4)
    assert np.all(cells['confidence'] >= 0) and np.all(cells['confidence'] <= 1)
    assert np.all(cells['ring_id'] < 4)

    print("[OK] All synthetic gates passed")

    # Re-binning tests
    T_identity = np.eye(4, dtype=np.float32)
    rebinned = rebin_stored_map(cells, T_identity)
    matches = np.sum(np.isin(cells['cell_key'], rebinned['cell_key']))
    assert matches / len(cells) > 0.99

    # Point count preservation
    T = np.eye(4, dtype=np.float32)
    T[0, 3] = 1.0
    rebinned = rebin_stored_map(cells, T)
    assert cells['point_count'].sum() == rebinned['point_count'].sum()

    # Memory table
    mem = compute_memory_table(points_xy, z, ring_ids)
    print(f"Memory: Foveated={mem['foveated_mem_kb']:.1f}KB, Uniform5cm={mem['uniform_5cm_mem_kb']:.1f}KB, Uniform50cm={mem['uniform_50cm_mem_kb']:.1f}KB")

    # Real data if available
    seq_dir, label_dir = find_semantickitti_dirs()
    if seq_dir is not None:
        print(f"Found SemanticKITTI seq 08 at: {Path(seq_dir).parent}")
        bin_files = sorted(Path(seq_dir).glob("*.bin"))[:5]
        label_files = sorted(Path(label_dir).glob("*.label"))[:5]
        for i, (bin_f, label_f) in enumerate(zip(bin_files, label_files)):
            cells, _, _, _ = process_scan(str(bin_f), str(label_f), timestamp=i)
            total = cells['point_count'].sum()
            scan = load_semantickitti_scan(str(bin_f))
            assert total == len(scan), f"Frame {i}: point loss"
            print(f"  Frame {i}: {len(scan)} pts -> {len(cells)} cells")
        print("[OK] Real data gates passed")
    else:
        print("[INFO] SemanticKITTI not found — synthetic validation only")

    print("\n[SUCCESS] grid_engine.py ready for Person 5")