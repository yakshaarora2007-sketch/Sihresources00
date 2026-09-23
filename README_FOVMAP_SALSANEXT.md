# FOVMAP + SalsaNext: Quickstart & Pipeline Guide

Welcome! This guide explains how to use the **FOVMAP** LiDAR data pipeline and connect it with the **SalsaNext** semantic segmentation model.

Whether you are running verification tests, loading LiDAR sweeps, or preparing range images for neural network inference, this document will get you up to speed in minutes.

---

## ⚡ Quickstart in 30 Seconds

Follow these three steps to get everything running right away.

### 1. Open your terminal in this repository
```bash
cd /path/to/Numpytest
```

### 2. Activate the Python environment
This project uses a pre-configured virtual environment with NumPy installed:
```bash
source .venv/bin/activate
```
*(If your terminal does not have `python` on your path, you can also run commands directly with `./.venv/bin/python`)*

### 3. Run the pipeline verification
The sample dataset includes KITTI Sequence **08**. Run the verification script with the `-s 08` flag:
```bash
python verify_gates.py -s 08
```

You should see all 7 verification gates pass cleanly:
```text
=== GATE A1: Read one .bin scan (000000.bin) [Seq: 08] ===
 Gate A1 PASSED!
=== GATE A2: Load poses.txt [Seq: 08] ===
 Gate A2 PASSED!
=== GATE A3: Parse Tr from calib.txt [Seq: 08] ===
 Gate A3 PASSED!
=== GATE A4: SemanticKITTILoader Frame 0 Access [Seq: 08] ===
 Gate A4 PASSED!
=== GATE A5: Full-Sequence Multi-Frame Iteration [Seq: 08] ===
 Gate A5 PASSED! All frames iterated cleanly.
=== GATE B1: Relative Pose T_rel Verification [Seq: 08] ===
 Gate B1 PASSED! Relative poses match ground truth perfectly.
=== GATE REMAP: 19->4 Semantic Mapping ===
 Gate REMAP PASSED! Lookup table mapping is correct.
```

---

## 🧭 What Does This Code Do?

In self-driving systems, a LiDAR sensor spins on top of the car, recording hundreds of thousands of 3D distance points every second. Our goal is to take those raw points and turn them into a real-time, 2.5D elevation and obstacle map around the vehicle.

Here is how data flows from the raw sensor to the final map:

```text
Raw KITTI LiDAR (.bin file)
       │
       ▼
[fovmap.data.loader]
       │  • Reads (N, 4) point cloud: [x, y, z, intensity]
       │  • Computes relative car motion between frames: T_rel (4, 4)
       ▼
[Range Image Projection]
       │  • Flattens 3D points into a 2D cylindrical range image: (5, 64, 2048)
       ▼
[SalsaNext Neural Network]
       │  • Classifies each pixel into 20 classes (road, car, person, tree, etc.)
       ▼
[Point Unprojection]
       │  • Projects 2D predicted labels back onto the 3D points
       ▼
[fovmap.data.remap]
       │  • Compresses predictions into 4 clean categories:
       │    TERRAIN (0), DRIVABLE (1), STATIC (2), OBJECT (3)
       ▼
[2.5D Grid Engine]
       • Builds concentric elevation rings (dense nearby, coarse further away)
```

---

## 📁 Repository Tour

Here are the key files and directories you will interact with:

```text
├── README_FOVMAP_SALSANEXT.md   # This guide!
├── verify_gates.py              # Test script verifying data integrity and poses
├── src/
│   └── fovmap/
│       └── data/
│           ├── __init__.py
│           ├── loader.py        # Fast scan reading, calibration parsing & T_rel
│           └── remap.py         # Instant 20-to-4 class taxonomy lookup tables
└── sample_kitti/
    └── sequences/
        └── 08/                  # Minimal 12-frame sample data (scans, poses, calib)
```

---

## 🛠️ Step-by-Step Code Examples

### 1. Reading a Single LiDAR Scan

Every `.bin` file in the KITTI dataset stores 3D points as raw 32-bit floats (`x`, `y`, `z`, `intensity`).

```python
import sys
from pathlib import Path

# Add src to python path
sys.path.insert(0, str(Path("src").resolve()))
from fovmap.data.loader import load_point_cloud

# Load frame 0 of sequence 08
bin_path = Path("sample_kitti/sequences/08/velodyne/000000.bin")
points = load_point_cloud(bin_path)

print(f"Points shape : {points.shape}")  # e.g., (17983, 4)
print(f"Data type    : {points.dtype}")  # float32
print(f"First point  : x={points[0,0]:.2f}m, y={points[0,1]:.2f}m, z={points[0,2]:.2f}m, intensity={points[0,3]:.2f}")
```

---

### 2. Loading an Entire Sequence with `SemanticKITTILoader`

The `SemanticKITTILoader` class acts like a list of frames. You can look up any frame by its index or loop through the entire sequence:

```python
from pathlib import Path
from fovmap.data.loader import SemanticKITTILoader

# Open the sequence directory
seq_dir = Path("sample_kitti/sequences/08")
loader = SemanticKITTILoader(seq_dir)

print(f"Total frames in sequence: {len(loader)}")

# Access a specific frame
points, pose_curr, pose_prev, Tr, frame_id = loader[0]
print(f"Loaded frame {frame_id} with {len(points)} points")

# Loop over the whole sequence
for points, pose, pose_prev, Tr, frame_id in loader:
    car_x, car_y, car_z = pose[0, 3], pose[1, 3], pose[2, 3]
    print(f"Frame {frame_id:02d} | Car position: x={car_x:.1f}m, y={car_y:.1f}m, z={car_z:.1f}m")
```

---

### 3. Understanding Relative Motion (`T_rel`)

When building a map while driving, we keep the map **robot-centric** (the car is always at `[0, 0, 0]` in the current frame). To combine the new scan with older scans, we need to know how much the car moved between frames:

```text
T_rel = inv(Tr) @ inv(pose_curr) @ pose_prev @ Tr
```

- `pose_curr` and `pose_prev` are the car's global positions in camera coordinates.
- `Tr` converts coordinates from the Velodyne LiDAR frame to the camera frame.
- `T_rel` tells us: *"How does point (x, y, z) from the previous LiDAR sweep move into the current LiDAR frame?"*

```python
# Calculate relative motion from previous frame to current frame
T_rel = loader.get_T_rel(frame_index=1)

print("T_rel 4x4 matrix:\n", T_rel)
print("Forward distance traveled:", T_rel[2, 3], "meters")
```

> [!TIP]
> **Golden Rule of Mapping**:
> Never transform the **current** point cloud by `T_rel`! The current scan is already in the right place (centered at the car). You only apply `T_rel` to shift **older** map data backward when merging history into the current view.

---

### 4. Turning Point Clouds into SalsaNext Range Images

SalsaNext is a convolutional neural network designed for images. Before feeding a 3D point cloud into SalsaNext, we project the points onto a 2D spherical range image:
- **Height (H)**: 64 rows (matching the 64 laser beams of the Velodyne HDL-64E).
- **Width (W)**: 2048 columns (representing 360 degrees around the vehicle).
- **5 Channels**: `[range, x, y, z, intensity]`.

Here is the spherical math in plain text:
- Range (distance): `r = sqrt(x^2 + y^2 + z^2)`
- Horizontal angle (yaw): `yaw = -atan2(y, x)`
- Vertical angle (pitch): `pitch = arcsin(z / r)`
- Column index `u`: `u = 0.5 * (1.0 - yaw / pi) * 2048`
- Row index `v`: `v = (1.0 - (pitch + 25 degrees) / 28 degrees) * 64`

Points closer to the sensor take priority if multiple points land on the same pixel.

---

### 5. Fast Semantic Remapping (20 Classes ➔ 4 Classes)

SalsaNext outputs 20 different learning classes (like `1` for car, `9` for road, `13` for building). For navigation and grid mapping, we simplify these into **4 high-level classes**:

| ID | Class Name | Description | Included KITTI Objects |
|:--:|:-----------|:------------|:-----------------------|
| **0** | **TERRAIN** | Ground not safe or intended for normal driving | Dirt, grass, vegetation, sidewalk |
| **1** | **DRIVABLE** | Safe driving surfaces | Road, highway, parking lots, lane markings |
| **2** | **STATIC** | Fixed obstacles the vehicle cannot drive through | Buildings, poles, fences, tree trunks |
| **3** | **OBJECT** | Movable or dynamic actors | Cars, trucks, pedestrians, cyclists |

Instead of slow Python `if-else` checks, we use a 2-step lookup table:

```python
import numpy as np
from fovmap.data.remap import map_to_4_classes, LEARNING_MAP_INV

# 1. Suppose SalsaNext predicts these 4 classes for 4 points:
#    9 (road), 13 (building), 1 (car), 17 (terrain)
salsanext_predictions = np.array([9, 13, 1, 17], dtype=np.uint32)

# 2. Step 1: Map SalsaNext learning IDs to raw KITTI IDs (instant array indexing)
raw_kitti_ids = LEARNING_MAP_INV[salsanext_predictions]
# Result: [40 (road), 50 (building), 10 (car), 72 (terrain)]

# 3. Step 2: Map raw KITTI IDs to our 4-class taxonomy
final_classes = map_to_4_classes(raw_kitti_ids)
# Result: [1 (DRIVABLE), 2 (STATIC), 3 (OBJECT), 0 (TERRAIN)]

print("Final 4-class output:", final_classes)
```

> [!NOTE]
> What about points lost during 2D range projection?
> Points that could not be projected (e.g. filtered out by minimum range or hidden behind nearer points) are given label `255` (`UNKNOWN`). This prevents unprojected points from accidentally being marked as `TERRAIN` (0).

---

## 🧪 The Verification Suite (`verify_gates.py`)

We provide an automated gate verification script to ensure all data pipelines, math calculations, and lookup tables are working properly:

```bash
# Verify the sample sequence
python verify_gates.py -s 08

# Verify a full dataset sequence (e.g., sequence 04)
python verify_gates.py -d /path/to/kitti/dataset -s 04

# Provide a specific ground-truth relative pose file
python verify_gates.py -s 08 -g sample_kitti/sequences/08/GROUND_TRUTH_T_rel.txt
```

### CLI Flags Cheat Sheet

| Flag | Long Flag | Default | What It Does |
|---|---|---|---|
| `-s` | `--seq` | `"04"` | Sequence ID to test (e.g. `08` for sample data) |
| `-d` | `--data-root` | `sample_kitti` | Path to folder containing the `sequences/` directory |
| `-g` | `--gt-file` | `None` | Optional explicit path to `GROUND_TRUTH_T_rel.txt` |

### What Each Gate Checks

1. **Gate A1 (Scan Reader)**: Verifies that `.bin` point clouds load with correct 4-float fields (`x, y, z, intensity`), file sizes match expected byte counts, values are finite, and intensity is within `[0.0, 1.0]`.
2. **Gate A2 (Poses)**: Checks that `poses.txt` loads 12-column matrices into `(4, 4)` transformations, frame 0 is an identity matrix, and vehicle translation between frames is reasonable (0.1m to 5.0m).
3. **Gate A3 (Calibration)**: Verifies that `calib.txt` parses the `Tr` sensor-to-camera matrix, the bottom row is `[0, 0, 0, 1]`, and the rotation submatrix determinant is 1.0.
4. **Gate A4 (Single Frame Access)**: Tests `SemanticKITTILoader[0]` to ensure frame 0 retrieval is fast and accurate.
5. **Gate A5 (Sequence Iteration)**: Loops through all scans in the sequence to ensure none are empty or corrupt.
6. **Gate B1 (Relative Motion T_rel)**: Compares computed `T_rel` transformations against ground truth. Fails with an error if the ground-truth file is missing or contains no valid transitions.
7. **Gate REMAP (4-Class Taxonomy)**: Verifies the 260-element and 20-element lookup tables against reference classes.

## 🖼️ Render the 2.5D Foveated Grid

After map replay has produced snapshots, render the latest sequence-08 map with:

```bash
python visualize_2_5d.py
```

The renderer reads only:

```text
SalsaNext-Fork/predictions/valid/maps/sequences/08/*.npz
```

and writes the PNG to:

```text
SalsaNext-Fork/predictions/valid/maps/visualizations/08/
```

To render a particular frame:

```bash
python visualize_2_5d.py --frame 000009
```

The four panels show the current semantic grid, elevation, traversability,
and foveated resolution rings. Dynamic cells are outlined on the ring view.
Visualization is intentionally separate from the headless timed pipeline.

---

## ❓ Common Questions & Troubleshooting

### Q: I see `zsh: command not found: python`
**A:** The system Python may not be linked in your terminal. Either activate the virtual environment:
```bash
source .venv/bin/activate
```
or run directly with:
```bash
./.venv/bin/python verify_gates.py -s 08
```

### Q: `python verify_gates.py` fails with `File not found ... sequences/04`
**A:** By default, `verify_gates.py` looks for sequence `"04"` (the standard benchmark sequence in full SemanticKITTI). The sample dataset in this repository only bundles sequence `"08"`. Simply pass `-s 08`:
```bash
python verify_gates.py -s 08
```

### Q: Why do we use pure NumPy instead of PyTorch or Open3D?
**A:** Speed and simplicity! Data loading, coordinate math, and lookup table indexing with NumPy are extremely fast (sub-millisecond per frame) and avoid heavy external dependencies or GPU transfers during preprocessing.

### Q: What should I do if a ground-truth file is missing for my sequence?
**A:** Gate B1 checks your computed relative poses against known ground-truth transitions. If you have your own ground truth file, provide it using the `-g` flag:
```bash
python verify_gates.py -s <seq_id> -g /path/to/GROUND_TRUTH_T_rel.txt
```
If no ground truth is provided, the script will let you know and exit with an error.

---

## 🚦 Core Guidelines for Contributors

1. **Keep the hot path in pure NumPy**: Don't introduce heavy libraries (like `open3d` or `scikit-learn`) into `fovmap.data`.
2. **Never drop points silently**: Every valid LiDAR point should either map to a grid cell or be counted as out-of-bounds.
3. **Keep the robot at the center**: Point clouds in the current frame should always stay in the vehicle's reference frame. Older maps are transformed to match the vehicle, not the other way around.
