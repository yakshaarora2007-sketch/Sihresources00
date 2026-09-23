"""Render the foveated 2.5D map snapshots produced by the replay pipeline.

The renderer is deliberately separate from the timed pipeline. It consumes
the existing ``.npz`` snapshots and writes a PNG, so dashboard or plotting
work cannot affect latency measurements.

Example:
    python visualize_2_5d.py --frame 000009
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap


RING_RESOLUTIONS = np.array([0.05, 0.10, 0.25, 0.50], dtype=np.float32)
SEMANTIC_NAMES = ("TERRAIN", "DRIVABLE", "STATIC", "OBJECT")
SEMANTIC_COLORS = ("#d9a441", "#2ca25f", "#64748b", "#d94841")
RING_COLORS = ("#2166ac", "#67a9cf", "#fdae61", "#d73027")


def _frame_path(maps_root: Path, sequence: str, frame: str) -> Path:
    frame_name = f"{int(frame):06d}.npz"
    return maps_root / "sequences" / f"{int(sequence):02d}" / frame_name


def _cell_centers(cells: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ring_ids = np.clip(cells["ring_id"].astype(np.int64), 0, 3)
    resolutions = RING_RESOLUTIONS[ring_ids]
    x = (cells["col"].astype(np.float32) + 0.5) * resolutions
    y = (cells["row"].astype(np.float32) + 0.5) * resolutions
    return x, y


def _scatter(ax, cells: np.ndarray, values: np.ndarray, *, cmap, norm=None, title: str):
    x, y = _cell_centers(cells)
    sizes = np.clip(180.0 * cells["ring_id"].astype(np.float32) + 18.0, 18.0, 90.0)
    plot = ax.scatter(x, y, c=values, s=sizes, cmap=cmap, norm=norm, linewidths=0)
    ax.scatter([0.0], [0.0], marker="x", c="black", s=45, linewidths=1.5)
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.grid(alpha=0.15)
    return plot


def render_snapshot(snapshot_path: Path, output_path: Path) -> None:
    with np.load(snapshot_path, allow_pickle=False) as snapshot:
        required = {"cells", "ratings", "dynamic_mask", "fused_cells"}
        missing = required.difference(snapshot.files)
        if missing:
            raise ValueError(
                f"{snapshot_path} is missing required fields: {sorted(missing)}"
            )
        cells = snapshot["cells"]
        ratings = np.asarray(snapshot["ratings"], dtype=np.float32)
        dynamic_mask = np.asarray(snapshot["dynamic_mask"], dtype=bool)
        fused_cells = snapshot["fused_cells"]

    if len(cells) != len(ratings) or len(cells) != len(dynamic_mask):
        raise ValueError(
            f"Snapshot alignment error: cells={len(cells)}, "
            f"ratings={len(ratings)}, dynamic_mask={len(dynamic_mask)}"
        )

    semantic_cmap = ListedColormap(SEMANTIC_COLORS)
    semantic_norm = BoundaryNorm(np.arange(-0.5, 4.5, 1.0), 4)
    ring_cmap = ListedColormap(RING_COLORS)
    ring_norm = BoundaryNorm(np.arange(-0.5, 4.5, 1.0), 4)

    figure, axes = plt.subplots(2, 2, figsize=(15, 11), constrained_layout=True)
    semantic_plot = _scatter(
        axes[0, 0],
        cells,
        cells["semantic_label"],
        cmap=semantic_cmap,
        norm=semantic_norm,
        title="Current semantic grid",
    )
    figure.colorbar(
        semantic_plot,
        ax=axes[0, 0],
        ticks=np.arange(4),
        fraction=0.046,
        pad=0.04,
    ).ax.set_yticklabels(SEMANTIC_NAMES)

    elevation_plot = _scatter(
        axes[0, 1],
        cells,
        cells["z_mean"],
        cmap="terrain",
        title="Current elevation (mean z)",
    )
    figure.colorbar(elevation_plot, ax=axes[0, 1], label="z (m)", fraction=0.046, pad=0.04)

    rating_plot = _scatter(
        axes[1, 0],
        cells,
        ratings,
        cmap="RdYlGn",
        title="Traversability score",
    )
    figure.colorbar(rating_plot, ax=axes[1, 0], label="0-100", fraction=0.046, pad=0.04)

    ring_plot = _scatter(
        axes[1, 1],
        cells,
        cells["ring_id"],
        cmap=ring_cmap,
        norm=ring_norm,
        title="Foveated resolution rings and dynamic overlay",
    )
    figure.colorbar(
        ring_plot,
        ax=axes[1, 1],
        ticks=np.arange(4),
        fraction=0.046,
        pad=0.04,
    ).ax.set_yticklabels(("5 cm", "10 cm", "25 cm", "50 cm"))

    if np.any(dynamic_mask):
        x_dynamic, y_dynamic = _cell_centers(cells[dynamic_mask])
        axes[1, 1].scatter(
            x_dynamic,
            y_dynamic,
            facecolors="none",
            edgecolors="#111111",
            marker="o",
            s=80,
            linewidths=1.2,
            label="dynamic current cells",
        )
        axes[1, 1].legend(loc="upper right", fontsize="small")

    figure.suptitle(
        f"Foveated Semantic 2.5D LiDAR Map — {snapshot_path.stem} | "
        f"current cells: {len(cells):,} | persistent cells: {len(fused_cells):,}",
        fontsize=14,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def _parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--maps-root",
        type=Path,
        default=project_root / "SalsaNext-Fork" / "predictions" / "valid" / "maps",
        help="Root containing sequences/<sequence>/<frame>.npz.",
    )
    parser.add_argument("--sequence", default="08", help="SemanticKITTI sequence.")
    parser.add_argument(
        "--frame",
        default=None,
        help="Frame number (default: newest available snapshot).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="PNG path (default: <maps-root>/visualizations/<sequence>/<frame>.png).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    sequence = f"{int(args.sequence):02d}"
    sequence_maps = args.maps_root / "sequences" / sequence
    snapshots = sorted(sequence_maps.glob("*.npz"))
    if not snapshots:
        raise FileNotFoundError(
            f"No map snapshots found in {sequence_maps}. "
            "Run infer.py with --pipeline first."
        )

    if args.frame is None:
        snapshot_path = snapshots[-1]
    else:
        snapshot_path = _frame_path(args.maps_root, sequence, args.frame)
        if not snapshot_path.is_file():
            raise FileNotFoundError(f"Map snapshot not found: {snapshot_path}")

    output_path = args.output or (
        args.maps_root / "visualizations" / sequence / f"{snapshot_path.stem}.png"
    )
    render_snapshot(snapshot_path, output_path)
    print(f"Rendered {snapshot_path} -> {output_path}")


if __name__ == "__main__":
    main()
