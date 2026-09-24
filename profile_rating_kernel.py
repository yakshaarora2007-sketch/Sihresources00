"""Function-level profile of Rating paths on ten real sequence-08 frames."""

from __future__ import annotations

import cProfile
import os
import pstats
from io import StringIO

from benchmark_rating_kernel import MatplotlibGridVisualizer, resolve_dataset, resolve_predictions
from fovmap.rating import rate_cells


def main():
    data_dir = resolve_dataset(None, "08")
    pred_dir = resolve_predictions(None, data_dir, "08")
    visualizer = MatplotlibGridVisualizer(data_dir, pred_dir=pred_dir)
    frames = [visualizer.load_frame_data(i)["cells"] for i in range(10)]
    for mode in ("baseline", "fast", "compiled"):
        os.environ["FOVMAP_RATING_IMPL"] = mode
        profiler = cProfile.Profile()
        profiler.enable()
        for cells in frames:
            rate_cells(cells)
        profiler.disable()
        output = StringIO()
        pstats.Stats(profiler, stream=output).sort_stats("cumtime").print_stats(15)
        print(f"\n=== {mode} ===\n{output.getvalue()}")


if __name__ == "__main__":
    main()
