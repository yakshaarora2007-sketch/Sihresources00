import ctypes
import threading
import time
from ctypes import wintypes
from pathlib import Path

from visualize_grid_matplotlib import MatplotlibGridVisualizer, resolve_dataset, resolve_predictions


class ProcessMemoryCounters(ctypes.Structure):
    _fields_ = [
        ("cb", wintypes.DWORD),
        ("PageFaultCount", wintypes.DWORD),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    ]


get_memory = ctypes.windll.psapi.GetProcessMemoryInfo
get_memory.argtypes = [
    wintypes.HANDLE,
    ctypes.POINTER(ProcessMemoryCounters),
    wintypes.DWORD,
]
get_memory.restype = wintypes.BOOL


class MemoryMonitor:
    def __init__(self):
        self.handle = ctypes.windll.kernel32.GetCurrentProcess()
        self.peak_working = 0
        self.peak_pagefile = 0
        self.stop = False
        self.thread = threading.Thread(target=self._sample)

    def _sample(self):
        while not self.stop:
            counters = ProcessMemoryCounters()
            counters.cb = ctypes.sizeof(counters)
            if get_memory(self.handle, ctypes.byref(counters), counters.cb):
                self.peak_working = max(self.peak_working, counters.WorkingSetSize)
                self.peak_pagefile = max(self.peak_pagefile, counters.PagefileUsage)
            time.sleep(0.002)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop = True
        self.thread.join()


def main():
    sequence = "08"
    output_path = Path(__file__).with_name("dataset08_frame_timings.txt")
    data_dir = resolve_dataset(None, sequence)
    pred_dir = resolve_predictions(None, data_dir, sequence)
    visualizer = MatplotlibGridVisualizer(data_dir, pred_dir=pred_dir)

    points = []
    cells = []
    frame_times = []
    with MemoryMonitor() as memory:
        run_start = time.perf_counter()
        for frame_index in range(visualizer.num_frames):
            frame_start = time.perf_counter()
            frame = visualizer.load_frame_data(frame_index)
            visualizer.render_bev_u8(frame, mode=1)
            visualizer.render_elevation_u8(frame)
            frame_times.append((time.perf_counter() - frame_start) * 1000.0)
            points.append(frame["num_pts"])
            cells.append(frame["num_cells"])
        total_seconds = time.perf_counter() - run_start

    frame_count = len(frame_times)
    average_points = sum(points) / frame_count
    average_cells = sum(cells) / frame_count
    average_reduction = 100.0 * (1.0 - sum(cells) / sum(points))
    reductions = [100.0 * (1.0 - cell / point) for point, cell in zip(points, cells)]

    lines = [
        "Dataset 08 full benchmark",
        "==========================",
        f"Frames: {frame_count}",
        f"Total time: {total_seconds:.3f} s",
        f"Average processing: {total_seconds / frame_count * 1000.0:.3f} ms/frame",
        f"Measured processing rate: {frame_count / total_seconds:.2f} FPS",
        f"Frame time range: {min(frame_times):.3f}-{max(frame_times):.3f} ms",
        f"Total points: {sum(points):,}",
        f"Points/frame: {min(points):,}-{max(points):,}, average {average_points:,.1f}",
        f"Total cells: {sum(cells):,}",
        f"Cells/frame: {min(cells):,}-{max(cells):,}, average {average_cells:,.1f}",
        f"Average reduction: {average_reduction:.2f}%",
        f"Reduction range: {min(reductions):.2f}-{max(reductions):.2f}%",
        f"Peak working memory: {memory.peak_working / 1048576.0:,.2f} MB",
        f"Peak private/pagefile memory: {memory.peak_pagefile / 1048576.0:,.2f} MB",
        f"Predictions loaded: {len(visualizer.pred_files)}/{frame_count}",
        "",
        "Per-frame timings",
        "-----------------",
        "frame,time_ms,points,cells,reduction_percent",
    ]
    lines.extend(
        f"{index:03d},{elapsed:.3f},{point},{cell},{reduction:.2f}"
        for index, (elapsed, point, cell, reduction) in enumerate(
            zip(frame_times, points, cells, reductions)
        )
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {frame_count} per-frame records to {output_path}")
    print("\n".join(lines[:17]))


if __name__ == "__main__":
    main()
