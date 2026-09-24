import os
import sys
import numpy as np
from pathlib import Path

# Set optimized kernels
os.environ["FOVMAP_RATING_IMPL"] = "compiled"
os.environ["FOVMAP_FUSION_IMPL"] = "cuda"

ROOT = Path("d:/Mainpro/Sihresources00")
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from fovmap.replay_pipeline import generate_prediction_sequence_in_memory

DATASET_ROOT = ROOT / "SalsaNext-Fork" / "dataset_test"
PREDICTION_ROOT = ROOT / "SalsaNext-Fork" / "predictions" / "uncertainty_valid"

def run_benchmark():
    print("Starting optimized pipeline benchmark...")
    generator = generate_prediction_sequence_in_memory(DATASET_ROOT, PREDICTION_ROOT, sequence="08")
    
    timings = {
        "io_and_prep": [],
        "grid_build": [],
        "fusion": [],
        "rating": [],
        "total": []
    }
    
    frames = []
    
    # Run through all frames
    for i, frame_data in enumerate(generator):
        t = frame_data["timing_ms"]
        frames.append({
            "frame_id": frame_data["frame_id"],
            "timing": t
        })
        for k in timings:
            timings[k].append(t[k])
            
        if (i + 1) % 50 == 0:
            print(f"Processed {i + 1} frames...")
            
    print("Benchmark complete. Generating report...")
    
    # Calculate statistics
    stats = {}
    for k in timings:
        arr = np.array(timings[k])
        stats[k] = {
            "mean": np.mean(arr),
            "p50": np.median(arr),
            "p95": np.percentile(arr, 95),
            "max": np.max(arr),
            "min": np.min(arr)
        }
        
    # Write report
    report_path = r"C:\Users\Yaksha's Laptop\.gemini\antigravity-ide\brain\7090dc49-d729-4d72-9a4d-cc1edc045734\optimized_pipeline_latency_report.md"
    
    with open(report_path, "w") as f:
        f.write("# Fully Optimized Pipeline End-to-End Latency Report\n\n")
        f.write("This report details the frame-by-frame latency of the entire data processing pipeline (I/O, Grid Build, Fusion, and Rating) using the **hyper-optimized C-compiled Rating kernel** and **PyTorch CUDA Fusion kernel**.\n\n")
        f.write("Disk writing to `.npz` files has been completely eliminated as this data is now held directly in RAM.\n\n")
        
        f.write("## Aggregate Statistics (over 271 frames)\n\n")
        f.write("| Stage | Mean (ms) | P50 (ms) | P95 (ms) | Max (ms) |\n")
        f.write("|---|---|---|---|---|\n")
        for k in ["io_and_prep", "grid_build", "fusion", "rating", "total"]:
            s = stats[k]
            f.write(f"| {k} | {s['mean']:.3f} | {s['p50']:.3f} | {s['p95']:.3f} | {s['max']:.3f} |\n")
            
        f.write("\n## Per-Frame Breakdown\n\n")
        f.write("| Frame | Total (ms) | I/O (ms) | Grid (ms) | Fusion (ms) | Rating (ms) |\n")
        f.write("|---|---|---|---|---|---|\n")
        for frame in frames:
            t = frame["timing"]
            f.write(f"| {frame['frame_id']} | **{t['total']:.2f}** | {t['io_and_prep']:.2f} | {t['grid_build']:.2f} | {t['fusion']:.2f} | {t['rating']:.2f} |\n")
            
    print(f"Report written to {report_path}")

if __name__ == "__main__":
    run_benchmark()
