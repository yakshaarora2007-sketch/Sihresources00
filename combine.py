import os
from pathlib import Path

in_grid = Path('d:/Mainpro/docs/grid_comparison (1).md')
in_lat = Path("C:/Users/Yaksha's Laptop/.gemini/antigravity-ide/brain/7090dc49-d729-4d72-9a4d-cc1edc045734/optimized_pipeline_latency_report.md")
out = Path("C:/Users/Yaksha's Laptop/.gemini/antigravity-ide/brain/7090dc49-d729-4d72-9a4d-cc1edc045734/comprehensive_performance_report.md")

grid_data = in_grid.read_text(encoding='utf-8')
lat_data = in_lat.read_text(encoding='utf-8')

parts = lat_data.split('## Per-Frame Breakdown')
agg = parts[0].replace('# Fully Optimized Pipeline End-to-End Latency Report\n\n', '')

combined = f"# Comprehensive Grid Comparison and Latency Report\n\n{grid_data}\n\n---\n## Pipeline Latency Addendum\n\n{agg}\n## Per-Frame Breakdown{parts[1]}"

out.write_text(combined, encoding='utf-8')
print("Successfully combined.")
