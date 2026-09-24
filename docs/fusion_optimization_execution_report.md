# Fusion Optimization Execution Report

Date: 2026-09-24  
Scope: Sequence 08, 271 saved cell inputs, NumPy 2.5.3, controlled identity `T_rel` benchmark.

The checkout contains the 271 saved prediction/cell snapshots but not the raw 271-frame scan and
pose dataset used by the historical full-pipeline run. The controlled benchmark therefore gates
the old and candidate Fusion call chains on the same saved inputs with an identity transform. It
must not be read as a replacement for the historical end-to-end timing.

## Implemented exact changes

| Strategy | Implementation | Result |
|---|---|---|
| S1 | Single-pass rebin computes the existing grouped fields and log-odds carry from the same `np.unique` inverse indices. | Retained; bit-exact in both harness modes. |
| S2 | One current/prior association is passed to dynamic detection and fusion. | Retained; bit-exact. |
| S3a | Sorted current and rebinned keys use direct `searchsorted` without re-sorting. | Retained; bit-exact. |
| S4 | Unmatched prior rows are derived from the shared match for the duplicate-free rebinned map. | Retained; bit-exact. |
| S6 | Fusion output is assembled into a preallocated structured buffer. | Retained; bit-exact. |

S5 (`reduceat` min/max) and S7 constant precomputation were not claimed as retained optimizations.
No approximation, reduced prune radius, S11, threading, Numba, GPU, or pipelining was introduced.

## Correctness gates

- Existing `test_fusion.py`: **9 passed**.
- Open-loop comparison across all 271 frames: **PASS, 0 diff**.
- Closed-loop comparison across all 271 frames: **PASS, 0 diff**.
- Frame-271 divergence: **0**.
- `dynamic_mask`, `fused_cells`, and `rebinned_prior`: identical dtype, shape, values, and row/key order.
- Floating-point fields: bitwise identical in the harness comparison.
- Additional two-step non-identity transform check: **0 diff**.

## Controlled Fusion benchmark

Three timing repeats were run for each implementation. The table reports the mean of the three
per-repeat statistics; repeat mean standard deviation is included for context.

| Metric | Baseline call chain | Candidate | Delta |
|---|---:|---:|---:|
| Fusion mean | 308.871 ms | 231.075 ms | -77.796 ms (-25.2%) |
| Fusion p50 | 307.209 ms | 232.553 ms | -74.656 ms (-24.3%) |
| Fusion p95 | 378.023 ms | 278.749 ms | -99.274 ms (-26.3%) |
| Fusion max | 472.478 ms | 360.421 ms | -112.057 ms (-23.7%) |
| Repeat-mean σ | 29.211 ms | 31.653 ms | — |
| Stored cells: min / max | 56,795 / 394,422 | 56,795 / 394,422 | unchanged |

The machine-level memory sampler was unavailable in the bundled environment, so no numeric peak
memory claim is made. Stored-state trajectory was unchanged.

## Comparison with previous project statistics

| Statistic | Previous measurement | Fusion execution result | Interpretation |
|---|---:|---:|---|
| Historical Fusion mean, `frontend_simulation_latency_report.txt` | 248.778 ms/frame | 231.075 ms/frame candidate controlled mean | Different workload boundary/transform set; directional only, not apples-to-apples. |
| Historical Fusion min / max | 29.576 / 481.366 ms |  — / 360.421 ms candidate controlled max | Candidate max is lower in the controlled run; not a full-pipeline replacement. |
| Historical full pipeline | 1,059.836 ms/frame | Not rerun; raw scan/pose inputs absent | No end-to-end FPS claim. |
| Historical rating | 208.123 ms/frame | Unchanged/out of scope | Fusion optimization does not alter Rating. |
| Historical snapshot serialization | 580.945 ms/frame | Unchanged/out of scope | Still dominates the recorded pipeline. |
| Historical model path | 231.375 ms/frame | Unchanged/out of scope | Still required for live 10 FPS planning. |
| Historical combined estimate | 1,365.321 ms/frame | Not rerun | No end-to-end target acceptance. |
| React playback | 10.11 FPS, 9.51 ms/frame | Unchanged/out of scope | Reads saved snapshots and does not execute Fusion. |

## Decision

`FAIL: exact path requires further structural work`

The retained changes are exact and materially faster, but the measured candidate Fusion mean is
still above the plan's approximately 80 ms pipelining ceiling. The result is not sufficient to
claim a 10 FPS live end-to-end system; the next plan-directed work is Phase 3 structural profiling
and experiments (S14/S8, with S21 only if key-stability measurements justify it), followed by a
full raw-data end-to-end rerun when the missing scan/pose dataset is available.

Reproduction:

```powershell
cd D:\Mainpro\Sihresources00
& .\SalsaNext-Fork\.venv\Scripts\python.exe fusion_optimization_benchmark.py --repeats 3 --benchmark-only
```
