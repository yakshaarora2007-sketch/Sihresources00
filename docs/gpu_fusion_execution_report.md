# CUDA Fusion execution report

## Scope

This report evaluates the experimental S16 PyTorch CUDA Fusion path against the
current exact NumPy Fusion implementation. The production NumPy path was not
replaced. Both paths were run with the same saved 271-frame inputs and the same
identity/nonidentity transform regimes.

## Environment

- Interpreter: `Sihresources00\\SalsaNext-Fork\\.venv\\Scripts\\python.exe`
- Python: 3.14.6
- PyTorch: 2.14.0+cu130
- CUDA build: 13.0
- GPU: NVIDIA GeForce RTX 5060 Laptop GPU
- NumPy: 2.5.3
- Frames: 271 per regime
- Repeats: 3

## Results

| Regime | CPU NumPy mean | GPU CUDA mean | GPU vs CPU | CPU-minus-GPU positive frames | Output comparison |
|---|---:|---:|---:|---:|---|
| Identity | 248.109 ms | 97.422 ms | 60.7% faster | 99.75% | Exact |
| Nonidentity | 218.520 ms | 86.884 ms | 60.2% faster | 100.0% | Exact |

Exactness was checked on the first repeat for fused cells, dynamic masks, and
rebinned priors. The checked arrays had equal dtypes and shapes, zero
mismatches, and zero measured floating-point deltas; the 5-frame smoke run also
matched exactly in both regimes.

## Comparison with the previous Rev4 candidate

The earlier Rev4 NumPy candidate means were 199.168 ms for identity and
204.257 ms for nonidentity. The CUDA path therefore improves on that candidate
by 51.1% in identity and 57.5% in nonidentity. Relative to the earlier Rev4
baseline means of 325.067 ms and 355.525 ms, the CUDA path improves by 70.0%
and 75.6%, respectively.

| Regime | Rev4 baseline | Rev4 NumPy candidate | S16 CUDA | CUDA vs Rev4 candidate |
|---|---:|---:|---:|---:|
| Identity | 325.067 ms | 199.168 ms | 97.422 ms | 51.1% faster |
| Nonidentity | 355.525 ms | 204.257 ms | 86.884 ms | 57.5% faster |

## Decision

The prototype meets the measured correctness and latency gates for this saved
cell-input benchmark. It remains isolated in `gpu_dynamics_fusion.py`; the
production implementation is unchanged pending an application-level integration
test using the live raw-scan/pose pipeline and a longer stability run.

Raw benchmark data: `gpu_fusion_benchmark.json`.
