# Fusion Optimization Execution Report — Rev. 4

Date: 2026-09-24  
Environment: NumPy 2.5.3, bundled Python 3.14.6, 271 Sequence-08 saved cell inputs, three repeats per arm and regime.

The raw 271-frame scan/pose dataset is not present in this checkout. These are controlled Fusion
benchmarks only; no end-to-end or FPS claim is made. Identity timing is comparative/correctness
evidence. Absolute Fusion timing in this report uses the composed non-identity regime.

## Implemented changes

| Item | Change | Exactness result |
|---|---|---|
| S5 | Replaced rebin `np.minimum.at`/`np.maximum.at` with a stable segment reduction. One grouping permutation is reused for separate `z_min` and `z_max` arrays. | Min/max only; all weighted sums remain `np.bincount`. Bit-exact in both regimes. |
| S7 | Added immutable module-level ring center offsets and reused the ring resolution table for rebin/prune calculations. | No arithmetic/ordering/constant semantic change. Bit-exact in both regimes. |

Previously retained S1, S2, S3a, S4, and S6 remain unchanged. S14, S8, S21, S9, S10, S18,
S11–S17, and all approximation paths remain out of scope.

## Correctness gates

- `test_fusion.py` and `test_dynamics.py`: **13 passed**; tests unmodified.
- Identity open-loop: **PASS**, 271/271, 0 diff.
- Identity closed-loop: **PASS**, 271/271, frame-271 divergence 0.
- Non-identity open-loop: **PASS**, 271/271, 0 diff.
- Non-identity closed-loop: **PASS**, 271/271, frame-271 divergence 0.
- `dynamic_mask`, `fused_cells`, and `rebinned_prior`: identical dtype, shape, values, and row/key order.
- Floating-point fields: bitwise identical.
- Two-step composed-transform smoke and full-sequence comparison: **PASS**.

## Dual-regime benchmark

Values are the mean of the three per-repeat statistics. Repeat-mean σ is shown separately.

### Regime A — identity transform

Identity is intentionally not representative of motion; it has trivial key stability and is used
for paired continuity with the prior report.

| Metric | Pre-S5/S7 baseline | S5/S7 candidate | Delta |
|---|---:|---:|---:|
| Fusion mean | 325.067 ms | 199.168 ms | -125.899 ms (-38.7%) |
| p50 | 324.466 ms | 201.615 ms | -122.851 ms |
| p95 | 399.095 ms | 246.820 ms | -152.275 ms |
| max | 583.967 ms | 281.460 ms | -302.507 ms |
| repeat-mean σ | 16.872 ms | 9.828 ms | — |

### Regime B — composed non-identity transform

Each frame uses a deterministic composition of two small rigid transforms. This is the regime used
for absolute Fusion timing and the S21 evidence base.

| Metric | Pre-S5/S7 baseline | S5/S7 candidate | Delta |
|---|---:|---:|---:|
| Fusion mean | 355.525 ms | 204.257 ms | -151.267 ms (-42.5%) |
| p50 | 350.853 ms | 201.030 ms | -149.823 ms |
| p95 | 469.640 ms | 261.451 ms | -208.188 ms |
| max | 607.663 ms | 309.257 ms | -298.406 ms |
| repeat-mean σ | 6.725 ms | 3.719 ms | — |
| paired baseline-minus-candidate mean | — | 151.267 ms | candidate faster on 99.88% of paired frames |

The regime-B candidate mean is above the plan’s approximately 80 ms practical pipelining ceiling.

## G0 profile — regime B candidate

Profile timings include instrumentation overhead and are attribution data, not replacement absolute
benchmark values.

| Helper phase | Mean | p50 | p95 | Share |
|---|---:|---:|---:|---:|
| Rebin | 136.091 ms | 134.537 ms | 168.304 ms | 65.3% |
| Output conversion | 31.440 ms | 31.233 ms | 41.132 ms | 15.1% |
| Assembly + prune | 25.153 ms | 24.781 ms | 32.691 ms | 12.1% |
| Matching | 6.630 ms | 6.457 ms | 9.041 ms | 3.2% |
| Dynamic detection | 6.687 ms | 6.436 ms | 9.710 ms | 3.2% |
| Measurement log-odds | 2.385 ms | 2.181 ms | 3.886 ms | 1.1% |

## Per-frame sizes — regime B

| Quantity | Min | Mean | Max |
|---|---:|---:|---:|
| N current cells | 38,258 | 51,379.31 | 61,521 |
| H stored cells | 0 | 344,998 | 390,010 |
| R rebinned unique keys | 0 | 342,254.55 | 387,320 |
| M matched current cells | 0 | 47,188.18 | 56,842 |
| Rebin collisions H−R | 0 | 2,743.45 | 3,294 |
| R/H | 0.9907 | 0.9921 | 0.9963 |

Stored-state trajectory stayed bounded at 0–390,010 cells in regime B. A peak-temporary sampler
was unavailable, so the temporary-memory portion of G3 is unverified.

## Key stability — regime B only

Identity stability is trivially 100% and is invalid evidence for S21. The non-identity results are:

| Ring | Mean stable fraction | p50 stable fraction | All-stable event rate | Stable-run p50 / p95 / max |
|---:|---:|---:|---:|---:|
| 0 | 0.0000 | 0.0000 | 0.0% | 116,418.5 / 117,557.95 / 118,006 |
| 1 | 0.0000 | 0.0000 | 0.0% | 140,606.5 / 151,196.5 / 152,646 |
| 2 | 0.0157 | 0.0000 | 0.0% | 49 / 148 / 82,756 |
| 3 | 0.6276 | 0.6166 | 0.0% | 38 / 114 / 22,928 |

The evidence signal is unfavorable for an all-or-nothing S21 fast path: no ring was all-stable on
any eligible frame. This is a go/no-go input for the program owner, not an S21 decision.

## Comparison with previous statistics

| Statistic | Previous | Rev. 4 result | Interpretation |
|---|---:|---:|---|
| Prior controlled candidate mean, identity | 231.075 ms | 199.168 ms | -13.8% directional improvement; harness/cache state differs, so not a causal isolation claim. |
| Historical Fusion mean | 248.778 ms | 204.257 ms regime B | Different workload boundary and missing raw poses; directional context only. |
| Historical full pipeline | 1,059.836 ms/frame | Not rerun | Raw scan/pose inputs absent. |
| Historical snapshot serialization | 580.945 ms/frame | Unchanged/out of scope | Still the largest recorded stage. |
| Historical Rating | 208.123 ms/frame | Unchanged/out of scope | No Rating code changed. |
| Historical model path | 231.375 ms/frame | Unchanged/out of scope | No inference code changed. |
| Historical combined estimate | 1,365.321 ms/frame | Not rerun | No end-to-end acceptance. |

## Gate ledger

| Gate | Status | Evidence |
|---|---|---|
| G0 profiling | PASS / memory partial | Regime-B attribution, N/H/R/M/collisions, stability; peak temporaries unavailable. |
| G1 exactness | PASS | Both regimes, open/closed loop, 0 diff, frame 271 = 0. |
| G2 runtime | PASS | Candidate faster than pre-S5/S7 baseline in both regimes. |
| G3 memory | PARTIAL | Stored state bounded; peak temporary memory unverified. |
| G4 S11 | N/A | S11 not implemented. |
| G5 S18 | NOT RUN | Pipelining deferred and regime-B Fusion is >80 ms. |
| G6 S21 | EVIDENCE ONLY | No implementation; no all-stable event observed. |
| G7 end-to-end | NOT RUN | Raw scan/pose dataset unavailable. |

## Decision

`FAIL: exact path requires further structural work`

The combined Tier-1 candidate is exact and materially faster, but regime-B Fusion remains above
80 ms. The per-helper profile and key-stability data are prepared for the Rev. 3 §3.4 program
decision; the decision itself belongs to the program owner. Serialization levers versus further
Fusion structural work remain open options. No end-to-end or FPS claim is made.

## Reproduction

```powershell
cd D:\Mainpro\Sihresources00
& .\SalsaNext-Fork\.venv\Scripts\python.exe fusion_optimization_benchmark.py --repeats 3 --output fusion_optimization_rev4_report.json
```

G0-only profile:

```powershell
& .\SalsaNext-Fork\.venv\Scripts\python.exe fusion_optimization_benchmark.py --profile-only --output fusion_optimization_g0_profile.json
```
