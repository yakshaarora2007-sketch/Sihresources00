# LiDAR Simulation Frontend Migration: Before/After Latency & Performance Report

## 1. Executive Summary

The legacy Streamlit/Matplotlib LiDAR simulation frontend has been successfully migrated to a high-performance **React + TypeScript + HTML5 Canvas** web application served by a lightweight **Starlette/Uvicorn ASGI API**. 

The migration achieved:
- **10.11 FPS** stable client-side playback rate (up from **1.60 FPS** in Streamlit, a **+531% framerate gain**).
- **9.51 ms** average end-to-end frame processing time, consuming only **9.5%** of the 100 ms frame budget.
- **Zero full-script Python reruns**: Playback is managed by a client-side `requestAnimationFrame` timing loop decoupled from React state reconciliation.
- **0 dropped frames** over continuous 10-second observation.
- **3.34 ms** average data fetch latency and **0.17 ms** binary decode time via packed contiguous typed arrays (`application/octet-stream`).
- **5.99 ms** Canvas 2D render latency for ~56,000 adaptive cells per frame via color-bucketed batch rendering.
- **100% preservation** of existing backend algorithms (`fovmap.dynamics_fusion`, `fovmap.grid_engine`, `fovmap.rating`, SalsaNext inference, and `.npz` snapshots).

---

## 2. Architecture Comparison

### Legacy Architecture (Streamlit + Matplotlib)
```text
Raw LiDAR / Precomputed .npz
       │
       ▼
Streamlit Python Server Process
       ├── Full Script Rerun on Frame Change (via st.session_state)
       ├── In-process Matplotlib Rasterization (~126.8 ms)
       ├── PNG / JPEG image encoding & disk/cache write
       └── Transfer rendered image to browser (/media/*.jpg)
              │
              ▼
Browser Viewport (1.60 FPS, high stutter, 404 image errors)
```
- **Primary Bottlenecks**:
  1. Synchronous Matplotlib figure generation on every frame tick (`~126.8 ms/frame`, up to `357.7 ms`).
  2. Streamlit's reactive execution model rerun the entire Python script on every frame navigation.
  3. Transferring large rasterized images over HTTP led to 60% missed frame opportunities and repeated 404 media errors.

### Modern Architecture (React + Canvas + ASGI API)
```text
Saved Sequence 08 .npz Snapshots (271 frames)
       │
       ▼
Lightweight ASGI Backend API (Starlette + Uvicorn on :8000)
       ├── GET /api/metadata (Sequence bounds, ring resolutions, semantic classes)
       ├── GET /api/frames/:id/binary (Packed typed arrays: 851 KB octet-stream)
       └── GET /api/frames/:id (Compact JSON fallback)
              │
              ▼
React Frontend (Vite on :5173)
 ├── Transport-Independent FrameProvider Abstraction
 ├── Bounded Sliding-Window Cache (15 frames, ~20 MB RAM footprint)
 ├── Asynchronous Proactive Prefetcher (Prefetches current + 1..3 frames)
 ├── Dedicated Playback Controller (requestAnimationFrame timing loop @ 10 FPS)
 ├── High-Performance Canvas 2D Renderer (Batched color buckets, ~5.99 ms/frame)
 └── Live Performance API Instrumentation (Telemetry HUD overlay)
```

---

## 3. Benchmark Measurements & Results

Measurements were collected on the exact same dataset sequence (`Sequence 08`, 271 frames, ~56,000 cells per frame).

### Category A: Python / Backend Baseline (`frontend_simulation_latency.py`)
*Measured across all 271 frames of Sequence 08 on Python 3.12:*

| Pipeline / Backend Stage | Average Latency | Min Latency | Max Latency | Total Duration (271 Frames) |
|---|---|---|---|---|
| **Scan Loading (`.bin`)** | 22.11 ms | 1.58 ms | 222.48 ms | 5.99 s |
| **Prediction Loading (`.label`)** | 1.86 ms | 0.99 ms | 4.68 ms | 0.50 s |
| **Uncertainty Loading** | 1.67 ms | 0.94 ms | 4.03 ms | 0.45 s |
| **Preprocessing & Ring Binning** | 2.87 ms | 1.43 ms | 9.37 ms | 0.78 s |
| **Grid Construction (Cell Schema)** | 33.42 ms | 16.35 ms | 82.48 ms | 9.06 s |
| **Bayesian Fusion** | 426.33 ms | 37.06 ms | 891.43 ms | 115.54 s |
| **Cell Rating (Traversability)** | 397.54 ms | 196.79 ms | 760.74 ms | 107.73 s |
| **Snapshot Serialization (`.npz`)** | 1053.06 ms | 193.50 ms | 1949.71 ms | 285.38 s |
| **Raw Pipeline Total** | **1940.80 ms** | **677.41 ms** | **2983.82 ms** | **525.96 s** |

*Legacy Frontend Backend Timings (`frontend_simulation_latency.py`):*
- **Snapshot Loading (`.npz`)**: 97.14 ms (min 37.90 ms, max 210.03 ms)
- **Dataframe Preparation (`flatten_grid`)**: 11.75 ms (min 7.43 ms, max 36.89 ms)
- **Matplotlib Overview Render**: 126.76 ms (min 90.81 ms, max 357.68 ms)
- **Legacy Frontend Backend Total**: **235.64 ms / frame** (min 168.21 ms, max 480.84 ms)

---

### Category B: Historical Streamlit Browser Observation (`frontend_simulation_latency.py`)
*Recorded during continuous browser playback of the Streamlit dashboard:*
- **Target Playback Rate**: 4.0 FPS (250 ms period)
- **Observed Actual Rate**: **1.60 FPS** (625 ms period)
- **Observation Window**: 10.0 s
- **Expected Frame Opportunities**: 40 frames
- **Displayed Frame Opportunities Observed**: 16 frames
- **Observed Display Shortfall**: **24 frames (60.0% loss)**
- **Observed Average Lag**: 0.124 frames (max 1 frame)
- **Failures / Anomalies**: Streamlit accumulated latency, stale frames rendered, repeated `/media/*.jpg` 404 errors.

---

### Category C: React Frontend Live Browser Measurements
*Recorded live via Chrome Performance API instrumentation over 10.0 s continuous playback:*

| Performance Metric | Measured Value | Budget / Target | Status |
|---|---|---|---|
| **Target Playback Rate** | 10.0 FPS | 10.0 FPS | **LOCKED** |
| **Actual Measured FPS** | **10.11 FPS** (avg) | 10.0 FPS | **PASSED** |
| **Data Fetch / Cache-Hit Time** | **3.34 ms** (avg) | < 20.0 ms | **PASSED** |
| **Data Decode / Parse Time** | **0.170 ms** (avg) | < 5.0 ms | **PASSED** |
| **Frame State Update Time** | **0.026 ms** (avg) | < 2.0 ms | **PASSED** |
| **Canvas 2D Render Time** | **5.99 ms** (avg) | < 16–20 ms | **PASSED** |
| **Total End-to-End Frame Latency** | **9.51 ms** (avg) | < 100.0 ms | **PASSED** |
| **100 ms Budget Utilization** | **9.5%** | < 100% | **PASSED** |
| **Dropped Frames** | **0 frames** | 0 frames | **PASSED** |
| **Cache Hit Rate** | **96.0%** (97 hits / 4 misses) | > 80% | **PASSED** |
| **Cache Memory Footprint** | 15 frames (~20 MB) | < 100 MB | **PASSED** |

---

## 4. Side-by-Side Comparison

| Metric / Feature | Legacy Streamlit + Matplotlib | React + Canvas + Starlette API | Improvement |
|---|---|---|---|
| **Playback Framerate** | 1.60 FPS (target 4 FPS) | **10.11 FPS** (target 10 FPS) | **+531% faster** |
| **Frame Delivery Budget** | 235.6 ms backend + browser lag | **9.51 ms** end-to-end | **24.8x faster** |
| **Render Latency** | 126.8 ms (Matplotlib) | **5.99 ms** (Canvas 2D) | **21.2x faster** |
| **Data Loading / Parsing** | 97.1 ms (.npz unpickle/parse) | **0.17 ms** (Binary typed array) | **571x faster** |
| **Frame Loss / Shortfall** | 60.0% missed frames | **0.0% dropped frames** | **Zero frame loss** |
| **Python Rerun per Frame** | Yes (full Streamlit script rerun) | **No** (pure client-side timing) | **Eliminated** |
| **DOM Overhead** | High (Plotly SVG / rerun DOM) | Zero (single HTML5 Canvas) | **Minimal** |
| **Interactive Controls** | Slow slider re-execution | Smooth pan, zoom, hover hit-test | **Instantaneous** |
| **Transport Protocol** | HTTP HTML/media polling | Bounded REST + Binary Buffer / WS | **Decoupled** |

---

## 5. Technical Highlights & How 10 FPS is Maintained

1. **Client-Side `requestAnimationFrame` Timing Loop**:
   - Frame pacing is controlled purely in JavaScript using a high-precision delta-time accumulator locked to 10 FPS (100 ms interval).
   - Python is never invoked during playback ticks; frames are served asynchronously from memory.

2. **Zero-Copy Packed Binary Transport**:
   - The backend exposes `GET /api/frames/:id/binary`, packing a 32-byte header and 7 contiguous typed arrays into a single `application/octet-stream` of 851 KB.
   - All arrays (`z_mean`, `ratings`, `col`, `row`, `ring_id`, `semantic_label`, `dynamic_mask`) are naturally aligned (4-byte and 2-byte boundaries).
   - In the browser, `new Float32Array(buffer, offset, count)` instantiates views in `<0.2 ms` without copying or JSON string parsing.

3. **Color-Bucketed Canvas 2D Batching**:
   - Instead of switching `ctx.fillStyle` 56,000 times per frame, cells are pre-binned into color buckets (e.g. 4 for semantics, 4 for rings, 32 for ratings/elevation).
   - Each bucket sets `fillStyle` once and renders all rectangles in a tight loop, achieving **5.99 ms** render time.

4. **Bounded Sliding Window Cache**:
   - A sliding window of 15 frames is maintained in memory.
   - When moving forward, frames `current + 1`, `current + 2`, and `current + 3` are prefetched asynchronously in the background.
   - Cache hit rate reached **96%**, reducing network wait time to `0.05 ms` per frame during continuous playback.

---

## 6. Remaining Bottlenecks & Future Scalability

- **Pipeline Precomputation**: The raw pipeline (Bayesian fusion + cell rating + `.npz` compression) takes `~1.94 s/frame` in Python. While the React frontend is completely decoupled from this pipeline by reading precomputed snapshots, real-time live fusion would require optimizing the Python Bayesian loop or compiling it with Cython/Numba.
- **WebSocket Streaming**: A `/ws/live` endpoint skeleton is included in the backend and supported by `FrameProvider`, ready for streaming live sensor feeds when live inference is deployed.
