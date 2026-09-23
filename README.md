# Adaptive 2.5D LiDAR Simulation — React Frontend & API

A high-performance **React + TypeScript + Canvas** frontend and lightweight **Starlette/Uvicorn ASGI API** replacing the legacy Streamlit/Matplotlib dashboard for adaptive variable-resolution LiDAR occupancy mapping.

Achieves locked **10 FPS client-side playback** with a **9.51 ms average frame latency** (under a 100 ms frame budget) without triggering Python reruns.

---

## Architecture Overview

```text
Precomputed Sequence 08 .npz Snapshots (271 frames)
       │
       ▼
Lightweight Backend API (Starlette / Uvicorn on :8000)
       │  GET /api/metadata
       │  GET /api/frames/:id/binary (Packed typed arrays)
       │  GET /api/frames/:id (Compact JSON fallback)
       ▼
React Frontend (Vite on :5173)
 ├── FrameProvider Abstraction (Decoupled REST / Binary / WebSocket)
 ├── Bounded Sliding-Window Cache (15 frames, asynchronous prefetch)
 ├── Dedicated Playback Controller (requestAnimationFrame timing loop @ 10 FPS)
 ├── High-Performance Canvas 2D Renderer (Batched color buckets, ~6 ms)
 └── Performance Instrumentation HUD (Live latency & FPS telemetry)
```

---

## Quickstart

### Prerequisites
- Python 3.10+ with existing virtual environment (`.venv`)
- Node.js 18+ (tested on Node 24 and npm 11)

### 1. Start Backend Server
From the repository root:
```powershell
.\.venv\Scripts\python.exe backend/server.py
```
The backend starts on `http://127.0.0.1:8000`.

Available endpoints:
- `GET /api/metadata` — Sequence name, frame count, target FPS, resolutions, semantic classes, ring boundaries.
- `GET /api/frames` — Summary list of all 271 frames.
- `GET /api/frames/{frame_id}` — Frame data in JSON format.
- `GET /api/frames/{frame_id}/binary` — High-speed packed binary buffer (`application/octet-stream`, ~851 KB/frame).
- `GET /api/health` — Health check.
- `WS  /ws/live` — Live frame WebSocket subscription endpoint.

### 2. Start React Frontend
In a new terminal:
```powershell
cd frontend
npm install
npm run dev
```
Open your browser at `http://localhost:5173`.

---

## Interactive Controls

| Control | Action |
|---|---|
| **`[Space]`** | Play / Pause (10 FPS default) |
| **`[D]` / `[→]`** | Step forward 1 frame |
| **`[A]` / `[←]`** | Step backward 1 frame |
| **`[R]`** | Reset to Frame 0 |
| **`[S]`** | Cycle Color Modes (Semantics → Elevation → Traversability → Rings → Dynamic) |
| **Scrubber Slider** | Drag to seek directly across all 271 frames |
| **Speed Buttons** | Switch playback speed (5, 10, 15, 20, 30 FPS) |
| **Mouse Drag** | Pan viewport |
| **Mouse Wheel** | Zoom in / out centered on mouse cursor |
| **Double Click / `[⌖]`** | Reset view to center on sensor origin |
| **Mouse Hover** | Inspect cell coordinate, ring, elevation, class, rating, and state |

---

## Performance Summary

Tested on Sequence 08 (271 frames, ~56,000 cells per frame):

- **Playback Framerate:** **10.11 FPS** (Target: 10.0 FPS)
- **Data Fetch / Cache-Hit Latency:** **3.34 ms** (Target: < 20.0 ms)
- **Data Parse Latency (Binary):** **0.17 ms** (Target: < 5.0 ms)
- **Canvas 2D Render Latency:** **5.99 ms** (Target: < 16–20 ms)
- **Total End-to-End Frame Latency:** **9.51 ms** (Budget: 100.0 ms, **9.5% utilized**)
- **Dropped Frames:** **0 frames**
- **Cache Hit Rate:** **96.0%** via bounded 15-frame sliding window

For complete architecture, invariants, and step-by-step developer recipes, see [ARCHITECTURE_AND_DEVELOPER_GUIDE.md](ARCHITECTURE_AND_DEVELOPER_GUIDE.md).
For before/after benchmark comparison against the legacy system, see [LATENCY_MIGRATION_REPORT.md](LATENCY_MIGRATION_REPORT.md).

---

## Project Structure

```text
├── backend/
│   └── server.py                  # Lightweight Starlette ASGI backend
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── LidarCanvas.tsx    # High-speed Canvas 2D renderer
│   │   │   ├── PlaybackControls.tsx # Scrubbing and playback bar
│   │   │   ├── TelemetryHUD.tsx   # Real-time Performance HUD
│   │   │   ├── ViewModeSelector.tsx # Color mode toggles
│   │   │   └── CellInspector.tsx  # Hover hit-test detail card
│   │   ├── hooks/
│   │   │   ├── usePlaybackController.ts # 10 FPS dedicated loop
│   │   │   └── usePerformanceMonitor.ts # Performance API telemetry
│   │   ├── services/
│   │   │   └── FrameProvider.ts   # Decoupled REST/Binary/WS provider & cache
│   │   ├── types/
│   │   │   └── simulation.ts      # TypeScript interfaces
│   │   ├── App.tsx                # Main application layout
│   │   └── main.tsx               # React entry point
│   ├── benchmark_browser.js       # Headless Chrome benchmark runner
│   ├── vite.config.ts             # Vite build & proxy config
│   └── package.json
├── frontend_simulation_latency.py # Legacy benchmark script
├── frontend_simulation_latency_report.txt # Legacy baseline output
├── LATENCY_MIGRATION_REPORT.md    # Before/After comparison report
└── README.md
```
