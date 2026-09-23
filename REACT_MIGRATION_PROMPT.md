# Task: Migrate the Streamlit Simulation Frontend to React

Replace only the frontend of the existing LiDAR simulation with React. Do not rewrite the LiDAR processing pipeline, grid construction, Fusion, Rating, SalsaNext inference, uncertainty computation, `.npz` snapshot generation, `process_frame()`, `rate_cells()`, or any backend algorithms.

The existing data flow is:

```text
LiDAR / precomputed data -> .npz snapshot -> backend/API -> React frontend
```

Preserve the existing `.npz` outputs and field meanings. Snapshots contain `cells`, `ratings`, `dynamic_mask`, `rebinned_prior`, `fused_cells`, and `point_uncertainty`.

## Goal

Build a React frontend with stable 10 FPS playback and a 100 ms frame budget. React must control playback client-side and must not trigger Python or Streamlit reruns per frame.

## Technology

Use React, TypeScript, Vite, Canvas or WebGL, CSS, and clean components. Prefer Canvas/WebGL over SVG for large cell counts. Do not use Matplotlib, Streamlit charts, PNG frame generation, screenshots, or thousands of DOM elements.

## Visualization

Display the adaptive grid, cell positions, elevation, semantic class, rating, dynamic/static state, current frame, and uncertainty when available.

Use the existing four semantic classes:

```text
TERRAIN, DRIVABLE, STATIC, OBJECT
```

Use the existing ring resolutions:

```text
ring 0: 0.05 m
ring 1: 0.10 m
ring 2: 0.25 m
ring 3: 0.50 m
```

Preserve the existing coordinate calculation: resolution comes from `ring_id`, then `x = (col + 0.5) * resolution` and `y = (row + 0.5) * resolution`.

Support pan, zoom, reset view, semantic coloring, rating coloring, dynamic highlighting, and frame updates without recreating the rendering surface.

## Playback controller

Implement play, pause, restart, previous frame, next frame, current frame, total frame count, and optional speed control. Use a dedicated timing loop. Do not rerender the entire React tree every 100 ms. Update only the visualization state that changes.

## API and data source

Create a lightweight backend/API without changing snapshot generation. Possible endpoints:

```text
GET /api/metadata
GET /api/frames
GET /api/frames/:frame_id
```

Metadata should include sequence, frame count, target FPS, semantic classes, and resolutions. Frame responses should contain only fields needed by the renderer. Evaluate typed arrays or binary transport if JSON is too large.

Create a transport-independent `FrameProvider` abstraction with `getFrame(id)`, `prefetchFrame(id)`, `getMetadata()`, and future live-frame subscription support. The renderer must not care whether frames come from REST, cache, or WebSocket.

## Cache and prefetch

Use a bounded cache, for example Previous, Current, Next, Next + 1, and Next + 2. Prefetch upcoming frames asynchronously. Do not preload all 271 frames unless measurements show that memory use is safe.

## Browser performance instrumentation

Measure with the browser Performance API or equivalent:

- Data fetch/cache-hit time
- Data decode/parse time
- Frame update time
- Canvas/WebGL render time
- End-to-end frame time
- Actual FPS and average FPS
- Maximum frame time
- Dropped frames

Target data update below 20 ms, visualization below 16–20 ms, minimal UI overhead, and total frame time below 100 ms. Display metrics in development mode.

## Required use of simulated latency benchmark

The repository contains `frontend_simulation_latency.py`. Treat it as a legacy baseline benchmark, not as part of the React runtime.

Inspect and run it before changing the frontend when its snapshots and dependencies are available. Use its Python-side timings for scan loading, prediction loading, uncertainty loading, preprocessing, grid construction, Fusion, Rating, snapshot serialization, snapshot loading, data preparation, and legacy Matplotlib rendering.

Rules:

1. Do not import or execute `frontend_simulation_latency.py` from React.
2. React must not depend on Matplotlib, Streamlit, or `visualize_grid_matplotlib.py`.
3. If its paths are stale, make only the smallest benchmark-only path/import fix needed to run it.
4. Do not modify Fusion, Rating, SalsaNext, uncertainty generation, or `.npz` generation to satisfy it.
5. Use the same saved sequence and snapshots for the legacy and React comparisons.
6. Preserve the script as a benchmark/reference unless it is explicitly removed later.

The script includes historical Streamlit browser observations. Do not present those as React measurements. Keep these categories separate:

```text
Python/backend baseline
React frontend measurement
Historical Streamlit observation
```

The React report must independently include legacy Python average/max frame time, React fetch/cache time, parse time, render time, end-to-end frame time, actual FPS, and dropped frames. Do not claim that React improves the Python pipeline unless the unchanged Python pipeline was separately measured.

## Migration process

Before implementation, inspect the current frontend and determine:

1. Current visualizations
2. Data used by each visualization
3. Existing controls
4. Information displayed to users
5. Current frame navigation behavior
6. `.npz` fields actually required
7. What `frontend_simulation_latency.py` measures and whether it runs against the current checkout

Reproduce useful behavior only. Do not reproduce implementation details that exist only because Streamlit requires them.

## Deliverables

Produce:

1. React frontend
2. TypeScript components
3. Canvas/WebGL visualization
4. Frame playback controller
5. Frame provider abstraction
6. Lightweight backend API
7. Metadata endpoint
8. Browser performance instrumentation
9. Bounded frame cache and prefetching
10. README with frontend/backend run instructions
11. Before/after latency report using the legacy benchmark where runnable

Document the old/new architecture, API and frame formats, how 10 FPS is maintained, measured React FPS, render/data-loading time, legacy Python baseline, historical Streamlit observations, and remaining bottlenecks.

## Acceptance criteria

- [ ] Streamlit is no longer required for the frontend.
- [ ] Matplotlib is not used for React rendering.
- [ ] PNG frame generation is not used.
- [ ] Existing `.npz` outputs remain unchanged.
- [ ] Fusion, Rating, SalsaNext, and uncertainty generation remain unchanged.
- [ ] React displays the essential simulation information.
- [ ] Frame navigation and play/pause work.
- [ ] React targets 10 FPS.
- [ ] Actual FPS and frame time are measured rather than assumed.
- [ ] Frame rendering does not trigger a full Python rerun.
- [ ] Canvas/WebGL is used instead of thousands of DOM elements.
- [ ] Frames are cached and prefetched intelligently.
- [ ] Visualization is decoupled from the data source.
- [ ] REST can later be replaced by WebSocket without rewriting the renderer.
- [ ] `frontend_simulation_latency.py` was inspected and used as the legacy baseline where runnable.
- [ ] React does not import or execute `frontend_simulation_latency.py`.
- [ ] React performance numbers come from live browser instrumentation.
- [ ] Python, React, and historical Streamlit timings are reported separately.
- [ ] The same saved inputs are used for baseline and React comparison.

## Constraints

Do not rewrite the entire project. Do not modify Fusion, Rating, SalsaNext, uncertainty computation, or the existing data-generation pipeline. Do not introduce PNGs as an intermediate solution.

First make the React frontend work with existing saved data. Then benchmark it. Only after benchmarking should additional performance optimizations be introduced.
