"""
Lightweight ASGI backend server for LiDAR simulation frames and metadata.
Serves precomputed .npz snapshots via JSON and high-speed packed binary payloads.
"""

from __future__ import annotations

import os
import struct
from pathlib import Path
from typing import Any
import sys
import time
import threading

import numpy as np
import uvicorn
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route, WebSocketRoute
from starlette.websockets import WebSocket

ROOT = Path(__file__).resolve().parent.parent

if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
from fovmap.replay_pipeline import generate_prediction_sequence_in_memory

DATASET_ROOT = ROOT / "SalsaNext-Fork" / "dataset_test"
PREDICTION_ROOT = ROOT / "SalsaNext-Fork" / "predictions" / "uncertainty_valid"

# Metadata constants
SEQUENCE = "08"
TARGET_FPS = 10.0
FRAME_BUDGET_MS = 100.0
SEMANTIC_CLASSES = ["TERRAIN", "DRIVABLE", "STATIC", "OBJECT"]
RESOLUTIONS = [0.05, 0.10, 0.25, 0.50]
RING_BOUNDARIES = [5.0, 12.0, 25.0, 50.0]
EXTENT = [-50.0, 50.0, -50.0, 50.0]

TOTAL_FRAMES = 271
IN_MEMORY_FRAMES = []

def background_pipeline_worker():
    print("Starting in-memory pipeline background thread...")
    os.environ["FOVMAP_RATING_IMPL"] = "compiled"
    os.environ["FOVMAP_FUSION_IMPL"] = "cuda"
    try:
        generator = generate_prediction_sequence_in_memory(DATASET_ROOT, PREDICTION_ROOT, sequence="08")
        for frame_data in generator:
            IN_MEMORY_FRAMES.append(frame_data)
            if len(IN_MEMORY_FRAMES) % 10 == 0:
                print(f"Processed {len(IN_MEMORY_FRAMES)} / {TOTAL_FRAMES} frames into memory...")
        print("In-memory pipeline completed. All frames are cached in RAM.")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error in background pipeline: {e}")

threading.Thread(target=background_pipeline_worker, daemon=True).start()

def get_snapshot(frame_id: int) -> dict[str, Any]:
    if frame_id < 0 or frame_id >= TOTAL_FRAMES:
        raise IndexError(f"Frame {frame_id} out of bounds (0 to {TOTAL_FRAMES - 1})")
    
    # Block until frame is ready in memory
    while len(IN_MEMORY_FRAMES) <= frame_id:
        time.sleep(0.05)
        
    return IN_MEMORY_FRAMES[frame_id]


async def health(request: Request) -> JSONResponse:
    return JSONResponse({
        "status": "healthy",
        "service": "lidar-simulation-backend",
        "total_frames": TOTAL_FRAMES,
        "sequence": SEQUENCE,
    })


async def get_metadata(request: Request) -> JSONResponse:
    return JSONResponse({
        "sequence": SEQUENCE,
        "frame_count": TOTAL_FRAMES,
        "target_fps": TARGET_FPS,
        "frame_budget_ms": FRAME_BUDGET_MS,
        "semantic_classes": SEMANTIC_CLASSES,
        "resolutions": RESOLUTIONS,
        "ring_boundaries": RING_BOUNDARIES,
        "extent": EXTENT,
    })


async def list_frames(request: Request) -> JSONResponse:
    frames_summary = [
        {
            "frame_id": i,
            "filename": f"{i:06d}.bin",
        }
        for i in range(TOTAL_FRAMES)
    ]
    return JSONResponse(frames_summary)


async def get_frame_json(request: Request) -> JSONResponse:
    frame_id = int(request.path_params["frame_id"])
    if frame_id < 0 or frame_id >= TOTAL_FRAMES:
        return JSONResponse({"error": f"Frame {frame_id} not found"}, status_code=404)

    snapshot = get_snapshot(frame_id)
    cells = snapshot["cells"]
    ratings = snapshot["ratings"]
    dynamic_mask = snapshot["dynamic_mask"]

    num_cells = len(cells)
    return JSONResponse({
        "frame_id": frame_id,
        "num_cells": num_cells,
        "ring_id": cells["ring_id"].tolist(),
        "row": cells["row"].tolist(),
        "col": cells["col"].tolist(),
        "z_mean": np.round(cells["z_mean"], 3).tolist(),
        "semantic_label": cells["semantic_label"].tolist(),
        "ratings": np.round(ratings, 3).tolist(),
        "dynamic_mask": dynamic_mask.tolist(),
    })


async def get_performance(request: Request) -> JSONResponse:
    timings = [frame.get("timing_ms", {}) for frame in IN_MEMORY_FRAMES]
    return JSONResponse(timings)


async def get_frame_binary(request: Request) -> Response:
    frame_id = int(request.path_params["frame_id"])
    if frame_id < 0 or frame_id >= TOTAL_FRAMES:
        return JSONResponse({"error": f"Frame {frame_id} not found"}, status_code=404)

    snapshot = get_snapshot(frame_id)
    cells = snapshot["cells"]
    ratings = snapshot["ratings"]
    dynamic_mask = snapshot["dynamic_mask"]
    num_cells = len(cells)

    # 32-byte header:
    # magic: 0x4C494452 ('LIDR'), version: 1, frame_id, num_cells, 16 bytes reserved
    header = struct.pack(
        "<IIII16x",
        0x4C494452,
        1,
        frame_id,
        num_cells,
    )

    # Contiguous naturally-aligned arrays:
    # 1. z_mean (Float32, offset 32, 4-byte aligned)
    # 2. ratings (Float32, offset 32 + 4N, 4-byte aligned)
    # 3. col (Int16, offset 32 + 8N, 2-byte aligned)
    # 4. row (Int16, offset 32 + 10N, 2-byte aligned)
    # 5. ring_id (Uint8, offset 32 + 12N)
    # 6. semantic_label (Uint8, offset 32 + 13N)
    # 7. dynamic_mask (Uint8, offset 32 + 14N)
    z_mean_bytes = cells["z_mean"].astype(np.float32).tobytes()
    ratings_bytes = ratings.astype(np.float32).tobytes()
    col_bytes = cells["col"].astype(np.int16).tobytes()
    row_bytes = cells["row"].astype(np.int16).tobytes()
    ring_id_bytes = cells["ring_id"].astype(np.uint8).tobytes()
    semantic_bytes = cells["semantic_label"].astype(np.uint8).tobytes()
    dynamic_bytes = dynamic_mask.astype(np.uint8).tobytes()

    payload = b"".join([
        header,
        z_mean_bytes,
        ratings_bytes,
        col_bytes,
        row_bytes,
        ring_id_bytes,
        semantic_bytes,
        dynamic_bytes,
    ])

    return Response(
        content=payload,
        media_type="application/octet-stream",
        headers={
            "X-Frame-ID": str(frame_id),
            "X-Num-Cells": str(num_cells),
            "Cache-Control": "public, max-age=3600",
        },
    )


async def live_frame_websocket(websocket: WebSocket) -> None:
    """Future live-frame subscription endpoint."""
    await websocket.accept()
    try:
        while True:
            msg = await websocket.receive_text()
            if msg == "ping":
                await websocket.send_text("pong")
    except Exception:
        pass
    finally:
        await websocket.close()


routes = [
    Route("/api/health", health, methods=["GET"]),
    Route("/api/metadata", get_metadata, methods=["GET"]),
    Route("/api/frames", list_frames, methods=["GET"]),
    Route("/api/frames/{frame_id:int}", get_frame_json, methods=["GET"]),
    Route("/api/frames/{frame_id:int}/binary", get_frame_binary, methods=["GET"]),
    Route("/api/performance", get_performance, methods=["GET"]),
    WebSocketRoute("/ws/live", live_frame_websocket),
]

middleware = [
    Middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Frame-ID", "X-Num-Cells"],
    )
]

app = Starlette(debug=False, routes=routes, middleware=middleware)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"Starting LiDAR Simulation Backend on port {port}...")
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="info")
