import React, { useRef, useEffect, useState, useCallback } from 'react';
import { ColorMode, FrameData, HoveredCellInfo, SimulationMetadata } from '../types/simulation';

interface LidarCanvasProps {
  frame: FrameData | null;
  metadata: SimulationMetadata | null;
  colorMode: ColorMode;
  onHoverCell?: (info: HoveredCellInfo | null) => void;
  onRenderComplete?: (renderMs: number) => void;
}

// 4-Class Semantics (High-contrast: 0: Warm Earth Brown, 1: Emerald Green, 2: Cool Steel Slate, 3: Vivid Coral Red)
const PALETTE_SEMANTICS = [
  'rgb(165, 142, 108)', // 0: TERRAIN (Warm earth brown/ochre)
  'rgb(16, 194, 133)',  // 1: DRIVABLE (Vibrant emerald green)
  'rgb(95, 135, 175)',  // 2: STATIC (Cool industrial steel blue)
  'rgb(255, 50, 50)',   // 3: OBJECT (Bright blaze red)
];

// Ring Resolutions (0: 5cm, 1: 10cm, 2: 25cm, 3: 50cm)
const PALETTE_RINGS = [
  'rgb(38, 115, 242)',  // Ring 0: Neon Blue
  'rgb(26, 184, 89)',   // Ring 1: Emerald Green
  'rgb(250, 166, 20)',  // Ring 2: Amber Orange
  'rgb(235, 51, 51)',   // Ring 3: Crimson Red
];

// Precomputed 64-color elevation colormap (Turbo/Terrain style)
const ELEVATION_LUT: string[] = [];
for (let i = 0; i < 64; i++) {
  const t = i / 63;
  // Deep blue -> cyan -> green -> yellow -> red
  let r = 0, g = 0, b = 0;
  if (t < 0.25) {
    const k = t / 0.25;
    r = Math.round(20 + 30 * k);
    g = Math.round(50 + 150 * k);
    b = Math.round(180 + 75 * k);
  } else if (t < 0.5) {
    const k = (t - 0.25) / 0.25;
    r = Math.round(50 + 50 * k);
    g = Math.round(200 + 40 * k);
    b = Math.round(255 - 150 * k);
  } else if (t < 0.75) {
    const k = (t - 0.5) / 0.25;
    r = Math.round(100 + 140 * k);
    g = Math.round(240 - 40 * k);
    b = Math.round(105 - 80 * k);
  } else {
    const k = (t - 0.75) / 0.25;
    r = Math.round(240 + 15 * k);
    g = Math.round(200 - 150 * k);
    b = Math.round(25 + 10 * k);
  }
  ELEVATION_LUT.push(`rgb(${r},${g},${b})`);
}

// Precomputed 32-color rating colormap (Traversability: 0 = Hazard (Red), 50 = Caution (Amber), 100 = Safe (Emerald Green))
const RATING_LUT: string[] = [];
for (let i = 0; i < 32; i++) {
  const t = i / 31; // 0 = Hazard (0), 1 = Safe (100)
  let r = 0, g = 0, b = 0;
  if (t < 0.5) {
    // 0 (Red) -> 0.5 (Amber)
    const k = t / 0.5;
    r = Math.round(235 + 15 * k);
    g = Math.round(45 + 135 * k);
    b = Math.round(45 - 20 * k);
  } else {
    // 0.5 (Amber) -> 1.0 (Emerald Green)
    const k = (t - 0.5) / 0.5;
    r = Math.round(250 - 234 * k);
    g = Math.round(180 + 14 * k);
    b = Math.round(25 + 108 * k);
  }
  RATING_LUT.push(`rgb(${r},${g},${b})`);
}

export const LidarCanvas: React.FC<LidarCanvasProps> = ({
  frame,
  metadata,
  colorMode,
  onHoverCell,
  onRenderComplete,
}) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const containerRef = useRef<HTMLDivElement | null>(null);

  // Viewport transform: pan offset in meters, zoom in pixels per meter
  const [viewTransform, setViewTransform] = useState({
    panX: 0, // meters
    panY: 0, // meters
    zoom: 7.0, // pixels per meter
  });

  const isDraggingRef = useRef(false);
  const dragStartRef = useRef({ x: 0, y: 0, panX: 0, panY: 0 });
  const viewTransformRef = useRef(viewTransform);
  viewTransformRef.current = viewTransform;

  // Reset view to center on vehicle origin
  const resetView = useCallback(() => {
    if (!containerRef.current) return;
    const { clientWidth, clientHeight } = containerRef.current;
    const minDim = Math.min(clientWidth, clientHeight);
    const initialZoom = (minDim * 0.85) / 100.0; // Fit 100m (-50 to +50m) comfortably
    setViewTransform({
      panX: 0,
      panY: 0,
      zoom: Math.max(4.0, initialZoom),
    });
  }, []);

  // Initialize view once container is sized
  useEffect(() => {
    resetView();
  }, [resetView]);

  // Main high-performance render effect
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !frame) return;

    const ctx = canvas.getContext('2d', { alpha: false });
    if (!ctx) return;

    const tStart = performance.now();

    const width = canvas.width;
    const height = canvas.height;
    const { panX, panY, zoom } = viewTransformRef.current;

    // Origin in canvas coordinates
    const originX = width / 2 + panX * zoom;
    const originY = height / 2 - panY * zoom; // Invert Y so positive Y is up

    // Clear background
    ctx.fillStyle = '#0f1715';
    ctx.fillRect(0, 0, width, height);

    // Draw concentric distance circles (5m, 12m, 25m, 50m)
    const boundaries = metadata?.ring_boundaries || [5.0, 12.0, 25.0, 50.0];
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 4]);
    ctx.strokeStyle = 'rgba(120, 199, 165, 0.22)';
    ctx.fillStyle = 'rgba(120, 199, 165, 0.55)';
    ctx.font = '10px monospace';

    for (const r of boundaries) {
      const radiusPx = r * zoom;
      ctx.beginPath();
      ctx.arc(originX, originY, radiusPx, 0, Math.PI * 2);
      ctx.stroke();

      // Label on top axis
      ctx.fillText(`${r}m`, originX + 4, originY - radiusPx + 12);
    }
    ctx.setLineDash([]); // Reset dash

    // Draw axis crosshair lines
    ctx.strokeStyle = 'rgba(51, 69, 63, 0.4)';
    ctx.beginPath();
    ctx.moveTo(0, originY);
    ctx.lineTo(width, originY);
    ctx.moveTo(originX, 0);
    ctx.lineTo(originX, height);
    ctx.stroke();

    // Batch rendering: group cells into buckets by color to minimize fillStyle switches
    const resolutions = metadata?.resolutions || [0.05, 0.10, 0.25, 0.50];
    const numCells = frame.numCells;
    const ringIds = frame.ringId;
    const cols = frame.col;
    const rows = frame.row;
    const zMeans = frame.zMean;
    const labels = frame.semanticLabel;
    const ratings = frame.ratings;
    const dynamics = frame.dynamicMask;

    // Bucket indices for batched rendering
    let bucketCount = 4;
    if (colorMode === 'elevation') bucketCount = 64;
    else if (colorMode === 'rating') bucketCount = 32;
    else if (colorMode === 'dynamic') bucketCount = 4;

    const buckets: number[][] = Array.from({ length: bucketCount }, () => []);

    for (let i = 0; i < numCells; i++) {
      let bucketIdx = 0;
      if (colorMode === 'semantic') {
        bucketIdx = labels[i] % 4;
      } else if (colorMode === 'resolution') {
        bucketIdx = ringIds[i] % 4;
      } else if (colorMode === 'dynamic') {
        const sem = labels[i];
        const isDyn = dynamics[i];
        // 3: Dynamic Obstacle (OBJECT class, or static obstacle flagged with temporal motion)
        // 2: Static Obstacle (Walls, buildings, poles without motion)
        // 1: Drivable Road (Static ground - not an obstacle, even if minor ray fluctuation)
        // 0: Terrain
        if (sem === 3 || (sem === 2 && isDyn)) {
          bucketIdx = 3;
        } else if (sem === 2) {
          bucketIdx = 2;
        } else if (sem === 1) {
          bucketIdx = 1;
        } else {
          bucketIdx = 0;
        }
      } else if (colorMode === 'elevation') {
        // Map zMean (-10m to +3m) to 0..63
        const z = zMeans[i];
        const normalized = Math.max(0, Math.min(1, (z + 10.0) / 13.0));
        bucketIdx = Math.min(63, Math.floor(normalized * 64));
      } else if (colorMode === 'rating') {
        // Map rating (0..100) to 0..31: 0 = Hazard (Red), 100 = Safe (Green)
        const r = ratings[i];
        const normalized = Math.max(0, Math.min(1, r / 100.0));
        bucketIdx = Math.min(31, Math.floor(normalized * 32));
      }
      buckets[bucketIdx].push(i);
    }

    // Render cells bucket by bucket
    for (let b = 0; b < bucketCount; b++) {
      const cellIndices = buckets[b];
      if (cellIndices.length === 0) continue;

      let color = '#78c7a5';
      if (colorMode === 'semantic') {
        color = PALETTE_SEMANTICS[b];
      } else if (colorMode === 'resolution') {
        color = PALETTE_RINGS[b];
      } else if (colorMode === 'dynamic') {
        const DYN_COLORS = [
          '#141a18',            // 0: Terrain (Muted dark ground)
          '#18362b',            // 1: Drivable Road (Dark emerald ground)
          'rgb(100, 135, 175)',  // 2: Static Obstacle (Cool Steel Blue / Slate)
          'rgb(255, 230, 0)',   // 3: Dynamic Obstacle (Fluorescent High-Vis Neon Yellow)
        ];
        color = DYN_COLORS[b];
      } else if (colorMode === 'elevation') {
        color = ELEVATION_LUT[b];
      } else if (colorMode === 'rating') {
        color = RATING_LUT[b];
      }

      ctx.fillStyle = color;

      for (let k = 0; k < cellIndices.length; k++) {
        const idx = cellIndices[k];
        const ring = ringIds[idx];
        const res = resolutions[ring];
        const col = cols[idx];
        const row = rows[idx];

        // World coordinates: x = (col + 0.5) * res, y = (row + 0.5) * res
        // Screen top-left corner:
        const worldLeft = col * res;
        const worldTop = (row + 1) * res;

        const screenLeft = originX + worldLeft * zoom;
        const screenTop = originY - worldTop * zoom;
        const cellSizePx = Math.max(1.2, res * zoom);

        // Viewport culling
        if (
          screenLeft + cellSizePx < 0 ||
          screenLeft > width ||
          screenTop + cellSizePx < 0 ||
          screenTop > height
        ) {
          continue;
        }

        ctx.fillRect(screenLeft, screenTop, cellSizePx, cellSizePx);
      }
    }

    // Draw sensor origin marker (Diamond at 0, 0)
    ctx.save();
    ctx.fillStyle = '#e08f5b';
    ctx.strokeStyle = '#f4eadf';
    ctx.lineWidth = 2;
    ctx.beginPath();
    const dSize = 7;
    ctx.moveTo(originX, originY - dSize);
    ctx.lineTo(originX + dSize, originY);
    ctx.lineTo(originX, originY + dSize);
    ctx.lineTo(originX - dSize, originY);
    ctx.closePath();
    ctx.fill();
    ctx.stroke();

    // Sensor label
    ctx.fillStyle = '#e08f5b';
    ctx.font = '11px sans-serif';
    ctx.fillText('Sensor (0,0)', originX + 10, originY + 4);
    ctx.restore();

    const tRender = performance.now() - tStart;
    if (onRenderComplete) {
      onRenderComplete(tRender);
    }
  }, [frame, metadata, colorMode, viewTransform, onRenderComplete]);

  // Handle Canvas Resizing
  useEffect(() => {
    const handleResize = () => {
      if (!containerRef.current || !canvasRef.current) return;
      const { clientWidth, clientHeight } = containerRef.current;
      canvasRef.current.width = clientWidth;
      canvasRef.current.height = clientHeight;
    };

    handleResize();
    const ro = new ResizeObserver(handleResize);
    if (containerRef.current) {
      ro.observe(containerRef.current);
    }
    return () => ro.disconnect();
  }, []);

  // Mouse Interaction: Pan and Zoom
  const handleMouseDown = (e: React.MouseEvent) => {
    isDraggingRef.current = true;
    dragStartRef.current = {
      x: e.clientX,
      y: e.clientY,
      panX: viewTransformRef.current.panX,
      panY: viewTransformRef.current.panY,
    };
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (isDraggingRef.current) {
      const dx = e.clientX - dragStartRef.current.x;
      const dy = e.clientY - dragStartRef.current.y;
      const zoom = viewTransformRef.current.zoom;
      setViewTransform(prev => ({
        ...prev,
        panX: dragStartRef.current.panX + dx / zoom,
        panY: dragStartRef.current.panY - dy / zoom,
      }));
      return;
    }

    // Hover inspection hit-test
    if (!frame || !canvasRef.current || !onHoverCell) return;
    const rect = canvasRef.current.getBoundingClientRect();
    const mouseCanvasX = e.clientX - rect.left;
    const mouseCanvasY = e.clientY - rect.top;

    const width = canvasRef.current.width;
    const height = canvasRef.current.height;
    const { panX, panY, zoom } = viewTransformRef.current;

    const originX = width / 2 + panX * zoom;
    const originY = height / 2 - panY * zoom;

    const worldX = (mouseCanvasX - originX) / zoom;
    const worldY = (originY - mouseCanvasY) / zoom;

    // Find cell containing or nearest to (worldX, worldY)
    const resolutions = metadata?.resolutions || [0.05, 0.10, 0.25, 0.50];
    const semClasses = metadata?.semantic_classes || ['TERRAIN', 'DRIVABLE', 'STATIC', 'OBJECT'];

    const numCells = frame.numCells;
    const ringIds = frame.ringId;
    const cols = frame.col;
    const rows = frame.row;

    let closestIdx = -1;
    let minDistanceSq = 1.0; // 1m radius search limit

    for (let i = 0; i < numCells; i += 4) { // Fast subsample search
      const res = resolutions[ringIds[i]];
      const cx = (cols[i] + 0.5) * res;
      const cy = (rows[i] + 0.5) * res;
      const distSq = (worldX - cx) ** 2 + (worldY - cy) ** 2;
      if (distSq < minDistanceSq) {
        minDistanceSq = distSq;
        closestIdx = i;
      }
    }

    if (closestIdx !== -1 && minDistanceSq < 0.36) {
      const ring = ringIds[closestIdx];
      const res = resolutions[ring];
      const sem = frame.semanticLabel[closestIdx];
      const isDyn = Boolean(frame.dynamicMask[closestIdx]);
      const isDynamicObstacle = sem === 3 || (sem === 2 && isDyn);
      onHoverCell({
        index: closestIdx,
        ringId: ring,
        resolution: res,
        row: rows[closestIdx],
        col: cols[closestIdx],
        x: Math.round((cols[closestIdx] + 0.5) * res * 100) / 100,
        y: Math.round((rows[closestIdx] + 0.5) * res * 100) / 100,
        zMean: Math.round(frame.zMean[closestIdx] * 100) / 100,
        semanticClass: semClasses[sem] || 'UNKNOWN',
        rating: Math.round(frame.ratings[closestIdx] * 10) / 10,
        dynamic: isDynamicObstacle,
      });
    } else {
      onHoverCell(null);
    }
  };

  const handleMouseUp = () => {
    isDraggingRef.current = false;
  };

  const handleWheel = (e: React.WheelEvent) => {
    e.preventDefault();
    const factor = e.deltaY < 0 ? 1.15 : 0.87;
    setViewTransform(prev => ({
      ...prev,
      zoom: Math.max(1.5, Math.min(80.0, prev.zoom * factor)),
    }));
  };

  const zoomIn = () => {
    setViewTransform(prev => ({ ...prev, zoom: Math.min(80.0, prev.zoom * 1.3) }));
  };

  const zoomOut = () => {
    setViewTransform(prev => ({ ...prev, zoom: Math.max(1.5, prev.zoom * 0.77) }));
  };

  return (
    <div
      ref={containerRef}
      className="relative w-full h-full overflow-hidden bg-slate-950 select-none cursor-crosshair"
      onMouseDown={handleMouseDown}
      onMouseMove={handleMouseMove}
      onMouseUp={handleMouseUp}
      onMouseLeave={handleMouseUp}
      onWheel={handleWheel}
      onDoubleClick={resetView}
    >
      <canvas ref={canvasRef} className="block w-full h-full" />

      {/* Floating Canvas View Controls */}
      <div className="absolute top-4 right-4 flex flex-col gap-2 z-10 bg-slate-900/85 backdrop-blur-md border border-slate-700/60 rounded-lg p-1.5 shadow-xl text-slate-300">
        <button
          onClick={zoomIn}
          title="Zoom In"
          className="p-2 hover:bg-slate-700/60 hover:text-white rounded transition text-sm font-bold flex items-center justify-center w-8 h-8"
        >
          +
        </button>
        <button
          onClick={zoomOut}
          title="Zoom Out"
          className="p-2 hover:bg-slate-700/60 hover:text-white rounded transition text-sm font-bold flex items-center justify-center w-8 h-8"
        >
          −
        </button>
        <div className="h-px bg-slate-700/60 my-0.5" />
        <button
          onClick={resetView}
          title="Reset View (Center Origin)"
          className="p-1 hover:bg-slate-700/60 hover:text-white rounded transition text-xs font-medium flex items-center justify-center w-8 h-8"
        >
          ⌖
        </button>
      </div>

      {/* Legend Badge Overlay */}
      <div className="absolute bottom-4 left-4 z-10 bg-slate-900/85 backdrop-blur-md border border-slate-700/60 rounded-lg px-3 py-2 shadow-xl text-xs text-slate-300 flex items-center gap-4">
        <span className="font-semibold text-slate-200">
          {colorMode.toUpperCase()}:
        </span>
        {colorMode === 'semantic' && (
          <div className="flex items-center gap-3">
            <span className="flex items-center gap-1.5">
              <span className="w-2.5 h-2.5 rounded-sm inline-block" style={{ background: PALETTE_SEMANTICS[0] }} />
              Terrain
            </span>
            <span className="flex items-center gap-1.5">
              <span className="w-2.5 h-2.5 rounded-sm inline-block" style={{ background: PALETTE_SEMANTICS[1] }} />
              Drivable
            </span>
            <span className="flex items-center gap-1.5">
              <span className="w-2.5 h-2.5 rounded-sm inline-block" style={{ background: PALETTE_SEMANTICS[2] }} />
              Static
            </span>
            <span className="flex items-center gap-1.5">
              <span className="w-2.5 h-2.5 rounded-sm inline-block" style={{ background: PALETTE_SEMANTICS[3] }} />
              Object
            </span>
          </div>
        )}
        {colorMode === 'resolution' && (
          <div className="flex items-center gap-3">
            <span className="flex items-center gap-1.5"><span className="w-2.5 h-2.5 rounded-sm inline-block" style={{ background: PALETTE_RINGS[0] }} /> 5cm</span>
            <span className="flex items-center gap-1.5"><span className="w-2.5 h-2.5 rounded-sm inline-block" style={{ background: PALETTE_RINGS[1] }} /> 10cm</span>
            <span className="flex items-center gap-1.5"><span className="w-2.5 h-2.5 rounded-sm inline-block" style={{ background: PALETTE_RINGS[2] }} /> 25cm</span>
            <span className="flex items-center gap-1.5"><span className="w-2.5 h-2.5 rounded-sm inline-block" style={{ background: PALETTE_RINGS[3] }} /> 50cm</span>
          </div>
        )}
        {colorMode === 'elevation' && (
          <div className="flex items-center gap-2">
            <span>-10m</span>
            <div className="w-24 h-2.5 rounded-sm" style={{ background: 'linear-gradient(to right, rgb(20,50,180), rgb(50,200,255), rgb(100,240,105), rgb(255,20,10))' }} />
            <span>+3m</span>
          </div>
        )}
        {colorMode === 'rating' && (
          <div className="flex items-center gap-2">
            <span className="flex items-center gap-1"><span className="w-2.5 h-2.5 rounded-sm inline-block" style={{ background: 'rgb(235,45,45)' }} /> Hazard (0)</span>
            <div className="w-24 h-2.5 rounded-sm" style={{ background: 'linear-gradient(to right, rgb(235,45,45), rgb(250,180,20), rgb(16,194,133))' }} />
            <span className="flex items-center gap-1"><span className="w-2.5 h-2.5 rounded-sm inline-block" style={{ background: 'rgb(16,194,133)' }} /> Safe (100)</span>
          </div>
        )}
        {colorMode === 'dynamic' && (
          <div className="flex items-center gap-3">
            <span className="flex items-center gap-1.5"><span className="w-2.5 h-2.5 rounded-sm inline-block" style={{ background: 'rgb(255, 230, 0)' }} /> Dynamic Obstacle</span>
            <span className="flex items-center gap-1.5"><span className="w-2.5 h-2.5 rounded-sm inline-block" style={{ background: 'rgb(100, 135, 175)' }} /> Static Obstacle</span>
            <span className="flex items-center gap-1.5"><span className="w-2.5 h-2.5 rounded-sm inline-block" style={{ background: '#18362b' }} /> Road</span>
          </div>
        )}
      </div>
    </div>
  );
};
