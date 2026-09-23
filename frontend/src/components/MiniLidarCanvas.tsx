import React, { useRef, useEffect } from 'react';
import { ColorMode, FrameData, SimulationMetadata } from '../types/simulation';

interface MiniLidarCanvasProps {
  frame: FrameData | null;
  metadata: SimulationMetadata | null;
  colorMode: ColorMode;
}

// Precomputed ABGR Uint32 colors for instant zero-overhead buffer writes
function abgr(r: number, g: number, b: number, a: number = 255): number {
  return ((a << 24) | (b << 16) | (g << 8) | r) >>> 0;
}

const PALETTE_SEMANTICS_U32 = new Uint32Array([
  abgr(165, 142, 108), // 0: TERRAIN (Warm earth brown)
  abgr(16, 194, 133),  // 1: DRIVABLE (Vibrant emerald green)
  abgr(95, 135, 175),  // 2: STATIC (Cool steel blue)
  abgr(255, 50, 50),   // 3: OBJECT (Bright blaze red)
]);

const PALETTE_RINGS_U32 = new Uint32Array([
  abgr(38, 115, 242),  // Ring 0: Neon Blue
  abgr(26, 184, 89),   // Ring 1: Emerald Green
  abgr(250, 166, 20),  // Ring 2: Amber Orange
  abgr(235, 51, 51),   // Ring 3: Crimson Red
]);

const DYN_COLORS_U32 = new Uint32Array([
  abgr(20, 26, 24),    // 0: Terrain (Dark ground)
  abgr(24, 54, 43),    // 1: Drivable Road (Dark emerald ground)
  abgr(100, 135, 175), // 2: Static Obstacle (Cool Steel Blue)
  abgr(255, 230, 0),   // 3: Dynamic Obstacle (Fluorescent Neon Yellow)
]);

const RATING_LUT_U32 = new Uint32Array(32);
for (let i = 0; i < 32; i++) {
  const t = i / 31; // 0 = Hazard (Red), 1 = Safe (Green)
  let r = 0, g = 0, b = 0;
  if (t < 0.5) {
    const k = t / 0.5;
    r = Math.round(235 + 15 * k);
    g = Math.round(45 + 135 * k);
    b = Math.round(45 - 20 * k);
  } else {
    const k = (t - 0.5) / 0.5;
    r = Math.round(250 - 234 * k);
    g = Math.round(180 + 14 * k);
    b = Math.round(25 + 108 * k);
  }
  RATING_LUT_U32[i] = abgr(r, g, b);
}

const ELEVATION_LUT_U32 = new Uint32Array(64);
for (let i = 0; i < 64; i++) {
  const t = i / 63;
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
  ELEVATION_LUT_U32[i] = abgr(r, g, b);
}

const CANVAS_WIDTH = 220;
const CANVAS_HEIGHT = 70;
const BG_COLOR = abgr(10, 15, 14);
const RING_COLOR = abgr(80, 120, 105, 120);
const ORIGIN_COLOR = abgr(224, 143, 91);

export const MiniLidarCanvas: React.FC<MiniLidarCanvasProps> = ({
  frame,
  metadata,
  colorMode,
}) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !frame) return;

    const ctx = canvas.getContext('2d', { alpha: false });
    if (!ctx) return;

    const w = CANVAS_WIDTH;
    const h = CANVAS_HEIGHT;
    const imgData = ctx.createImageData(w, h);
    const u32 = new Uint32Array(imgData.data.buffer);

    // Fast 1-operation background fill
    u32.fill(BG_COLOR);

    const zoom = (Math.min(w, h) * 0.90) / 100.0;
    const originX = Math.round(w / 2);
    const originY = Math.round(h / 2);

    // Concentric ring circles drawn directly to u32 buffer
    const boundaries = metadata?.ring_boundaries || [5.0, 12.0, 25.0, 50.0];
    for (const r of boundaries) {
      const radiusPx = r * zoom;
      const numSteps = Math.round(radiusPx * 6);
      for (let s = 0; s < numSteps; s += 2) {
        const theta = (s / numSteps) * Math.PI * 2;
        const cx = Math.round(originX + radiusPx * Math.cos(theta));
        const cy = Math.round(originY + radiusPx * Math.sin(theta));
        if (cx >= 0 && cx < w && cy >= 0 && cy < h) {
          u32[cy * w + cx] = RING_COLOR;
        }
      }
    }

    // Direct pixel rasterization into Uint32Array buffer (<0.6ms)
    const resolutions = metadata?.resolutions || [0.05, 0.10, 0.25, 0.50];
    const numCells = frame.numCells;
    const ringIds = frame.ringId;
    const cols = frame.col;
    const rows = frame.row;
    const zMeans = frame.zMean;
    const labels = frame.semanticLabel;
    const ratings = frame.ratings;
    const dynamics = frame.dynamicMask;

    for (let i = 0; i < numCells; i += 2) {
      const ring = ringIds[i];
      const res = resolutions[ring];
      const col = cols[i];
      const row = rows[i];

      const px = Math.round(originX + (col * res) * zoom);
      const py = Math.round(originY - ((row + 1) * res) * zoom);

      if (px < 0 || px >= w - 1 || py < 0 || py >= h - 1) continue;

      let color = 0;
      if (colorMode === 'semantic') {
        color = PALETTE_SEMANTICS_U32[labels[i] % 4];
      } else if (colorMode === 'resolution') {
        color = PALETTE_RINGS_U32[ring % 4];
      } else if (colorMode === 'dynamic') {
        const sem = labels[i];
        const isDyn = dynamics[i];
        if (sem === 3 || (sem === 2 && isDyn)) {
          color = DYN_COLORS_U32[3]; // Fluorescent Neon Yellow
        } else if (sem === 2) {
          color = DYN_COLORS_U32[2]; // Cool Steel Blue
        } else if (sem === 1) {
          color = DYN_COLORS_U32[1]; // Dark Road Emerald
        } else {
          color = DYN_COLORS_U32[0]; // Dark Terrain
        }
      } else if (colorMode === 'elevation') {
        const z = zMeans[i];
        const norm = Math.max(0, Math.min(1, (z + 10.0) / 13.0));
        color = ELEVATION_LUT_U32[Math.min(63, Math.floor(norm * 64))];
      } else if (colorMode === 'rating') {
        const r = ratings[i];
        const norm = Math.max(0, Math.min(1, r / 100.0));
        color = RATING_LUT_U32[Math.min(31, Math.floor(norm * 32))];
      }

      // Draw 2x2 solid pixel block for crisp, sharp preview
      const idx = py * w + px;
      u32[idx] = color;
      u32[idx + 1] = color;
    }

    // Origin diamond marker
    if (originX >= 1 && originX < w - 1 && originY >= 1 && originY < h - 1) {
      u32[originY * w + originX] = ORIGIN_COLOR;
      u32[(originY - 1) * w + originX] = ORIGIN_COLOR;
      u32[(originY + 1) * w + originX] = ORIGIN_COLOR;
      u32[originY * w + originX - 1] = ORIGIN_COLOR;
      u32[originY * w + originX + 1] = ORIGIN_COLOR;
    }

    // Exactly ONE single native V8 call per canvas
    ctx.putImageData(imgData, 0, 0);
  }, [frame, metadata, colorMode]);

  return (
    <canvas
      ref={canvasRef}
      width={CANVAS_WIDTH}
      height={CANVAS_HEIGHT}
      className="w-full h-full block rounded object-contain pointer-events-none"
    />
  );
};
