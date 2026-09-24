export type SemanticClassName = 'TERRAIN' | 'DRIVABLE' | 'STATIC' | 'OBJECT';

export type ColorMode = 'semantic' | 'elevation' | 'rating' | 'resolution' | 'dynamic';
export type AppView = 'canvas' | 'performance';

export interface PipelineTiming {
  io_and_prep: number;
  grid_build: number;
  fusion: number;
  rating: number;
  total: number;
}

export interface SimulationMetadata {
  sequence: string;
  frame_count: number;
  target_fps: number;
  frame_budget_ms: number;
  semantic_classes: string[];
  resolutions: number[];
  ring_boundaries: number[];
  extent: number[];
}

export interface FrameData {
  frameId: number;
  numCells: number;
  ringId: Uint8Array;
  row: Int16Array;
  col: Int16Array;
  zMean: Float32Array;
  semanticLabel: Uint8Array;
  ratings: Float32Array;
  dynamicMask: Uint8Array;
  // Computed or cached fields
  receivedAt?: number;
  isBinary?: boolean;
}

export interface HoveredCellInfo {
  index: number;
  ringId: number;
  resolution: number;
  row: number;
  col: number;
  x: number;
  y: number;
  zMean: number;
  semanticClass: string;
  rating: number;
  dynamic: boolean;
}

export interface PerformanceMetrics {
  fetchTimeMs: number;
  parseTimeMs: number;
  updateTimeMs: number;
  renderTimeMs: number;
  totalFrameTimeMs: number;
  actualFps: number;
  avgFps: number;
  maxFrameTimeMs: number;
  droppedFrames: number;
  cacheHits: number;
  cacheMisses: number;
  cachedFramesCount: number;
}
