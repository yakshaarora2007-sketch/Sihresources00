import { FrameData, SimulationMetadata } from '../types/simulation';

export interface FrameFetchResult {
  frame: FrameData;
  fetchMs: number;
  parseMs: number;
  cacheHit: boolean;
}

export interface FrameProvider {
  getMetadata(): Promise<SimulationMetadata>;
  getFrame(frameId: number): Promise<FrameFetchResult>;
  prefetchFrame(frameId: number): void;
  subscribeLiveFrames?(callback: (frame: FrameData) => void): () => void;
  getCachedFrameCount(): number;
  isCached(frameId: number): boolean;
  clearCache(): void;
}

const BINARY_MAGIC = 0x4C494452; // 'LIDR'

export class RestFrameProvider implements FrameProvider {
  private baseUrl: string;
  private metadataCache: SimulationMetadata | null = null;
  private frameCache = new Map<number, FrameData>();
  private inFlightFetches = new Map<number, Promise<FrameData>>();
  private maxCacheSize: number;

  constructor(baseUrl: string = '', maxCacheSize: number = 15) {
    this.baseUrl = baseUrl.replace(/\/+$/, '');
    this.maxCacheSize = maxCacheSize;
  }

  async getMetadata(): Promise<SimulationMetadata> {
    if (this.metadataCache) {
      return this.metadataCache;
    }
    const res = await fetch(`${this.baseUrl}/api/metadata`);
    if (!res.ok) {
      throw new Error(`Failed to load simulation metadata: ${res.statusText}`);
    }
    this.metadataCache = (await res.json()) as SimulationMetadata;
    return this.metadataCache;
  }

  getCachedFrameCount(): number {
    return this.frameCache.size;
  }

  isCached(frameId: number): boolean {
    return this.frameCache.has(frameId);
  }

  clearCache(): void {
    this.frameCache.clear();
  }

  private pruneCache(currentFrameId: number): void {
    if (this.frameCache.size <= this.maxCacheSize) {
      return;
    }

    // Keep frames closest to currentFrameId, prioritizing forward frames
    const entries = Array.from(this.frameCache.keys());
    entries.sort((a, b) => {
      // Forward distance has slight advantage
      const distA = a >= currentFrameId ? (a - currentFrameId) : (currentFrameId - a) * 1.5;
      const distB = b >= currentFrameId ? (b - currentFrameId) : (currentFrameId - b) * 1.5;
      return distB - distA; // furthest first
    });

    while (this.frameCache.size > this.maxCacheSize && entries.length > 0) {
      const furthest = entries.shift()!;
      this.frameCache.delete(furthest);
    }
  }

  private async fetchFrameFromNetwork(frameId: number): Promise<{ frame: FrameData; fetchMs: number; parseMs: number }> {
    const t0 = performance.now();
    
    // Try binary transport first for ultra-low latency (<0.2ms decode)
    try {
      const response = await fetch(`${this.baseUrl}/api/frames/${frameId}/binary`, {
        headers: { Accept: 'application/octet-stream' },
      });

      if (!response.ok) {
        throw new Error(`Binary endpoint returned ${response.status}`);
      }

      const fetchMs = performance.now() - t0;
      const tParse = performance.now();
      const buffer = await response.arrayBuffer();

      const view = new DataView(buffer);
      const magic = view.getUint32(0, true);
      if (magic !== BINARY_MAGIC) {
        throw new Error(`Invalid binary magic: 0x${magic.toString(16)}`);
      }

      const receivedFrameId = view.getUint32(8, true);
      const numCells = view.getUint32(12, true);

      // Extract typed arrays with exact zero-copy byte offsets
      let offset = 32;
      const zMean = new Float32Array(buffer, offset, numCells);
      offset += numCells * 4;

      const ratings = new Float32Array(buffer, offset, numCells);
      offset += numCells * 4;

      const col = new Int16Array(buffer, offset, numCells);
      offset += numCells * 2;

      const row = new Int16Array(buffer, offset, numCells);
      offset += numCells * 2;

      const ringId = new Uint8Array(buffer, offset, numCells);
      offset += numCells;

      const semanticLabel = new Uint8Array(buffer, offset, numCells);
      offset += numCells;

      const dynamicMask = new Uint8Array(buffer, offset, numCells);

      const parseMs = performance.now() - tParse;

      const frame: FrameData = {
        frameId: receivedFrameId,
        numCells,
        ringId,
        row,
        col,
        zMean,
        semanticLabel,
        ratings,
        dynamicMask,
        receivedAt: performance.now(),
        isBinary: true,
      };

      return { frame, fetchMs, parseMs };
    } catch (binErr) {
      // Fallback to JSON transport if binary is unavailable
      console.warn(`Binary fetch failed for frame ${frameId}, falling back to JSON:`, binErr);
      const res = await fetch(`${this.baseUrl}/api/frames/${frameId}`);
      if (!res.ok) {
        throw new Error(`Failed to load frame ${frameId}: ${res.statusText}`);
      }
      const fetchMs = performance.now() - t0;
      const tParse = performance.now();
      const json = await res.json();

      const numCells = json.num_cells;
      const frame: FrameData = {
        frameId: json.frame_id,
        numCells,
        ringId: new Uint8Array(json.ring_id),
        row: new Int16Array(json.row),
        col: new Int16Array(json.col),
        zMean: new Float32Array(json.z_mean),
        semanticLabel: new Uint8Array(json.semantic_label),
        ratings: new Float32Array(json.ratings),
        dynamicMask: new Uint8Array(json.dynamic_mask),
        receivedAt: performance.now(),
        isBinary: false,
      };
      const parseMs = performance.now() - tParse;

      return { frame, fetchMs, parseMs };
    }
  }

  async getFrame(frameId: number): Promise<FrameFetchResult> {
    if (this.frameCache.has(frameId)) {
      const frame = this.frameCache.get(frameId)!;
      this.pruneCache(frameId);
      return {
        frame,
        fetchMs: 0.05, // instant cache hit
        parseMs: 0.01,
        cacheHit: true,
      };
    }

    if (this.inFlightFetches.has(frameId)) {
      const t0 = performance.now();
      const frame = await this.inFlightFetches.get(frameId)!;
      const fetchMs = performance.now() - t0;
      this.frameCache.set(frameId, frame);
      this.pruneCache(frameId);
      return {
        frame,
        fetchMs,
        parseMs: 0.02,
        cacheHit: false,
      };
    }

    const fetchPromise = this.fetchFrameFromNetwork(frameId);
    this.inFlightFetches.set(frameId, fetchPromise.then(r => r.frame));

    try {
      const result = await fetchPromise;
      this.frameCache.set(frameId, result.frame);
      this.pruneCache(frameId);
      return {
        frame: result.frame,
        fetchMs: result.fetchMs,
        parseMs: result.parseMs,
        cacheHit: false,
      };
    } finally {
      this.inFlightFetches.delete(frameId);
    }
  }

  prefetchFrame(frameId: number): void {
    if (this.frameCache.has(frameId) || this.inFlightFetches.has(frameId)) {
      return;
    }
    const fetchPromise = this.fetchFrameFromNetwork(frameId);
    this.inFlightFetches.set(frameId, fetchPromise.then(r => r.frame));
    fetchPromise
      .then(result => {
        this.frameCache.set(frameId, result.frame);
        this.pruneCache(frameId);
      })
      .catch(err => {
        console.debug(`Prefetch error for frame ${frameId}:`, err);
      })
      .finally(() => {
        this.inFlightFetches.delete(frameId);
      });
  }

  subscribeLiveFrames(callback: (frame: FrameData) => void): () => void {
    const wsUrl = this.baseUrl.replace(/^http/, 'ws') + '/ws/live';
    const ws = new WebSocket(wsUrl);
    ws.binaryType = 'arraybuffer';

    ws.onmessage = (event) => {
      if (typeof event.data === 'string') return;
      const buffer = event.data as ArrayBuffer;
      const view = new DataView(buffer);
      if (view.getUint32(0, true) !== BINARY_MAGIC) return;

      const frameId = view.getUint32(8, true);
      const numCells = view.getUint32(12, true);
      let offset = 32;
      const zMean = new Float32Array(buffer, offset, numCells);
      offset += numCells * 4;
      const ratings = new Float32Array(buffer, offset, numCells);
      offset += numCells * 4;
      const col = new Int16Array(buffer, offset, numCells);
      offset += numCells * 2;
      const row = new Int16Array(buffer, offset, numCells);
      offset += numCells * 2;
      const ringId = new Uint8Array(buffer, offset, numCells);
      offset += numCells;
      const semanticLabel = new Uint8Array(buffer, offset, numCells);
      offset += numCells;
      const dynamicMask = new Uint8Array(buffer, offset, numCells);

      const frame: FrameData = {
        frameId,
        numCells,
        ringId,
        row,
        col,
        zMean,
        semanticLabel,
        ratings,
        dynamicMask,
        receivedAt: performance.now(),
        isBinary: true,
      };
      callback(frame);
    };

    return () => {
      ws.close();
    };
  }
}
