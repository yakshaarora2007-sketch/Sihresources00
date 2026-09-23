import { useRef, useState, useCallback } from 'react';
import { PerformanceMetrics } from '../types/simulation';

export function usePerformanceMonitor() {
  const [metrics, setMetrics] = useState<PerformanceMetrics>({
    fetchTimeMs: 0,
    parseTimeMs: 0,
    updateTimeMs: 0,
    renderTimeMs: 0,
    totalFrameTimeMs: 0,
    actualFps: 10.0,
    avgFps: 10.0,
    maxFrameTimeMs: 0,
    droppedFrames: 0,
    cacheHits: 0,
    cacheMisses: 0,
    cachedFramesCount: 0,
  });

  const frameTimesRef = useRef<number[]>([]);
  const lastFrameTimestampRef = useRef<number>(performance.now());
  const maxTimeRef = useRef<number>(0);
  const droppedRef = useRef<number>(0);
  const hitsRef = useRef<number>(0);
  const missesRef = useRef<number>(0);
  const lastMetricUpdateRef = useRef<number>(performance.now());

  const recordFrame = useCallback(
    (params: {
      fetchMs: number;
      parseMs: number;
      updateMs: number;
      renderMs: number;
      cacheHit: boolean;
      cachedFramesCount: number;
    }) => {
      const now = performance.now();
      const deltaSinceLastFrame = now - lastFrameTimestampRef.current;
      lastFrameTimestampRef.current = now;

      const totalFrameTime = params.fetchMs + params.parseMs + params.updateMs + params.renderMs;
      if (totalFrameTime > maxTimeRef.current) {
        maxTimeRef.current = totalFrameTime;
      }

      // Check for dropped frame if budget is 100ms and delta is >150ms during continuous playback
      if (deltaSinceLastFrame > 150 && frameTimesRef.current.length > 5) {
        droppedRef.current += Math.floor(deltaSinceLastFrame / 100) - 1;
      }

      if (params.cacheHit) {
        hitsRef.current++;
      } else {
        missesRef.current++;
      }

      const instantaneousFps = deltaSinceLastFrame > 0 ? Math.min(60, 1000 / deltaSinceLastFrame) : 10;
      frameTimesRef.current.push(instantaneousFps);
      if (frameTimesRef.current.length > 60) {
        frameTimesRef.current.shift();
      }

      const avgFps =
        frameTimesRef.current.reduce((sum, v) => sum + v, 0) / (frameTimesRef.current.length || 1);

      const currentSnapshot = {
        fetchTimeMs: Math.round(params.fetchMs * 10) / 10,
        parseTimeMs: Math.round(params.parseMs * 100) / 100,
        updateTimeMs: Math.round(params.updateMs * 10) / 10,
        renderTimeMs: Math.round(params.renderMs * 10) / 10,
        totalFrameTimeMs: Math.round(totalFrameTime * 10) / 10,
        actualFps: Math.round(instantaneousFps * 10) / 10,
        avgFps: Math.round(avgFps * 10) / 10,
        maxFrameTimeMs: Math.round(maxTimeRef.current * 10) / 10,
        droppedFrames: droppedRef.current,
        cacheHits: hitsRef.current,
        cacheMisses: missesRef.current,
        cachedFramesCount: params.cachedFramesCount,
      };

      if (typeof window !== 'undefined') {
        (window as any).__LIDAR_TELEMETRY__ = currentSnapshot;
      }

      // Throttle React state updates to ~5 Hz to avoid layout thrashing
      if (now - lastMetricUpdateRef.current > 180) {
        lastMetricUpdateRef.current = now;
        setMetrics(currentSnapshot);
      }
    },
    []
  );

  const resetMetrics = useCallback(() => {
    frameTimesRef.current = [];
    maxTimeRef.current = 0;
    droppedRef.current = 0;
    hitsRef.current = 0;
    missesRef.current = 0;
    lastFrameTimestampRef.current = performance.now();
  }, []);

  return { metrics, recordFrame, resetMetrics };
}
