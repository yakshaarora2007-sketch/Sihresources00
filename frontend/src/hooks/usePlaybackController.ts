import { useState, useRef, useEffect, useCallback } from 'react';
import { FrameProvider } from '../services/FrameProvider';
import { FrameData } from '../types/simulation';

interface UsePlaybackControllerOptions {
  frameProvider: FrameProvider;
  totalFrames: number;
  initialFps?: number;
  onFrameChange?: (frame: FrameData, meta: { fetchMs: number; parseMs: number; cacheHit: boolean }) => void;
}

export function usePlaybackController({
  frameProvider,
  totalFrames,
  initialFps = 10.0,
  onFrameChange,
}: UsePlaybackControllerOptions) {
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentFrameId, setCurrentFrameId] = useState(0);
  const [targetFps, setTargetFpsState] = useState(initialFps);

  const isPlayingRef = useRef(false);
  const currentFrameRef = useRef(0);
  const targetFpsRef = useRef(initialFps);
  const onFrameChangeRef = useRef(onFrameChange);
  const isLoadingFrameRef = useRef(false);
  const totalFramesRef = useRef(totalFrames);

  onFrameChangeRef.current = onFrameChange;
  totalFramesRef.current = totalFrames;

  const loadFrame = useCallback(
    async (frameId: number) => {
      if (frameId < 0 || frameId >= totalFramesRef.current) return;
      isLoadingFrameRef.current = true;
      try {
        const result = await frameProvider.getFrame(frameId);
        currentFrameRef.current = frameId;
        setCurrentFrameId(frameId);
        if (onFrameChangeRef.current) {
          onFrameChangeRef.current(result.frame, {
            fetchMs: result.fetchMs,
            parseMs: result.parseMs,
            cacheHit: result.cacheHit,
          });
        }

        // Prefetch subsequent frames asynchronously
        const prefetchCount = 3;
        for (let i = 1; i <= prefetchCount; i++) {
          const nextId = frameId + i;
          if (nextId < totalFramesRef.current) {
            frameProvider.prefetchFrame(nextId);
          }
        }
      } catch (err) {
        console.error(`Failed to load frame ${frameId}:`, err);
      } finally {
        isLoadingFrameRef.current = false;
      }
    },
    [frameProvider]
  );

  // Dedicated requestAnimationFrame playback loop
  useEffect(() => {
    let rafId: number;
    let lastTime = performance.now();
    let accumulator = 0;

    const loop = async (timestamp: number) => {
      const dt = timestamp - lastTime;
      lastTime = timestamp;

      if (isPlayingRef.current) {
        accumulator += dt;
        const frameInterval = 1000 / targetFpsRef.current;

        // Cap accumulator to avoid fast-forward explosion if tab was backgrounded
        if (accumulator > frameInterval * 2) {
          accumulator = frameInterval * 2;
        }

        if (accumulator >= frameInterval && !isLoadingFrameRef.current) {
          accumulator -= frameInterval;
          if (totalFramesRef.current > 0) {
            const nextFrame = (currentFrameRef.current + 1) % totalFramesRef.current;
            await loadFrame(nextFrame);
          }
        }
      } else {
        accumulator = 0;
      }

      rafId = requestAnimationFrame(loop);
    };

    rafId = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(rafId);
  }, [loadFrame]);

  const play = useCallback(() => {
    isPlayingRef.current = true;
    setIsPlaying(true);
  }, []);

  const pause = useCallback(() => {
    isPlayingRef.current = false;
    setIsPlaying(false);
  }, []);

  const togglePlay = useCallback(() => {
    if (isPlayingRef.current) {
      pause();
    } else {
      play();
    }
  }, [play, pause]);

  const restart = useCallback(() => {
    loadFrame(0);
  }, [loadFrame]);

  const stepForward = useCallback(() => {
    pause();
    const next = Math.min(totalFramesRef.current - 1, currentFrameRef.current + 1);
    loadFrame(next);
  }, [pause, loadFrame]);

  const stepBackward = useCallback(() => {
    pause();
    const prev = Math.max(0, currentFrameRef.current - 1);
    loadFrame(prev);
  }, [pause, loadFrame]);

  const seekTo = useCallback(
    (frameId: number) => {
      const target = Math.max(0, Math.min(totalFramesRef.current - 1, frameId));
      loadFrame(target);
    },
    [loadFrame]
  );

  const setTargetFps = useCallback((fps: number) => {
    const clamped = Math.max(1, Math.min(60, fps));
    targetFpsRef.current = clamped;
    setTargetFpsState(clamped);
  }, []);

  return {
    isPlaying,
    currentFrameId,
    targetFps,
    play,
    pause,
    togglePlay,
    restart,
    stepForward,
    stepBackward,
    seekTo,
    setTargetFps,
    loadFrame,
  };
}
