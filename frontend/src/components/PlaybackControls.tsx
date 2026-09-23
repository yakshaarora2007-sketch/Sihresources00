import React from 'react';
import { Play, Pause, RotateCcw, ChevronLeft, ChevronRight } from 'lucide-react';

interface PlaybackControlsProps {
  isPlaying: boolean;
  currentFrameId: number;
  totalFrames: number;
  targetFps: number;
  isCached: (frameId: number) => boolean;
  onTogglePlay: () => void;
  onRestart: () => void;
  onStepForward: () => void;
  onStepBackward: () => void;
  onSeek: (frameId: number) => void;
  onSetTargetFps: (fps: number) => void;
}

const SPEED_OPTIONS = [5, 10, 15, 20, 30];

export const PlaybackControls: React.FC<PlaybackControlsProps> = ({
  isPlaying,
  currentFrameId,
  totalFrames,
  targetFps,
  onTogglePlay,
  onRestart,
  onStepForward,
  onStepBackward,
  onSeek,
  onSetTargetFps,
}) => {
  const maxFrame = Math.max(0, totalFrames - 1);
  const progressPct = maxFrame > 0 ? (currentFrameId / maxFrame) * 100 : 0;

  return (
    <div className="bg-slate-900/90 border-t border-slate-800 px-6 py-3 flex flex-col gap-2.5 backdrop-blur-lg select-none">
      {/* Frame Scrubber Bar */}
      <div className="flex items-center gap-4">
        <span className="text-xs font-mono text-emerald-400 font-semibold min-w-24">
          Frame {String(currentFrameId).padStart(3, '0')} / {String(maxFrame).padStart(3, '0')}
        </span>

        <div className="relative flex-1 flex items-center group">
          <input
            type="range"
            min={0}
            max={maxFrame}
            value={currentFrameId}
            onChange={(e) => onSeek(parseInt(e.target.value, 10))}
            className="w-full h-2 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-emerald-400 focus:outline-none transition-all"
            style={{
              background: `linear-gradient(to right, #10b981 0%, #10b981 ${progressPct}%, #1e293b ${progressPct}%, #1e293b 100%)`
            }}
          />
        </div>

        <span className="text-xs font-mono text-slate-400 min-w-16 text-right">
          {((currentFrameId / 10.0)).toFixed(1)}s
        </span>
      </div>

      {/* Main Buttons Bar */}
      <div className="flex items-center justify-between">
        {/* Left: Quick Actions */}
        <div className="flex items-center gap-2">
          <button
            onClick={onRestart}
            title="Restart from Frame 0 [R]"
            className="p-2 text-slate-400 hover:text-white hover:bg-slate-800 rounded-lg transition"
          >
            <RotateCcw size={17} />
          </button>
          <button
            onClick={onStepBackward}
            title="Step Back 1 Frame [Left Arrow]"
            className="p-2 text-slate-400 hover:text-white hover:bg-slate-800 rounded-lg transition"
          >
            <ChevronLeft size={20} />
          </button>
          <button
            onClick={onTogglePlay}
            title="Play / Pause [Space]"
            className={`flex items-center justify-center w-10 h-10 rounded-full font-bold transition shadow-lg ${
              isPlaying
                ? 'bg-amber-500 hover:bg-amber-400 text-slate-950'
                : 'bg-emerald-500 hover:bg-emerald-400 text-slate-950'
            }`}
          >
            {isPlaying ? <Pause size={18} /> : <Play size={18} className="translate-x-0.5" />}
          </button>
          <button
            onClick={onStepForward}
            title="Step Forward 1 Frame [Right Arrow]"
            className="p-2 text-slate-400 hover:text-white hover:bg-slate-800 rounded-lg transition"
          >
            <ChevronRight size={20} />
          </button>
        </div>

        {/* Center: Playback Status */}
        <div className="hidden md:flex items-center gap-3 text-xs text-slate-400">
          <span className="flex items-center gap-1.5">
            <span className={`w-2 h-2 rounded-full ${isPlaying ? 'bg-emerald-400 animate-pulse' : 'bg-slate-600'}`} />
            {isPlaying ? 'PLAYING (Client Timing Loop)' : 'PAUSED'}
          </span>
          <span className="text-slate-600">|</span>
          <span>100 ms Frame Budget</span>
        </div>

        {/* Right: Target FPS / Speed Selector */}
        <div className="flex items-center gap-2">
          <span className="text-xs text-slate-400 mr-1">Target Speed:</span>
          <div className="flex bg-slate-800 rounded-lg p-0.5 border border-slate-700/60">
            {SPEED_OPTIONS.map((fps) => (
              <button
                key={fps}
                onClick={() => onSetTargetFps(fps)}
                className={`px-2.5 py-1 text-xs font-mono font-medium rounded-md transition ${
                  targetFps === fps
                    ? 'bg-emerald-500 text-slate-950 font-bold shadow'
                    : 'text-slate-400 hover:text-white hover:bg-slate-700/50'
                }`}
              >
                {fps} FPS
              </button>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};
