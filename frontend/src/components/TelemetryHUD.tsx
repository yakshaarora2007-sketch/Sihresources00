import React, { useState } from 'react';
import { PerformanceMetrics } from '../types/simulation';
import { Activity, ChevronDown, ChevronUp } from 'lucide-react';

interface TelemetryHUDProps {
  metrics: PerformanceMetrics;
  numCells: number;
}

export const TelemetryHUD: React.FC<TelemetryHUDProps> = ({ metrics, numCells }) => {
  const [isExpanded, setIsExpanded] = useState(true);

  // Targets: data update < 20ms, render < 16-20ms, total < 100ms
  const isBudgetOk = metrics.totalFrameTimeMs <= 100;
  const isDataOk = (metrics.fetchTimeMs + metrics.parseTimeMs) <= 20;
  const isRenderOk = metrics.renderTimeMs <= 20;

  return (
    <div className="absolute top-5 left-5 z-20 flex flex-col font-mono select-none">
      <div className="bg-slate-900/90 backdrop-blur-md border border-slate-700/60 rounded-lg shadow-2xl overflow-hidden transition-all duration-200 min-w-80">
        {/* Header Bar */}
        <div
          onClick={() => setIsExpanded(!isExpanded)}
          className="flex items-center justify-between px-4 py-2.5 bg-slate-800/80 cursor-pointer hover:bg-slate-800 transition border-b border-slate-700/50"
        >
          <div className="flex items-center gap-2 text-sm font-semibold text-emerald-400">
            <Activity size={16} />
            <span>BROWSER TELEMETRY HUD</span>
          </div>
          <div className="flex items-center gap-2">
            <span
              className={`px-1.5 py-0.5 rounded text-[10px] font-bold ${
                isBudgetOk ? 'bg-emerald-950 text-emerald-300 border border-emerald-800' : 'bg-rose-950 text-rose-300 border border-rose-800'
              }`}
            >
              {metrics.totalFrameTimeMs.toFixed(1)} ms / 100 ms
            </span>
            {isExpanded ? <ChevronUp size={16} className="text-slate-400" /> : <ChevronDown size={16} className="text-slate-400" />}
          </div>
        </div>

        {/* Content Body */}
        {isExpanded && (
          <div className="p-4 text-sm space-y-3 text-slate-300">
            {/* FPS & Jitter */}
            <div className="grid grid-cols-2 gap-3 bg-slate-950/60 p-3 rounded border border-slate-800">
              <div>
                <span className="text-[11px] text-slate-400 block">ACTUAL FPS</span>
                <span className={`text-base font-bold ${metrics.actualFps >= 9.5 ? 'text-emerald-400' : 'text-amber-400'}`}>
                  {metrics.actualFps.toFixed(1)} <span className="text-[11px] text-slate-500 font-normal">FPS</span>
                </span>
              </div>
              <div>
                <span className="text-[11px] text-slate-400 block">AVG FPS (60f)</span>
                <span className="text-base font-bold text-slate-200">
                  {metrics.avgFps.toFixed(1)} <span className="text-[11px] text-slate-500 font-normal">FPS</span>
                </span>
              </div>
            </div>

            {/* Frame Latency Breakdown */}
            <div className="space-y-1.5">
              <span className="text-[11px] text-slate-400 tracking-wider">LATENCY BREAKDOWN</span>
              <div className="space-y-1.5 text-xs">
                <div className="flex justify-between items-center">
                  <span className="text-slate-400">Fetch / Cache Time:</span>
                  <span className={isDataOk ? 'text-emerald-400 font-semibold' : 'text-amber-400 font-semibold'}>
                    {metrics.fetchTimeMs.toFixed(1)} ms
                  </span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-slate-400">Decode / Parse Time:</span>
                  <span className="text-emerald-400 font-semibold">
                    {metrics.parseTimeMs.toFixed(2)} ms
                  </span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-slate-400">Canvas 2D Render Time:</span>
                  <span className={isRenderOk ? 'text-emerald-400 font-semibold' : 'text-amber-400 font-semibold'}>
                    {metrics.renderTimeMs.toFixed(1)} ms
                  </span>
                </div>
                <div className="flex justify-between items-center pt-1 border-t border-slate-800">
                  <span className="font-semibold text-slate-300">Total Frame Latency:</span>
                  <span className={`font-bold ${isBudgetOk ? 'text-emerald-400' : 'text-rose-400'}`}>
                    {metrics.totalFrameTimeMs.toFixed(1)} ms
                  </span>
                </div>
              </div>
            </div>

            {/* Budget Bar Visualization */}
            <div>
              <div className="flex justify-between text-[11px] text-slate-400 mb-1">
                <span>100ms Frame Budget Utilization</span>
                <span>{Math.min(100, Math.round((metrics.totalFrameTimeMs / 100) * 100))}%</span>
              </div>
              <div className="w-full h-2 bg-slate-800 rounded-full overflow-hidden">
                <div
                  className={`h-full transition-all duration-150 ${isBudgetOk ? 'bg-emerald-500' : 'bg-rose-500'}`}
                  style={{ width: `${Math.min(100, (metrics.totalFrameTimeMs / 100) * 100)}%` }}
                />
              </div>
            </div>

            {/* Cache & Cell Counters */}
            <div className="grid grid-cols-2 gap-3 text-[11px] text-slate-400 pt-2 border-t border-slate-800">
              <div>
                <span>Active Cells: </span>
                <span className="text-slate-200 font-semibold">{numCells.toLocaleString()}</span>
              </div>
              <div>
                <span>Max Frame Time: </span>
                <span className="text-slate-200 font-semibold">{metrics.maxFrameTimeMs.toFixed(0)} ms</span>
              </div>
              <div>
                <span>Cached Frames: </span>
                <span className="text-emerald-400 font-semibold">{metrics.cachedFramesCount} / 15</span>
              </div>
              <div>
                <span>Dropped Frames: </span>
                <span className={metrics.droppedFrames === 0 ? 'text-emerald-400 font-semibold' : 'text-rose-400 font-semibold'}>
                  {metrics.droppedFrames}
                </span>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
