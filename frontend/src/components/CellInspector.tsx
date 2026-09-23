import React from 'react';
import { HoveredCellInfo } from '../types/simulation';
import { Crosshair } from 'lucide-react';

interface CellInspectorProps {
  info: HoveredCellInfo | null;
}

export const CellInspector: React.FC<CellInspectorProps> = ({ info }) => {
  if (!info) {
    return (
      <div className="bg-slate-900/80 border border-slate-800 rounded-lg p-2.5 text-xs text-slate-500 flex items-center gap-2">
        <Crosshair size={14} className="text-slate-600" />
        <span>Hover cursor over grid to inspect cell attributes</span>
      </div>
    );
  }

  const getBadgeColor = (sem: string) => {
    switch (sem) {
      case 'DRIVABLE': return 'bg-emerald-950 text-emerald-300 border-emerald-800';
      case 'OBJECT': return 'bg-rose-950 text-rose-300 border-rose-800';
      case 'STATIC': return 'bg-slate-800 text-slate-300 border-slate-700';
      default: return 'bg-blue-950 text-blue-300 border-blue-800';
    }
  };

  return (
    <div className="bg-slate-900/90 border border-slate-700/60 rounded-lg p-3 text-xs font-mono text-slate-300 shadow-xl backdrop-blur-md flex items-center gap-6">
      <div className="flex items-center gap-2">
        <Crosshair size={15} className="text-emerald-400" />
        <div>
          <span className="text-[10px] text-slate-400 block">COORDINATE</span>
          <span className="font-bold text-slate-200">
            X: {info.x.toFixed(2)}m, Y: {info.y.toFixed(2)}m
          </span>
        </div>
      </div>

      <div className="h-6 w-px bg-slate-800" />

      <div>
        <span className="text-[10px] text-slate-400 block">GRID INDEX</span>
        <span className="text-slate-200">
          [{info.row}, {info.col}] (Ring {info.ringId}: {(info.resolution * 100).toFixed(0)}cm)
        </span>
      </div>

      <div className="h-6 w-px bg-slate-800" />

      <div>
        <span className="text-[10px] text-slate-400 block">ELEVATION</span>
        <span className="text-slate-200 font-semibold">{info.zMean.toFixed(2)}m</span>
      </div>

      <div className="h-6 w-px bg-slate-800" />

      <div>
        <span className="text-[10px] text-slate-400 block">SEMANTIC CLASS</span>
        <span className={`px-1.5 py-0.5 rounded text-[10px] font-bold border ${getBadgeColor(info.semanticClass)}`}>
          {info.semanticClass}
        </span>
      </div>

      <div className="h-6 w-px bg-slate-800" />

      <div>
        <span className="text-[10px] text-slate-400 block">TRAVERSABILITY</span>
        <span className={`font-semibold ${info.rating >= 70 ? 'text-emerald-400' : info.rating >= 35 ? 'text-amber-400' : 'text-rose-400'}`}>
          {info.rating.toFixed(1)} / 100 ({info.rating >= 70 ? 'SAFE' : info.rating >= 35 ? 'CAUTION' : 'HAZARD'})
        </span>
      </div>

      <div className="h-6 w-px bg-slate-800" />

      <div>
        <span className="text-[10px] text-slate-400 block">STATE</span>
        <span className={info.dynamic ? 'text-yellow-300 font-bold bg-yellow-950/70 border border-yellow-500/50 px-1.5 py-0.5 rounded' : 'text-slate-400'}>
          {info.dynamic ? '⚡ DYNAMIC' : 'STATIC'}
        </span>
      </div>
    </div>
  );
};
