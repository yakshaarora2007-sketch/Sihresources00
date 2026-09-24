import React from 'react';
import { ColorMode, AppView } from '../types/simulation';
import { Layers, Mountain, ShieldAlert, CircleDot, Zap, ActivitySquare } from 'lucide-react';

interface ViewModeSelectorProps {
  colorMode: ColorMode;
  appView: AppView;
  onChangeColorMode: (mode: ColorMode) => void;
  onChangeAppView: (view: AppView) => void;
}

const MODES: { id: ColorMode; label: string; icon: React.ReactNode; desc: string }[] = [
  {
    id: 'semantic',
    label: 'Semantics',
    icon: <Layers size={14} />,
    desc: 'Terrain, Drivable, Static, and Dynamic Object classes',
  },
  {
    id: 'elevation',
    label: 'Elevation',
    icon: <Mountain size={14} />,
    desc: 'Topographic 2.5D z_mean height relief',
  },
  {
    id: 'rating',
    label: 'Traversability',
    icon: <ShieldAlert size={14} />,
    desc: 'Cost-to-go / traversability safety score (0-100)',
  },
  {
    id: 'resolution',
    label: 'Resolution Rings',
    icon: <CircleDot size={14} />,
    desc: 'Foveated ring distance bands (5cm, 10cm, 25cm, 50cm)',
  },
  {
    id: 'dynamic',
    label: 'Dynamic Highlight',
    icon: <Zap size={14} />,
    desc: 'Bayesian temporal fusion dynamic motion mask',
  },
];

export const ViewModeSelector: React.FC<ViewModeSelectorProps> = ({
  colorMode,
  appView,
  onChangeColorMode,
  onChangeAppView,
}) => {
  return (
    <div className="flex bg-slate-900/80 p-1 rounded-lg border border-slate-800 backdrop-blur-md gap-1">
      {MODES.map((mode) => {
        const isSelected = appView === 'canvas' && colorMode === mode.id;
        return (
          <button
            key={mode.id}
            onClick={() => {
              onChangeAppView('canvas');
              onChangeColorMode(mode.id);
            }}
            title={mode.desc}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium transition ${
              isSelected
                ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/40 shadow-sm'
                : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/60 border border-transparent'
            }`}
          >
            {mode.icon}
            <span>{mode.label}</span>
          </button>
        );
      })}
      
      <div className="w-px h-6 bg-slate-800 mx-1 self-center" />

      <button
        onClick={() => onChangeAppView('performance')}
        title="View Performance & Analytics Dashboard"
        className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-bold transition ${
          appView === 'performance'
            ? 'bg-rose-500/20 text-rose-400 border border-rose-500/50 shadow-sm'
            : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/60 border border-transparent'
        }`}
      >
        <ActivitySquare size={14} />
        <span>Performance</span>
      </button>
    </div>
  );
};
