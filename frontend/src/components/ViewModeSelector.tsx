import React from 'react';
import { ColorMode } from '../types/simulation';
import { Layers, Mountain, ShieldAlert, CircleDot, Zap } from 'lucide-react';

interface ViewModeSelectorProps {
  colorMode: ColorMode;
  onChangeColorMode: (mode: ColorMode) => void;
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
  onChangeColorMode,
}) => {
  return (
    <div className="flex bg-slate-900/80 p-1 rounded-lg border border-slate-800 backdrop-blur-md gap-1">
      {MODES.map((mode) => {
        const isSelected = colorMode === mode.id;
        return (
          <button
            key={mode.id}
            onClick={() => onChangeColorMode(mode.id)}
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
    </div>
  );
};
