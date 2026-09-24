import React from 'react';
import { ColorMode, FrameData, SimulationMetadata } from '../types/simulation';
import { MiniLidarCanvas } from './MiniLidarCanvas';
import { Layers, Mountain, ShieldAlert, CircleDot, Zap } from 'lucide-react';

interface SidebarViewCardsProps {
  currentFrame: FrameData | null;
  metadata: SimulationMetadata | null;
  selectedColorMode: ColorMode;
  onSelectColorMode: (mode: ColorMode) => void;
}

interface ViewCardConfig {
  id: ColorMode;
  title: string;
  subtitle: string;
  icon: React.ReactNode;
}

const VIEW_CARDS: ViewCardConfig[] = [
  {
    id: 'semantic',
    title: 'Semantics',
    subtitle: 'Terrain • Drivable • Static • Object',
    icon: <Layers size={12} />,
  },
  {
    id: 'elevation',
    title: 'Elevation',
    subtitle: '2.5D Topographic Height Relief',
    icon: <Mountain size={12} />,
  },
  {
    id: 'rating',
    title: 'Traversability',
    subtitle: 'Safe (Green) • Caution • Hazard (Red)',
    icon: <ShieldAlert size={12} />,
  },
  {
    id: 'resolution',
    title: 'Resolution Rings',
    subtitle: 'Foveated Bands: 5cm to 50cm',
    icon: <CircleDot size={12} />,
  },
  {
    id: 'dynamic',
    title: 'Dynamic Highlight',
    subtitle: 'Neon Yellow: Dynamic Obstacles',
    icon: <Zap size={12} />,
  },
];

export const SidebarViewCards: React.FC<SidebarViewCardsProps> = ({
  currentFrame,
  metadata,
  selectedColorMode,
  onSelectColorMode,
}) => {
  return (
    <aside className="w-64 sm:w-72 xl:w-80 h-full bg-slate-900/95 border-r border-slate-800 flex flex-col shrink-0 overflow-hidden select-none backdrop-blur-md z-20">
      {/* Sidebar Compact Header */}
      <div className="px-4 py-2 border-b border-slate-800/80 bg-slate-900/50 shrink-0 flex items-center justify-between">
        <span className="text-xs font-bold text-slate-200 tracking-wider">
          LIVE PREVIEW (5x)
        </span>
        <span className="text-[10px] font-mono text-emerald-400 bg-emerald-950/80 border border-emerald-800/50 px-1.5 py-0.5 rounded">
          REAL-TIME
        </span>
      </div>

      {/* 5 Cards Container: Flex column dividing available height into 5 equal non-scrolling rows */}
      <div className="p-2 flex-1 min-h-0 flex flex-col gap-2 justify-between overflow-hidden">
        {VIEW_CARDS.map((card) => {
          const isSelected = selectedColorMode === card.id;

          return (
            <div
              key={card.id}
              onClick={() => onSelectColorMode(card.id)}
              className={`group flex-1 min-h-0 flex flex-col justify-between rounded-lg border transition-all duration-150 cursor-pointer p-2 overflow-hidden ${
                isSelected
                  ? 'bg-slate-800/95 border-emerald-500 shadow-[0_0_12px_rgba(16,185,129,0.3)] ring-1 ring-emerald-500/60'
                  : 'bg-slate-950/50 border-slate-800 hover:bg-slate-800/40 hover:border-slate-700'
              }`}
            >
              {/* Card Title Row */}
              <div className="flex items-center justify-between shrink-0 leading-none">
                <div className="flex items-center gap-1.5">
                  <span className={isSelected ? 'text-emerald-400' : 'text-slate-400 group-hover:text-slate-200'}>
                    {card.icon}
                  </span>
                  <span className={`text-xs font-semibold ${isSelected ? 'text-white' : 'text-slate-300'}`}>
                    {card.title}
                  </span>
                </div>

                {isSelected ? (
                    <span className="text-[9px] font-bold font-mono text-emerald-950 bg-emerald-400 px-1.5 py-0.5 rounded-full shadow">
                    ACTIVE
                  </span>
                ) : (
                  <span className="text-[9px] font-mono text-slate-500 group-hover:text-slate-400">
                    Click
                  </span>
                )}
              </div>

              {/* Live Canvas Viewport: Flex-1 fills available card height */}
              <div className="w-full flex-1 min-h-0 bg-slate-950 rounded overflow-hidden border border-slate-800/60 relative my-0.5">
                <MiniLidarCanvas
                  frame={currentFrame}
                  metadata={metadata}
                  colorMode={card.id}
                />
              </div>

              {/* Subtitle Caption */}
              <span className="text-[10px] text-slate-400 truncate leading-none">
                {card.subtitle}
              </span>
            </div>
          );
        })}
      </div>
    </aside>
  );
};
