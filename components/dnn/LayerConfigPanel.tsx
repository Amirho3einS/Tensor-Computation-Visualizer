import React from 'react';
import type { LayerSpec, NetworkSpec } from '../../dnn/types';
import { RefreshCw, ChevronDown, ChevronUp } from 'lucide-react';

const POWER_OF_2_OPTIONS = [1, 2, 4, 8, 16];
const KERNEL_OPTIONS = [3, 5];

function pow2Index(value: number): number {
  const idx = POWER_OF_2_OPTIONS.indexOf(value);
  return idx >= 0 ? idx : 2;
}

function kernelIndex(k: number): number {
  const idx = KERNEL_OPTIONS.indexOf(k);
  return idx >= 0 ? idx : 0;
}

export interface LayerConfigPanelProps {
  spec: NetworkSpec;
  onChange: (next: NetworkSpec) => void;
  validationError: string | null;
  onRefreshData: () => void;
  collapsed: boolean;
  onToggleCollapsed: () => void;
}

const defaultLayerForKind = (kind: LayerSpec['kind']): LayerSpec => {
  switch (kind) {
    case 'conv1d':
      return { kind: 'conv1d', outChannels: 4, kernelSize: 3 };
    case 'conv2d':
      return { kind: 'conv2d', outChannels: 4, kernelH: 3, kernelW: 3 };
    case 'linear':
      return { kind: 'linear', outFeatures: 4 };
    case 'relu':
      return { kind: 'relu' };
    case 'flatten':
      return { kind: 'flatten' };
  }
};

const LayerConfigPanel: React.FC<LayerConfigPanelProps> = ({
  spec,
  onChange,
  validationError,
  onRefreshData,
  collapsed,
  onToggleCollapsed,
}) => {
  const setInput = (key: 'B' | 'C' | 'L' | 'H' | 'W', v: number) => {
    if (spec.inputShape.mode === '1d') {
      if (key === 'H' || key === 'W') return;
      onChange({ ...spec, inputShape: { ...spec.inputShape, [key]: v } });
      return;
    }
    if (key === 'L') return;
    onChange({ ...spec, inputShape: { ...spec.inputShape, [key]: v } });
  };

  const setInputMode = (mode: '1d' | '2d') => {
    if (mode === spec.inputShape.mode) return;
    if (mode === '1d') {
      onChange({
        ...spec,
        inputShape: { mode: '1d', B: spec.inputShape.B, C: spec.inputShape.C, L: 8 },
        layers: [
          { kind: 'conv1d', outChannels: 4, kernelSize: 3 },
          { kind: 'relu' },
          { kind: 'conv1d', outChannels: 8, kernelSize: 3 },
          { kind: 'relu' },
          { kind: 'flatten' },
          { kind: 'linear', outFeatures: 4 },
        ],
      });
      return;
    }
    onChange({
      ...spec,
      inputShape: { mode: '2d', B: spec.inputShape.B, C: spec.inputShape.C, H: 8, W: 8 },
      layers: [
        { kind: 'conv2d', outChannels: 4, kernelH: 3, kernelW: 3 },
        { kind: 'relu' },
        { kind: 'conv2d', outChannels: 8, kernelH: 3, kernelW: 3 },
        { kind: 'relu' },
        { kind: 'flatten' },
        { kind: 'linear', outFeatures: 4 },
      ],
    });
  };

  const updateLayer = (idx: number, next: LayerSpec) => {
    const layers = [...spec.layers];
    layers[idx] = next;
    onChange({ ...spec, layers });
  };

  const removeLayer = (idx: number) => {
    const layers = spec.layers.filter((_, i) => i !== idx);
    onChange({ ...spec, layers });
  };

  const addLayer = (kind: LayerSpec['kind']) => {
    onChange({ ...spec, layers: [...spec.layers, defaultLayerForKind(kind)] });
  };

  return (
    <div className="px-4 py-3 bg-gray-900/60 border-b border-gray-800/50 flex flex-col gap-3">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-2">
          <div className="text-xs uppercase tracking-wide text-gray-500 font-semibold">Network config</div>
          <button
            type="button"
            onClick={onToggleCollapsed}
            className="flex items-center gap-1.5 px-2 py-1 bg-gray-800 hover:bg-gray-700 border border-gray-700 rounded-lg text-[11px] text-gray-300 transition-colors"
            title={collapsed ? 'Show config' : 'Hide config'}
          >
            {collapsed ? <ChevronDown size={14} /> : <ChevronUp size={14} />}
            {collapsed ? 'Show' : 'Hide'}
          </button>
        </div>
        <button
          type="button"
          onClick={onRefreshData}
          className="flex items-center gap-1.5 px-3 py-1.5 bg-gray-800 hover:bg-gray-700 border border-gray-700 rounded-lg text-xs text-gray-300 transition-colors"
        >
          <RefreshCw size={14} />
          Refresh data
        </button>
      </div>

      {validationError && (
        <div className="text-xs text-red-400 bg-red-900/20 border border-red-800/50 rounded-lg px-3 py-2">
          {validationError}
        </div>
      )}

      {collapsed && (
        <div className="text-xs text-gray-300 bg-gray-800/60 border border-gray-700/60 rounded-lg px-3 py-2">
          <span className="text-gray-400">Input:</span>{' '}
          <span className="font-mono">
            {spec.inputShape.mode === '1d'
              ? `1D B=${spec.inputShape.B}, C=${spec.inputShape.C}, L=${spec.inputShape.L}`
              : `2D B=${spec.inputShape.B}, C=${spec.inputShape.C}, H=${spec.inputShape.H}, W=${spec.inputShape.W}`}
          </span>
          <span className="text-gray-500">  |  </span>
          <span className="text-gray-400">Layers:</span>{' '}
          <span className="font-mono">{spec.layers.length}</span>
          <span className="text-gray-500">  |  </span>
          <span className="font-mono">
            {spec.layers
              .map(layer => {
                if (layer.kind === 'conv1d') return `conv(k${layer.kernelSize},c${layer.outChannels})`;
                if (layer.kind === 'conv2d') return `conv2d(${layer.kernelH}x${layer.kernelW},c${layer.outChannels})`;
                if (layer.kind === 'linear') return `linear(${layer.outFeatures})`;
                return layer.kind;
              })
              .join(' -> ')}
          </span>
        </div>
      )}

      {!collapsed && (
        <>

      <div className="flex flex-wrap gap-3 items-center">
        <span className="text-xs text-gray-400">Input mode</span>
        <div className="flex rounded-lg overflow-hidden border border-gray-700">
          <button
            type="button"
            onClick={() => setInputMode('1d')}
            className={`px-3 py-1 text-xs ${spec.inputShape.mode === '1d' ? 'bg-sky-600 text-white' : 'bg-gray-800 text-gray-400'}`}
          >
            1D
          </button>
          <button
            type="button"
            onClick={() => setInputMode('2d')}
            className={`px-3 py-1 text-xs ${spec.inputShape.mode === '2d' ? 'bg-sky-600 text-white' : 'bg-gray-800 text-gray-400'}`}
          >
            2D
          </button>
        </div>
      </div>

      <div className="flex flex-wrap gap-6 items-end">
        {(['B', 'C'] as const).map(key => (
          <div key={key} className="flex items-center gap-2">
            <span className="text-xs font-mono text-gray-400">{key}</span>
            <input
              type="range"
              min={0}
              max={4}
              step={1}
              value={pow2Index(spec.inputShape[key])}
              onChange={e => setInput(key, POWER_OF_2_OPTIONS[parseInt(e.target.value, 10)])}
              className="w-20 h-1.5 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-sky-500"
            />
            <span className="text-xs font-mono text-sky-400 tabular-nums w-6">{spec.inputShape[key]}</span>
          </div>
        ))}
        {spec.inputShape.mode === '1d' && (
          <div className="flex items-center gap-2">
            <span className="text-xs font-mono text-gray-400">L</span>
            <input
              type="range"
              min={0}
              max={4}
              step={1}
              value={pow2Index(spec.inputShape.L)}
              onChange={e => setInput('L', POWER_OF_2_OPTIONS[parseInt(e.target.value, 10)])}
              className="w-20 h-1.5 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-sky-500"
            />
            <span className="text-xs font-mono text-sky-400 tabular-nums w-6">{spec.inputShape.L}</span>
          </div>
        )}
        {spec.inputShape.mode === '2d' && (
          <>
            {(['H', 'W'] as const).map(key => (
              <div key={key} className="flex items-center gap-2">
                <span className="text-xs font-mono text-gray-400">{key}</span>
                <input
                  type="range"
                  min={0}
                  max={4}
                  step={1}
                  value={pow2Index(spec.inputShape[key])}
                  onChange={e => setInput(key, POWER_OF_2_OPTIONS[parseInt(e.target.value, 10)])}
                  className="w-20 h-1.5 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-sky-500"
                />
                <span className="text-xs font-mono text-sky-400 tabular-nums w-6">{spec.inputShape[key]}</span>
              </div>
            ))}
          </>
        )}
      </div>

      <div className="flex flex-col gap-3">
        <div className="text-[10px] uppercase text-gray-500 font-semibold">Layers</div>
        {spec.layers.map((layer, idx) => (
          <div
            key={idx}
            className="flex flex-wrap items-center gap-3 p-2 rounded-lg bg-gray-800/50 border border-gray-700/50"
          >
            <span className="text-[10px] text-gray-500 w-6">{idx + 1}</span>
            {layer.kind === 'conv1d' && (
              <>
                <span className="text-xs text-amber-300 font-medium">Conv1D</span>
                <label className="flex items-center gap-1 text-xs text-gray-400">
                  C_out
                  <input
                    type="range"
                    min={0}
                    max={POWER_OF_2_OPTIONS.length - 1}
                    value={pow2Index(layer.outChannels)}
                    onChange={e =>
                      updateLayer(idx, {
                        ...layer,
                        outChannels: POWER_OF_2_OPTIONS[parseInt(e.target.value, 10)],
                      })
                    }
                    className="w-16 h-1.5 bg-gray-700 rounded-lg accent-amber-500"
                  />
                  <span className="font-mono text-amber-400 w-6">{layer.outChannels}</span>
                </label>
                <label className="flex items-center gap-1 text-xs text-gray-400">
                  K
                  <input
                    type="range"
                    min={0}
                    max={1}
                    value={kernelIndex(layer.kernelSize)}
                    onChange={e =>
                      updateLayer(idx, {
                        ...layer,
                        kernelSize: KERNEL_OPTIONS[parseInt(e.target.value, 10)],
                      })
                    }
                    className="w-12 h-1.5 bg-gray-700 rounded-lg accent-amber-500"
                  />
                  <span className="font-mono text-amber-400 w-6">{layer.kernelSize}</span>
                </label>
              </>
            )}
            {layer.kind === 'conv2d' && (
              <>
                <span className="text-xs text-amber-300 font-medium">Conv2D</span>
                <label className="flex items-center gap-1 text-xs text-gray-400">
                  C_out
                  <input
                    type="range"
                    min={0}
                    max={POWER_OF_2_OPTIONS.length - 1}
                    value={pow2Index(layer.outChannels)}
                    onChange={e =>
                      updateLayer(idx, {
                        ...layer,
                        outChannels: POWER_OF_2_OPTIONS[parseInt(e.target.value, 10)],
                      })
                    }
                    className="w-16 h-1.5 bg-gray-700 rounded-lg accent-amber-500"
                  />
                  <span className="font-mono text-amber-400 w-6">{layer.outChannels}</span>
                </label>
                <label className="flex items-center gap-1 text-xs text-gray-400">
                  Kh
                  <input
                    type="range"
                    min={0}
                    max={1}
                    value={kernelIndex(layer.kernelH)}
                    onChange={e =>
                      updateLayer(idx, {
                        ...layer,
                        kernelH: KERNEL_OPTIONS[parseInt(e.target.value, 10)],
                      })
                    }
                    className="w-12 h-1.5 bg-gray-700 rounded-lg accent-amber-500"
                  />
                  <span className="font-mono text-amber-400 w-6">{layer.kernelH}</span>
                </label>
                <label className="flex items-center gap-1 text-xs text-gray-400">
                  Kw
                  <input
                    type="range"
                    min={0}
                    max={1}
                    value={kernelIndex(layer.kernelW)}
                    onChange={e =>
                      updateLayer(idx, {
                        ...layer,
                        kernelW: KERNEL_OPTIONS[parseInt(e.target.value, 10)],
                      })
                    }
                    className="w-12 h-1.5 bg-gray-700 rounded-lg accent-amber-500"
                  />
                  <span className="font-mono text-amber-400 w-6">{layer.kernelW}</span>
                </label>
              </>
            )}
            {layer.kind === 'linear' && (
              <>
                <span className="text-xs text-violet-300 font-medium">Linear</span>
                <label className="flex items-center gap-1 text-xs text-gray-400">
                  out
                  <input
                    type="range"
                    min={0}
                    max={POWER_OF_2_OPTIONS.length - 1}
                    value={pow2Index(layer.outFeatures)}
                    onChange={e =>
                      updateLayer(idx, {
                        ...layer,
                        outFeatures: POWER_OF_2_OPTIONS[parseInt(e.target.value, 10)],
                      })
                    }
                    className="w-16 h-1.5 bg-gray-700 rounded-lg accent-violet-500"
                  />
                  <span className="font-mono text-violet-300 w-6">{layer.outFeatures}</span>
                </label>
              </>
            )}
            {layer.kind === 'relu' && <span className="text-xs text-green-300 font-medium">ReLU</span>}
            {layer.kind === 'flatten' && <span className="text-xs text-cyan-300 font-medium">Flatten</span>}
            <button
              type="button"
              onClick={() => removeLayer(idx)}
              className="ml-auto text-xs text-red-400 hover:text-red-300 px-2 py-1 rounded border border-red-900/50"
            >
              Remove
            </button>
          </div>
        ))}
      </div>

      <div className="flex flex-wrap gap-2 items-center">
        <span className="text-[10px] text-gray-500 uppercase">Add</span>
        {(['conv1d', 'conv2d', 'relu', 'flatten', 'linear'] as const).map(kind => (
          <button
            key={kind}
            type="button"
            onClick={() => addLayer(kind)}
            className="px-2 py-1 text-xs rounded border border-gray-600 bg-gray-800 hover:bg-gray-700 text-gray-300"
          >
            + {kind}
          </button>
        ))}
      </div>
        </>
      )}
    </div>
  );
};

export default LayerConfigPanel;
