import React from 'react';
import type { LayerRuntime } from '../../dnn/types';
import type { OperationStep } from '../../types';
import { ArrowLeft, ArrowRight } from 'lucide-react';

export interface NetworkDiagramProps {
  runtimes: LayerRuntime[];
  /** activations[0] = input, activations[i+1] = output of layer i */
  activationShapes: number[][];
  /** gradOutputs[i] = ∂L/∂ activations[i], same length as activationShapes */
  gradShapes: number[][];
  selectedLayerIdx: number;
  onSelectLayer: (idx: number) => void;
  currentStep: OperationStep | undefined;
}

function shapeBadge(shape: number[]): string {
  return shape.join('×');
}

const NetworkDiagram: React.FC<NetworkDiagramProps> = ({
  runtimes,
  activationShapes,
  gradShapes,
  selectedLayerIdx,
  onSelectLayer,
  currentStep,
}) => {
  const pulseIdx = currentStep?.dnnMeta?.layerIdx;

  return (
    <div className="network-graph px-4 py-4 bg-gray-900/40 border-b border-gray-800/50 overflow-x-auto">
      <div className="text-[10px] uppercase tracking-wide text-gray-500 font-semibold mb-3">Network graph</div>

      <div className="flex flex-col gap-6 min-w-max">
        {/* Forward */}
        <div className="flex items-center gap-1">
          <span className="text-[9px] uppercase text-sky-500/90 w-14 shrink-0">Forward</span>
          <div className="flex items-center gap-1 flex-1">
            <button
              type="button"
              onClick={() => onSelectLayer(-1)}
              className={`shrink-0 px-2 py-2 rounded-lg border text-left transition-all max-w-[100px] ${
                selectedLayerIdx === -1
                  ? 'border-sky-500 bg-sky-900/30 ring-1 ring-sky-500/50'
                  : 'border-gray-700 bg-gray-800/80 hover:border-gray-600'
              }`}
            >
              <div className="text-[10px] text-gray-500">Input</div>
              <div className="text-xs font-mono text-sky-200">{shapeBadge(activationShapes[0] ?? [])}</div>
            </button>
            {runtimes.map((rt, i) => (
              <React.Fragment key={i}>
                <ArrowRight size={16} className="text-sky-600/80 shrink-0" />
                <button
                  type="button"
                  onClick={() => onSelectLayer(i)}
                  className={`shrink-0 px-2 py-2 rounded-lg border text-left transition-all max-w-[120px] ${
                    selectedLayerIdx === i
                      ? 'border-sky-500 bg-sky-900/30 ring-1 ring-sky-500/50'
                      : 'border-gray-700 bg-gray-800/80 hover:border-gray-600'
                  } ${pulseIdx === i ? 'animate-pulse ring-2 ring-amber-500/40' : ''}`}
                >
                  <div className="text-[10px] text-gray-500 truncate">
                    {rt.spec.kind === 'conv1d' && `Conv1D k=${rt.spec.kernelSize}`}
                    {rt.spec.kind === 'conv2d' && `Conv2D ${rt.spec.kernelH}x${rt.spec.kernelW}`}
                    {rt.spec.kind === 'linear' && `Linear →${rt.spec.outFeatures}`}
                    {rt.spec.kind === 'relu' && 'ReLU'}
                    {rt.spec.kind === 'flatten' && 'Flatten'}
                  </div>
                  <div className="text-xs font-mono text-sky-200">{shapeBadge(activationShapes[i + 1] ?? rt.outputShape)}</div>
                </button>
              </React.Fragment>
            ))}
          </div>
        </div>

        {/* Gradients ∂L/∂h[i] — same shapes as activations[i], left-to-right */}
        <div className="flex items-center gap-1">
          <span className="text-[9px] uppercase text-amber-500/90 w-14 shrink-0">∂L/∂·</span>
          <div className="flex items-center gap-1 flex-1">
            <button
              type="button"
              onClick={() => onSelectLayer(-1)}
              className={`shrink-0 px-2 py-2 rounded-lg border text-left transition-all max-w-[110px] ${
                selectedLayerIdx === -1
                  ? 'border-amber-500 bg-amber-900/20 ring-1 ring-amber-500/50'
                  : 'border-gray-700 bg-gray-800/80 hover:border-gray-600'
              }`}
            >
              <div className="text-[10px] text-gray-500">∂L/∂h₀</div>
              <div className="text-xs font-mono text-amber-200/90">{shapeBadge(gradShapes[0] ?? [])}</div>
            </button>
            {runtimes.map((rt, i) => (
              <React.Fragment key={`g-${i}`}>
                <ArrowLeft size={16} className="text-amber-700/70 shrink-0" />
                <button
                  type="button"
                  onClick={() => onSelectLayer(i)}
                  className={`shrink-0 px-2 py-2 rounded-lg border text-left transition-all max-w-[120px] ${
                    selectedLayerIdx === i
                      ? 'border-amber-500 bg-amber-900/20 ring-1 ring-amber-500/50'
                      : 'border-gray-700 bg-gray-800/80 hover:border-gray-600'
                  } ${pulseIdx === i ? 'animate-pulse ring-2 ring-amber-500/40' : ''}`}
                >
                  <div className="text-[10px] text-gray-500 truncate">
                    ∂L/∂h{i + 1} · L{i + 1} {rt.spec.kind}
                  </div>
                  <div className="text-xs font-mono text-amber-200/90">{shapeBadge(gradShapes[i + 1] ?? [])}</div>
                </button>
              </React.Fragment>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};

export default NetworkDiagram;
