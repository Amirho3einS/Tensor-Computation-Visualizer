import React from 'react';
import type { LayerRuntime, ForwardBackwardResult } from '../../dnn/types';
import type { OperationStep, TensorSpec } from '../../types';
import TensorGrid from '../TensorGrid';
import TimelineControls from '../TimelineControls';
import { applyDimensionSizes } from '../../utils/einsumParser';
import { dimensionSizesForTensor, tensorSpecForName } from '../../dnn/stepGenerator';

function getTensorNames(
  rt: LayerRuntime,
  phase: 'forward' | 'backward',
  subPhase: 'dX' | 'dW' | 'db'
): string[] {
  const { spec } = rt;
  if (phase === 'forward') {
    if (spec.kind === 'conv1d') return ['X', 'W', 'bias', 'Y'];
    if (spec.kind === 'conv2d') return ['X', 'W', 'bias', 'Y'];
    if (spec.kind === 'linear') return ['X', 'W', 'bias', 'Y'];
    if (spec.kind === 'relu') return ['X', 'Y'];
    if (spec.kind === 'flatten') return ['X', 'Y'];
  }
  if (spec.kind === 'conv1d') {
    if (subPhase === 'dW') return ['dY', 'X', 'dW'];
    if (subPhase === 'dX') return ['dY', 'W', 'dX'];
    if (subPhase === 'db') return ['dY', 'db'];
  }
  if (spec.kind === 'conv2d') {
    if (subPhase === 'dW') return ['dY', 'X', 'dW'];
    if (subPhase === 'dX') return ['dY', 'W', 'dX'];
    if (subPhase === 'db') return ['dY', 'db'];
  }
  if (spec.kind === 'linear') {
    if (subPhase === 'dW') return ['dY', 'X', 'dW'];
    if (subPhase === 'dX') return ['dY', 'W', 'dX'];
    if (subPhase === 'db') return ['dY', 'db'];
  }
  if (spec.kind === 'relu') return ['dY', 'dX'];
  if (spec.kind === 'flatten') return ['dY', 'dX'];
  return [];
}

function getTensorData(
  name: string,
  layerIdx: number,
  fb: ForwardBackwardResult
): number[] | null | undefined {
  const L = layerIdx;
  switch (name) {
    case 'X':
      return fb.activations[L];
    case 'Y':
      return fb.activations[L + 1];
    case 'W':
      return fb.runtimes[L].weights;
    case 'bias':
      return fb.runtimes[L].bias;
    case 'dX':
      return fb.gradOutputs[L];
    case 'dY':
      return fb.gradOutputs[L + 1];
    case 'dW':
      return fb.gradWeights[L];
    case 'db':
      return fb.gradBiases[L];
    default:
      return null;
  }
}

export interface LayerViewProps {
  selectedLayerIdx: number;
  runtimes: LayerRuntime[];
  fb: ForwardBackwardResult;
  phase: 'forward' | 'backward';
  subPhase: 'dX' | 'dW' | 'db';
  steps: OperationStep[];
  currentStepIndex: number;
  onStepChange: (i: number) => void;
  isPlaying: boolean;
  onPlayPause: () => void;
  onReplay: () => void;
}

const LayerView: React.FC<LayerViewProps> = ({
  selectedLayerIdx,
  runtimes,
  fb,
  phase,
  subPhase,
  steps,
  currentStepIndex,
  onStepChange,
  isPlaying,
  onPlayPause,
  onReplay,
}) => {
  const currentStep = steps[currentStepIndex];

  if (selectedLayerIdx < 0) {
    const in0 = runtimes[0]?.inputShape ?? [1, 1, 8];
    if (in0.length === 4) {
      const [B, C, H, W] = in0;
      const sizes = { b: B, c: C, h: H, w: W };
      return (
        <div className="flex flex-col flex-1 min-h-0 p-4 gap-4">
          <div className="text-sm text-gray-400">
            Input tensor h0 and gradient dL/dh0 (select a layer card for step-by-step ops).
          </div>
          <div className="flex flex-wrap gap-6 justify-center items-start">
            <TensorGrid
              tensor={applyDimensionSizes({ name: 'input', indices: ['b', 'c', 'h', 'w'], shape: [] }, sizes)}
              sizes={sizes}
              highlight={null}
              label="h0 (input)"
              data={fb.activations[0]}
              showValues
            />
            <TensorGrid
              tensor={applyDimensionSizes({ name: 'dInput', indices: ['b', 'c', 'h', 'w'], shape: [] }, sizes)}
              sizes={sizes}
              highlight={null}
              label="dL/dh0"
              data={fb.gradOutputs[0]}
              showValues
            />
          </div>
          <TimelineControls
            steps={[]}
            currentStepIndex={0}
            isPlaying={false}
            onPlayPause={() => {}}
            onStepChange={() => {}}
            onReplay={onReplay}
          />
        </div>
      );
    }
    if (in0.length === 3) {
      const [B, C, L] = in0;
      const sizes = { b: B, c: C, l: L };
      const inputTensor: TensorSpec = { name: 'input', indices: ['b', 'c', 'l'], shape: [] };
      const gradIn = fb.gradOutputs[0];
      return (
        <div className="flex flex-col flex-1 min-h-0 p-4 gap-4">
          <div className="text-sm text-gray-400">
            Input tensor h₀ and gradient ∂L/∂h₀ (select a layer card for step-by-step ops).
          </div>
          <div className="flex flex-wrap gap-6 justify-center items-start">
            <TensorGrid
              tensor={applyDimensionSizes(inputTensor, sizes)}
              sizes={sizes}
              highlight={null}
              label="h₀ (input)"
              data={fb.activations[0]}
              showValues
            />
            <TensorGrid
              tensor={applyDimensionSizes({ name: 'dInput', indices: ['b', 'c', 'l'], shape: [] }, sizes)}
              sizes={sizes}
              highlight={null}
              label="∂L/∂h₀"
              data={gradIn}
              showValues
            />
          </div>
          <TimelineControls
            steps={[]}
            currentStepIndex={0}
            isPlaying={false}
            onPlayPause={() => {}}
            onStepChange={() => {}}
            onReplay={onReplay}
          />
        </div>
      );
    }
    const [B, Din] = in0;
    const sizes = { b: B, i: Din };
    return (
      <div className="flex flex-col flex-1 min-h-0 p-4 gap-4">
        <div className="text-sm text-gray-400">
          Input h₀ and ∂L/∂h₀ (2D). Select a layer for stepped ops.
        </div>
        <div className="flex flex-wrap gap-6 justify-center items-start">
          <TensorGrid
            tensor={applyDimensionSizes({ name: 'input', indices: ['b', 'i'], shape: [] }, sizes)}
            sizes={sizes}
            highlight={null}
            label="h₀ (input)"
            data={fb.activations[0]}
            showValues
          />
          <TensorGrid
            tensor={applyDimensionSizes({ name: 'dInput', indices: ['b', 'i'], shape: [] }, sizes)}
            sizes={sizes}
            highlight={null}
            label="∂L/∂h₀"
            data={fb.gradOutputs[0]}
            showValues
          />
        </div>
        <TimelineControls
          steps={[]}
          currentStepIndex={0}
          isPlaying={false}
          onPlayPause={() => {}}
          onStepChange={() => {}}
          onReplay={onReplay}
        />
      </div>
    );
  }

  const rt = runtimes[selectedLayerIdx];
  const names = getTensorNames(rt, phase, subPhase);

  const renderOne = (name: string) => {
    const meta = tensorSpecForName(name, rt);
    const sizes = dimensionSizesForTensor(name, rt, rt.spec);
    const data = getTensorData(name, selectedLayerIdx, fb);
    if (!meta || !sizes || !data) return null;
    const tensor: TensorSpec = { name: meta.name, indices: meta.indices, shape: [] };
    const isOutput =
      (phase === 'forward' && name === 'Y') ||
      (phase === 'backward' && (name === 'dW' || name === 'dX' || name === 'db'));
    const hl = currentStep?.highlights[name] ?? null;
    return (
      <TensorGrid
        key={name}
        tensor={applyDimensionSizes(tensor, sizes)}
        sizes={sizes}
        highlight={hl}
        label={name}
        data={isOutput ? null : data}
        computedData={isOutput ? data : null}
        expectedData={isOutput ? data : null}
        showValues
      />
    );
  };

  return (
    <div className="flex flex-col flex-1 min-h-0">
      <div className="flex-1 overflow-auto flex flex-wrap gap-4 p-4 justify-center items-start">
        {names.map(n => renderOne(n))}
      </div>
      <TimelineControls
        steps={steps}
        currentStepIndex={currentStepIndex}
        isPlaying={isPlaying}
        onPlayPause={onPlayPause}
        onStepChange={onStepChange}
        onReplay={onReplay}
      />
    </div>
  );
};

export default LayerView;
