import React, { useState, useMemo, useEffect, useCallback } from 'react';
import type { NetworkSpec } from '../../dnn/types';
import {
  validateNetwork,
  buildRuntimes,
  forwardBackward,
  randomInput,
} from '../../dnn/compute';
import { generateLayerSteps } from '../../dnn/stepGenerator';
import LayerConfigPanel from './LayerConfigPanel';
import NetworkDiagram from './NetworkDiagram';
import LayerView from './LayerView';
import { Cpu, Network } from 'lucide-react';

const DEFAULT_SPEC: NetworkSpec = {
  inputShape: { mode: '1d', B: 1, C: 1, L: 8 },
  layers: [
    { kind: 'conv1d', outChannels: 4, kernelSize: 3 },
    { kind: 'relu' },
    { kind: 'conv1d', outChannels: 8, kernelSize: 3 },
    { kind: 'relu' },
    { kind: 'flatten' },
    { kind: 'linear', outFeatures: 4 },
  ],
};

const DNNApp: React.FC = () => {
  const [spec, setSpec] = useState<NetworkSpec>(DEFAULT_SPEC);
  const [refreshKey, setRefreshKey] = useState(0);
  const [selectedLayerIdx, setSelectedLayerIdx] = useState(0);
  const [phase, setPhase] = useState<'forward' | 'backward'>('forward');
  const [subPhase, setSubPhase] = useState<'dX' | 'dW' | 'db'>('dX');
  const [currentStepIndex, setCurrentStepIndex] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [configCollapsed, setConfigCollapsed] = useState(false);

  const validationError = useMemo(() => validateNetwork(spec), [spec]);

  const runtimes = useMemo(() => {
    if (validationError) return [];
    try {
      return buildRuntimes(spec);
    } catch {
      return [];
    }
  }, [spec, validationError, refreshKey]);

  const fb = useMemo(() => {
    if (runtimes.length === 0) return null;
    const x = randomInput(spec);
    return forwardBackward(runtimes, x);
  }, [runtimes, spec, refreshKey]);

  const activationShapes = useMemo(() => {
    if (runtimes.length === 0) return [];
    const shapes: number[][] = [runtimes[0].inputShape];
    for (const rt of runtimes) shapes.push(rt.outputShape);
    return shapes;
  }, [runtimes]);

  const stepBundle = useMemo(() => {
    if (selectedLayerIdx < 0 || selectedLayerIdx >= runtimes.length) return null;
    return generateLayerSteps(selectedLayerIdx, runtimes[selectedLayerIdx]);
  }, [selectedLayerIdx, runtimes]);

  const steps = useMemo(() => {
    if (!stepBundle) return [];
    if (phase === 'forward') return stepBundle.forward;
    const b = stepBundle.backward;
    if (subPhase === 'dW' && b.dW?.length) return b.dW;
    if (subPhase === 'db' && b.db?.length) return b.db;
    return b.dX;
  }, [phase, subPhase, stepBundle]);

  const currentStep = steps[currentStepIndex];

  useEffect(() => {
    if (steps.length > 0 && currentStepIndex >= steps.length) {
      setCurrentStepIndex(Math.max(0, steps.length - 1));
    }
  }, [steps.length, currentStepIndex]);

  useEffect(() => {
    setCurrentStepIndex(0);
    setIsPlaying(false);
  }, [selectedLayerIdx, phase, subPhase, stepBundle]);

  useEffect(() => {
    if (phase !== 'backward') return;
    const b = stepBundle?.backward;
    if (subPhase === 'dW' && (!b?.dW || b.dW.length === 0)) setSubPhase('dX');
    if (subPhase === 'db' && (!b?.db || b.db.length === 0)) setSubPhase('dX');
  }, [phase, subPhase, stepBundle]);

  useEffect(() => {
    if (isPlaying && steps.length > 0) {
      const id = window.setInterval(() => {
        setCurrentStepIndex(prev => {
          if (prev >= steps.length - 1) {
            setIsPlaying(false);
            return prev;
          }
          return prev + 1;
        });
      }, 300);
      return () => clearInterval(id);
    }
  }, [isPlaying, steps.length]);

  useEffect(() => {
    if (runtimes.length === 0) {
      setSelectedLayerIdx(-1);
      return;
    }
    if (selectedLayerIdx >= runtimes.length) {
      setSelectedLayerIdx(runtimes.length - 1);
    }
  }, [runtimes.length, selectedLayerIdx]);

  const onRefreshData = useCallback(() => {
    setRefreshKey(k => k + 1);
    setCurrentStepIndex(0);
    setIsPlaying(false);
  }, []);

  const showDW = !!(stepBundle?.backward.dW && stepBundle.backward.dW.length > 0);
  const showDB = !!(stepBundle?.backward.db && stepBundle.backward.db.length > 0);

  return (
    <div className="h-full min-h-0 w-full flex flex-col bg-gray-950 text-white overflow-hidden font-sans">
      <header className="flex items-center gap-3 px-4 py-3 bg-gray-900/80 border-b border-gray-800 shrink-0">
        <div className="p-2 bg-gradient-to-br from-violet-500 to-purple-600 rounded-lg shadow-lg">
          <Network size={22} className="text-white" />
        </div>
        <div>
          <h1 className="text-base sm:text-lg font-bold tracking-tight">DNN forward / backward</h1>
          <p className="text-[11px] text-gray-500 hidden sm:block">
            Conv1D/Conv2D · ReLU · Flatten · Linear — tunable shapes, per-layer timeline (MSE target on output).
          </p>
        </div>
      </header>

      <LayerConfigPanel
        spec={spec}
        onChange={setSpec}
        validationError={validationError}
        onRefreshData={onRefreshData}
        collapsed={configCollapsed}
        onToggleCollapsed={() => setConfigCollapsed(prev => !prev)}
      />

      {!fb || validationError ? (
        <div className="flex-1 flex items-center justify-center text-gray-500 p-8">
          <Cpu size={40} className="opacity-30 mr-3" />
          {validationError || 'Adjust layers to a valid network.'}
        </div>
      ) : (
        <>
          <NetworkDiagram
            runtimes={runtimes}
            activationShapes={activationShapes}
            gradShapes={activationShapes}
            selectedLayerIdx={selectedLayerIdx}
            onSelectLayer={setSelectedLayerIdx}
            currentStep={currentStep}
          />

          <div className="flex flex-wrap items-center gap-2 px-4 py-2 bg-gray-900/50 border-b border-gray-800/50 shrink-0">
            <span className="text-[10px] uppercase text-gray-500 font-semibold">Layer detail</span>
            <div className="flex rounded-lg overflow-hidden border border-gray-700">
              <button
                type="button"
                onClick={() => setPhase('forward')}
                className={`px-3 py-1 text-xs font-medium ${
                  phase === 'forward'
                    ? 'bg-sky-600 text-white'
                    : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
                }`}
              >
                Forward
              </button>
              <button
                type="button"
                onClick={() => setPhase('backward')}
                className={`px-3 py-1 text-xs font-medium ${
                  phase === 'backward'
                    ? 'bg-amber-600 text-white'
                    : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
                }`}
              >
                Backward
              </button>
            </div>
            {phase === 'backward' && (
              <div className="flex rounded-lg overflow-hidden border border-gray-700">
                <button
                  type="button"
                  onClick={() => setSubPhase('dX')}
                  className={`px-3 py-1 text-xs font-medium ${
                    subPhase === 'dX' ? 'bg-amber-700 text-white' : 'bg-gray-800 text-gray-400'
                  }`}
                >
                  ∂L/∂X
                </button>
                {showDW && (
                  <button
                    type="button"
                    onClick={() => setSubPhase('dW')}
                    className={`px-3 py-1 text-xs font-medium ${
                      subPhase === 'dW' ? 'bg-amber-700 text-white' : 'bg-gray-800 text-gray-400'
                    }`}
                  >
                    ∂L/∂W
                  </button>
                )}
                {showDB && (
                  <button
                    type="button"
                    onClick={() => setSubPhase('db')}
                    className={`px-3 py-1 text-xs font-medium ${
                      subPhase === 'db' ? 'bg-amber-700 text-white' : 'bg-gray-800 text-gray-400'
                    }`}
                  >
                    ∂L/∂b
                  </button>
                )}
              </div>
            )}
          </div>

          <main className="flex-1 flex flex-col min-h-0 overflow-hidden">
            <LayerView
              selectedLayerIdx={selectedLayerIdx}
              runtimes={runtimes}
              fb={fb}
              phase={phase}
              subPhase={subPhase}
              steps={steps}
              currentStepIndex={currentStepIndex}
              onStepChange={setCurrentStepIndex}
              isPlaying={isPlaying}
              onPlayPause={() => setIsPlaying(p => !p)}
              onReplay={() => {
                setCurrentStepIndex(0);
                setIsPlaying(true);
              }}
            />
          </main>
        </>
      )}
    </div>
  );
};

export default DNNApp;
