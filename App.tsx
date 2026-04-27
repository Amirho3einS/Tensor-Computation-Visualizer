import React, { useState, useEffect, useRef, useMemo, useCallback } from 'react';
import { ContractionSpec, DimensionSizes, OperationStep, VectorizationOption, TensorDataMap, TensorData, VerificationResult } from './types';
import TensorGrid from './components/TensorGrid';
import ViewPanel from './components/ViewPanel';
import TimelineControls from './components/TimelineControls';
import {
  parseEinsum,
  getVectorizationOptions,
  getAllIndices,
  getDefaultDimensionSizes,
  EINSUM_PRESETS,
  applyDimensionSizes
} from './utils/einsumParser';
import { generateSteps, getStrategyDescription } from './utils/stepGenerator';
import {
  initializeTensorData,
  simulateMACStep,
  verifyOutput,
  createZeroTensor,
  getTensorShape
} from './utils/tensorData';
import {
  Cpu,
  Zap,
  Radio,
  ChevronDown,
  Info,
  RefreshCw,
  CheckCircle,
  XCircle
} from 'lucide-react';

// Power of 2 options for sliders
const POWER_OF_2_OPTIONS = [1, 2, 4, 8, 16];

const App: React.FC = () => {
  // --- Core State ---
  const [einsumInput, setEinsumInput] = useState<string>('ik,kj->ij');
  const [contraction, setContraction] = useState<ContractionSpec | null>(null);
  const [dimensionSizes, setDimensionSizes] = useState<DimensionSizes>({});
  const [vectorWidth, setVectorWidth] = useState<number>(4);

  // --- Vectorization Selection ---
  const [selectedVecDim, setSelectedVecDim] = useState<string | null>(null);
  const [vectorizationOptions, setVectorizationOptions] = useState<VectorizationOption[]>([]);

  // --- Animation State ---
  const [steps, setSteps] = useState<OperationStep[]>([]);
  const [currentStepIndex, setCurrentStepIndex] = useState<number>(0);
  const [isPlaying, setIsPlaying] = useState<boolean>(false);

  // --- Tensor Data (for checking) ---
  const [inputData, setInputData] = useState<TensorDataMap>({});
  const [expectedOutput, setExpectedOutput] = useState<TensorData>([]);
  const [computedOutput, setComputedOutput] = useState<TensorData>([]);
  const [verification, setVerification] = useState<VerificationResult | null>(null);

  // --- UI State ---
  const [showPresets, setShowPresets] = useState<boolean>(false);
  const [parseError, setParseError] = useState<string | null>(null);

  const playIntervalRef = useRef<number | null>(null);
  const dropdownRef = useRef<HTMLDivElement>(null);

  // Close dropdown on outside click
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setShowPresets(false);
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  // --- Initialize tensor data ---
  const regenerateData = useCallback(() => {
    if (!contraction) return;

    const { inputData: newInputData, expectedOutput: newExpected } = initializeTensorData(contraction, dimensionSizes);
    setInputData(newInputData);
    setExpectedOutput(newExpected);

    // Reset computed output to zeros
    const outputShape = getTensorShape(contraction.output, dimensionSizes);
    setComputedOutput(createZeroTensor(outputShape));
    setVerification(null);
    setCurrentStepIndex(0);
    setIsPlaying(false);
  }, [contraction, dimensionSizes]);

  // --- Parse einsum on input change ---
  useEffect(() => {
    const parsed = parseEinsum(einsumInput);
    if (parsed) {
      setContraction(parsed);
      setParseError(null);

      // Get vectorization options
      const options = getVectorizationOptions(parsed);
      setVectorizationOptions(options);

      // Reset to defaults (powers of 2)
      const defaults = getDefaultDimensionSizes(parsed);
      Object.keys(defaults).forEach(k => {
        defaults[k] = 4; // Default to 4
      });
      setDimensionSizes(defaults);
      setSelectedVecDim(options.length > 0 ? options[0].dimension : null);
    } else {
      setContraction(null);
      setParseError('Invalid einsum expression');
      setVectorizationOptions([]);
      setSelectedVecDim(null);
    }

    // Reset animation
    setSteps([]);
    setCurrentStepIndex(0);
    setIsPlaying(false);
    setInputData({});
    setExpectedOutput([]);
    setComputedOutput([]);
    setVerification(null);
  }, [einsumInput]);

  // --- Generate steps and tensor data when params change ---
  useEffect(() => {
    if (!contraction || !selectedVecDim) {
      setSteps([]);
      return;
    }

    const option = vectorizationOptions.find(o => o.dimension === selectedVecDim);
    if (!option) {
      setSteps([]);
      return;
    }

    const newSteps = generateSteps(contraction, dimensionSizes, selectedVecDim, vectorWidth, option);
    setSteps(newSteps);

    // Initialize tensor data
    const { inputData: newInputData, expectedOutput: newExpected } = initializeTensorData(contraction, dimensionSizes);
    setInputData(newInputData);
    setExpectedOutput(newExpected);

    // Reset computed output to zeros
    const outputShape = getTensorShape(contraction.output, dimensionSizes);
    setComputedOutput(createZeroTensor(outputShape));
    setVerification(null);
    setCurrentStepIndex(0);
    setIsPlaying(false);
  }, [contraction, selectedVecDim, dimensionSizes, vectorWidth, vectorizationOptions]);

  // --- Simulate MAC operations as we step through ---
  useEffect(() => {
    if (!contraction || steps.length === 0 || Object.keys(inputData).length === 0) return;

    // Recompute from step 0 to currentStepIndex
    const outputShape = getTensorShape(contraction.output, dimensionSizes);
    let output = createZeroTensor(outputShape);

    for (let i = 0; i <= currentStepIndex; i++) {
      output = simulateMACStep(steps[i], contraction, dimensionSizes, inputData, output);
    }

    setComputedOutput(output);

    // Verify when at the end
    if (currentStepIndex === steps.length - 1) {
      const result = verifyOutput(output, expectedOutput);
      setVerification(result);
    } else {
      setVerification(null);
    }
  }, [currentStepIndex, steps, contraction, dimensionSizes, inputData, expectedOutput]);

  // --- Playback ---
  useEffect(() => {
    if (isPlaying && steps.length > 0) {
      playIntervalRef.current = window.setInterval(() => {
        setCurrentStepIndex(prev => {
          if (prev >= steps.length - 1) {
            setIsPlaying(false);
            return prev;
          }
          return prev + 1;
        });
      }, 300);
    } else {
      if (playIntervalRef.current) clearInterval(playIntervalRef.current);
    }
    return () => {
      if (playIntervalRef.current) clearInterval(playIntervalRef.current);
    };
  }, [isPlaying, steps.length]);

  // --- Derived values ---
  const currentStep = steps[currentStepIndex];
  const selectedOption = vectorizationOptions.find(o => o.dimension === selectedVecDim);

  // Build tensors with applied sizes
  const tensorsWithSizes = useMemo(() => {
    if (!contraction) return null;
    return {
      inputs: contraction.inputs.map(t => applyDimensionSizes(t, dimensionSizes)),
      output: applyDimensionSizes(contraction.output, dimensionSizes)
    };
  }, [contraction, dimensionSizes]);

  // Get all dimension indices for the slider panel
  const allIndices = contraction ? getAllIndices(contraction) : [];

  // --- Handlers ---
  const handlePresetSelect = (einsum: string) => {
    setEinsumInput(einsum);
    setShowPresets(false);
  };

  const handleDimSizeChange = (dim: string, size: number) => {
    setDimensionSizes(prev => ({ ...prev, [dim]: size }));
  };

  // Find nearest power of 2 index
  const getPowerOf2Index = (value: number) => {
    const idx = POWER_OF_2_OPTIONS.indexOf(value);
    return idx >= 0 ? idx : 2; // Default to 4 (index 2)
  };

  return (
    <div className="h-screen w-full flex flex-col bg-gray-950 text-white overflow-hidden font-sans">

      {/* Header */}
      <header className="flex flex-col sm:flex-row sm:items-center justify-between px-4 sm:px-6 py-3 bg-gray-900/80 backdrop-blur border-b border-gray-800 shrink-0 gap-3 relative z-50">

        {/* Title & Einsum Input */}
        <div className="flex items-center gap-3 flex-1 min-w-0">
          <div className="p-2 bg-gradient-to-br from-amber-500 to-orange-600 rounded-lg shadow-lg shadow-orange-900/30">
            <Cpu size={22} className="text-white" />
          </div>

          <div className="flex flex-col flex-1 min-w-0">
            <h1 className="text-base sm:text-lg font-bold tracking-tight text-white hidden sm:block">
              Tensor Contraction Visualizer
            </h1>

            {/* Einsum Input with Presets */}
            <div className="flex items-center gap-2 mt-1">
              <div className="relative flex-1 max-w-xs">
                <input
                  type="text"
                  value={einsumInput}
                  onChange={e => setEinsumInput(e.target.value)}
                  placeholder="e.g., ik,kj->ij"
                  className={`w-full px-3 py-1.5 bg-gray-800 border rounded-lg text-sm font-mono focus:outline-none focus:ring-2 transition-colors ${parseError
                    ? 'border-red-500/50 focus:ring-red-500/30'
                    : 'border-gray-700 focus:ring-amber-500/30 focus:border-amber-500/50'
                    }`}
                />
                {parseError && (
                  <div className="absolute -bottom-5 left-0 text-[10px] text-red-400">
                    {parseError}
                  </div>
                )}
              </div>

              {/* Presets Dropdown */}
              <div className="relative" ref={dropdownRef}>
                <button
                  onClick={() => setShowPresets(!showPresets)}
                  className="flex items-center gap-1 px-3 py-1.5 bg-gray-800 hover:bg-gray-700 border border-gray-700 rounded-lg text-xs text-gray-300 transition-colors"
                >
                  Presets <ChevronDown size={14} className={`transition-transform ${showPresets ? 'rotate-180' : ''}`} />
                </button>

                {showPresets && (
                  <div className="absolute top-full mt-2 left-0 z-[200] w-64 bg-gray-900 border border-gray-600 rounded-xl shadow-2xl overflow-hidden"
                    style={{ boxShadow: '0 25px 50px -12px rgba(0, 0, 0, 0.8)' }}>
                    <div className="text-[10px] uppercase text-gray-500 px-3 py-2 bg-gray-800 border-b border-gray-700 font-semibold tracking-wide">
                      Select Preset
                    </div>
                    {EINSUM_PRESETS.map(preset => (
                      <button
                        key={preset.einsum}
                        onClick={() => handlePresetSelect(preset.einsum)}
                        className="w-full px-3 py-2.5 text-left hover:bg-gray-800 transition-colors border-b border-gray-800 last:border-0"
                      >
                        <div className="flex justify-between items-center">
                          <span className="font-mono text-amber-400 text-sm">{preset.einsum}</span>
                          <span className="text-[10px] text-gray-500 uppercase bg-gray-800 px-1.5 py-0.5 rounded">{preset.label}</span>
                        </div>
                        <div className="text-[11px] text-gray-400 mt-0.5">{preset.description}</div>
                      </button>
                    ))}
                  </div>
                )}
              </div>

              {/* Regenerate Data Button */}
              <button
                onClick={regenerateData}
                className="flex items-center gap-1.5 px-3 py-1.5 bg-gray-800 hover:bg-gray-700 border border-gray-700 rounded-lg text-xs text-gray-300 transition-colors"
                title="Regenerate random tensor values"
              >
                <RefreshCw size={14} />
                <span className="hidden sm:inline">Refresh Data</span>
              </button>
            </div>
          </div>
        </div>

        {/* Verification Status */}
        {verification && (
          <div className={`flex items-center gap-2 px-4 py-2 rounded-lg ${verification.passed
            ? 'bg-green-900/50 border border-green-700/50'
            : 'bg-red-900/50 border border-red-700/50'
            }`}>
            {verification.passed ? (
              <CheckCircle size={20} className="text-green-400" />
            ) : (
              <XCircle size={20} className="text-red-400" />
            )}
            <div className="text-sm">
              <div className={`font-bold ${verification.passed ? 'text-green-400' : 'text-red-400'}`}>
                {verification.passed ? 'PASS' : 'FAIL'}
              </div>
              <div className="text-[10px] text-gray-400">
                {verification.passed
                  ? `All ${verification.totalElements} elements match`
                  : `${verification.mismatchCount}/${verification.totalElements} mismatch`
                }
              </div>
            </div>
          </div>
        )}
      </header>

      {/* Control Panel - Always Visible */}
      <div className="px-4 py-3 bg-gray-900/60 border-b border-gray-800/50 flex flex-wrap items-center gap-6">

        {/* Vector Width Slider */}
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-1.5 text-gray-400">
            <Zap size={14} />
            <span className="text-xs font-medium">w</span>
          </div>
          <div className="flex items-center gap-2">
            <input
              type="range"
              min="0"
              max="4"
              step="1"
              value={getPowerOf2Index(vectorWidth)}
              onChange={e => setVectorWidth(POWER_OF_2_OPTIONS[parseInt(e.target.value)])}
              className="w-24 h-1.5 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-amber-500"
            />
            <span className="text-xs font-mono font-bold text-amber-400 bg-amber-900/30 px-2 py-0.5 rounded min-w-[2rem] text-center">
              {vectorWidth}
            </span>
          </div>
        </div>

        {/* Divider */}
        <div className="h-6 w-px bg-gray-700" />

        {/* Dimension Size Sliders */}
        {allIndices.map(dim => (
          <div key={dim} className="flex items-center gap-3">
            <div className="flex items-center gap-1.5">
              <span className="text-xs font-mono font-bold text-gray-300">{dim}</span>
              <span className={`text-[9px] uppercase px-1 py-0.5 rounded ${contraction?.reductionIndices.includes(dim)
                ? 'text-red-400 bg-red-900/30'
                : 'text-green-400 bg-green-900/30'
                }`}>
                {contraction?.reductionIndices.includes(dim) ? 'red' : 'out'}
              </span>
            </div>
            <div className="flex items-center gap-2">
              <input
                type="range"
                min="0"
                max="4"
                step="1"
                value={getPowerOf2Index(dimensionSizes[dim] || 4)}
                onChange={e => handleDimSizeChange(dim, POWER_OF_2_OPTIONS[parseInt(e.target.value)])}
                className="w-16 h-1.5 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-sky-500"
              />
              <span className="text-xs font-mono font-bold text-sky-400 bg-sky-900/30 px-2 py-0.5 rounded min-w-[2rem] text-center">
                {dimensionSizes[dim] || 4}
              </span>
            </div>
          </div>
        ))}
      </div>

      {/* Vectorization Strategy Selector */}
      {contraction && vectorizationOptions.length > 0 && (
        <div className="px-4 py-2.5 bg-gray-900/40 border-b border-gray-800/50 flex flex-wrap items-center gap-3">
          <div className="flex items-center gap-2 text-gray-400">
            <Radio size={14} />
            <span className="text-xs font-medium uppercase tracking-wide">Vectorize Along:</span>
          </div>

          <div className="flex flex-wrap gap-2">
            {vectorizationOptions.map(option => (
              <button
                key={option.dimension}
                onClick={() => setSelectedVecDim(option.dimension)}
                className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-all flex items-center gap-2 ${selectedVecDim === option.dimension
                  ? 'bg-gradient-to-r from-amber-500 to-orange-600 text-white shadow-lg shadow-orange-500/20'
                  : 'bg-gray-800 text-gray-300 hover:bg-gray-700 border border-gray-700'
                  }`}
              >
                <span className="font-mono font-bold">{option.dimension}</span>
                {option.broadcastTensors.length > 0 ? (
                  <span className="text-[10px] opacity-80">
                    → broadcast {option.broadcastTensors.join(', ')}
                  </span>
                ) : (
                  <span className="text-[10px] opacity-80 text-green-300">
                    ✓ no broadcast
                  </span>
                )}
              </button>
            ))}
          </div>

          {/* Strategy description */}
          {selectedOption && (
            <div className="flex items-center gap-2 ml-auto text-[11px] text-gray-500">
              <Info size={12} />
              {getStrategyDescription(selectedOption, vectorWidth, contraction || undefined)}
            </div>
          )}
        </div>
      )}

      {/* Main Content - Tensor Visualization */}
      <main className="flex-1 relative overflow-auto flex flex-col">

        {!contraction && (
          <div className="flex-1 flex items-center justify-center">
            <div className="text-center text-gray-500">
              <Cpu size={48} className="mx-auto mb-4 opacity-30" />
              <p>Enter a valid einsum expression to visualize</p>
              <p className="text-xs mt-2 text-gray-600">Example: ik,kj{'->'} ij for matrix multiply</p>
            </div>
          </div>
        )}

        {contraction && selectedOption && (
          <ViewPanel
            contraction={contraction}
            dimensionSizes={dimensionSizes}
            selectedVecDim={selectedVecDim!}
            vectorizationOption={selectedOption}
            vectorWidth={vectorWidth}
            currentStep={currentStep}
            inputData={inputData}
            computedOutput={computedOutput}
            expectedOutput={expectedOutput}
          />
        )}

        {/* Legend */}
        {contraction && (
          <div className="absolute bottom-4 left-4 flex gap-3 text-[10px] bg-gray-900/80 backdrop-blur px-3 py-2 rounded-lg border border-gray-800">
            <div className="flex items-center gap-1.5">
              <div className="w-3 h-3 rounded bg-sky-500 shadow-sm" />
              <span className="text-gray-400">Vector</span>
            </div>
            <div className="flex items-center gap-1.5">
              <div className="w-3 h-3 rounded bg-amber-500 shadow-sm" />
              <span className="text-gray-400">Broadcast</span>
            </div>
            <div className="flex items-center gap-1.5">
              <div className="w-3 h-3 rounded bg-violet-500 shadow-sm" />
              <span className="text-gray-400">Output</span>
            </div>
            <div className="h-3 w-px bg-gray-700" />
            <div className="flex items-center gap-1.5">
              <div className="w-3 h-3 rounded bg-green-500 shadow-sm" />
              <span className="text-gray-400">Match</span>
            </div>
            <div className="flex items-center gap-1.5">
              <div className="w-3 h-3 rounded bg-red-500 shadow-sm" />
              <span className="text-gray-400">Mismatch</span>
            </div>
          </div>
        )}

      </main>

      {/* Footer Controls */}
      <TimelineControls
        steps={steps}
        currentStepIndex={currentStepIndex}
        isPlaying={isPlaying}
        onPlayPause={() => setIsPlaying(!isPlaying)}
        onStepChange={setCurrentStepIndex}
        onReplay={() => {
          setCurrentStepIndex(0);
          setIsPlaying(true);
        }}
      />

    </div>
  );
};

export default App;
