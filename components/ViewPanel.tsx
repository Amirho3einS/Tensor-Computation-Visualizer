import React, { useState, useMemo } from 'react';
import { ContractionSpec, DimensionSizes, OperationStep, VectorizationOption, TensorDataMap, TensorData, TensorSpec } from '../types';
import { analyzeHardwareLayouts, TensorLayout, LayoutPermutation } from '../utils/hardwareLayout';
import TensorGrid from './TensorGrid';
import { applyDimensionSizes } from '../utils/einsumParser';
import { Layers, Cpu, ArrowRight, Shuffle, Lock } from 'lucide-react';

interface ViewPanelProps {
    contraction: ContractionSpec;
    dimensionSizes: DimensionSizes;
    selectedVecDim: string;
    vectorizationOption: VectorizationOption;
    vectorWidth: number;
    currentStep: OperationStep | undefined;
    inputData: TensorDataMap;
    computedOutput: TensorData;
    expectedOutput: TensorData;
}

type ViewMode = 'logical' | 'hardware';

/**
 * Transpose a flat tensor data array according to index reordering
 * oldIndices: original index order e.g. ['i', 'k']
 * newIndices: new index order e.g. ['k', 'i']
 * sizes: dimension sizes
 */
function transposeTensorData(
    data: TensorData,
    oldIndices: string[],
    newIndices: string[],
    sizes: DimensionSizes
): TensorData {
    if (!data || data.length === 0) return data;
    if (JSON.stringify(oldIndices) === JSON.stringify(newIndices)) return data;

    const oldShape = oldIndices.map(idx => sizes[idx] || 4);
    const newShape = newIndices.map(idx => sizes[idx] || 4);
    const totalSize = oldShape.reduce((a, b) => a * b, 1);

    // Create mapping from old index positions to new index positions
    const permutation = newIndices.map(idx => oldIndices.indexOf(idx));

    const result = new Array(totalSize).fill(0);

    // For each element in the old tensor
    for (let flatIdx = 0; flatIdx < totalSize; flatIdx++) {
        // Convert flat index to multi-dimensional indices in old order
        const oldMultiIdx: number[] = [];
        let remaining = flatIdx;
        for (let d = oldShape.length - 1; d >= 0; d--) {
            oldMultiIdx[d] = remaining % oldShape[d];
            remaining = Math.floor(remaining / oldShape[d]);
        }

        // Permute to new order
        const newMultiIdx = permutation.map(p => oldMultiIdx[p]);

        // Convert new multi-dimensional indices to flat index
        let newFlatIdx = 0;
        let multiplier = 1;
        for (let d = newShape.length - 1; d >= 0; d--) {
            newFlatIdx += newMultiIdx[d] * multiplier;
            multiplier *= newShape[d];
        }

        result[newFlatIdx] = data[flatIdx];
    }

    return result;
}

/**
 * Get PyTorch-style permutation string
 * e.g., [i,k] -> [k,i] becomes ".permute(1,0)"
 */
function getPermutationString(oldIndices: string[], newIndices: string[]): string {
    if (JSON.stringify(oldIndices) === JSON.stringify(newIndices)) {
        return '';
    }

    // For each position in the new indices, find where it came from in old indices
    const permutation = newIndices.map(idx => oldIndices.indexOf(idx));
    return `.permute(${permutation.join(',')})`;
}

/**
 * Generate scalar loop nest string (original computation)
 */
function generateScalarLoopNest(contraction: ContractionSpec, sizes: DimensionSizes): string {
    const { inputs, output, reductionIndices } = contraction;
    const hasReduction = reductionIndices.length > 0;
    const isUnary = inputs.length === 1;

    // All indices: output indices first, then reduction indices
    const allIndices = [...output.indices, ...reductionIndices];

    const lines: string[] = [];
    allIndices.forEach((idx, depth) => {
        const indent = '  '.repeat(depth);
        const size = sizes[idx] || idx.toUpperCase();
        lines.push(`${indent}for ${idx} in range(${size}):`);
    });

    const finalIndent = '  '.repeat(allIndices.length);
    const outputRef = `${output.name}[${output.indices.join(',')}]`;

    if (isUnary) {
        const inputRef = `${inputs[0].name}[${inputs[0].indices.join(',')}]`;
        lines.push(`${finalIndent}${outputRef} = f(${inputRef})`);
    } else if (hasReduction) {
        const inputRefs = inputs.map(t => `${t.name}[${t.indices.join(',')}]`).join(' * ');
        lines.push(`${finalIndent}${outputRef} += ${inputRefs}`);
    } else {
        const inputRefs = inputs.map(t => `${t.name}[${t.indices.join(',')}]`).join(' \u2295 ');
        lines.push(`${finalIndent}${outputRef} = ${inputRefs}`);
    }

    return lines.join('\n');
}

/**
 * Generate vectorized loop nest string using the selected layout permutation
 */
function generateVectorizedLoopNest(
    contraction: ContractionSpec,
    vectorizationOption: VectorizationOption,
    layoutPerm: LayoutPermutation | null,
    sizes: DimensionSizes,
    vectorWidth: number
): string {
    const { inputs, output } = contraction;
    const vecDim = vectorizationOption.dimension;
    const vecDimSize = sizes[vecDim] || 4;

    // Use the loop order from the layout permutation
    const loopIndices = layoutPerm?.loopOrder || [];

    const lines: string[] = [];
    loopIndices.forEach((idx, depth) => {
        const indent = '  '.repeat(depth);
        const size = sizes[idx] || idx.toUpperCase();
        lines.push(`${indent}for ${idx} in range(${size}):`);
    });

    const numVectorOps = Math.ceil(vecDimSize / vectorWidth);
    let currentDepth = loopIndices.length;

    if (numVectorOps > 1) {
        const innerIndent = '  '.repeat(currentDepth);
        lines.push(`${innerIndent}for ${vecDim}_chunk in range(${numVectorOps}):  # ${vecDimSize}/${vectorWidth} chunks`);
        currentDepth++;
    }

    const formatTensorRef = (name: string, indices: string[], isBroadcast: boolean): string => {
        const layout = layoutPerm?.tensorLayouts[name];
        const displayIndices = layout?.needsTranspose ? layout.transposedIndices : indices;
        const displayName = layout?.needsTranspose ? `${name}ᵀ` : name;

        const formattedIndices = displayIndices.map(idx => {
            if (idx === vecDim) {
                if (isBroadcast) return idx;
                return numVectorOps > 1 ? `${vecDim}_chunk*w:(${vecDim}_chunk+1)*w` : ':';
            }
            return idx;
        });

        return `${displayName}[${formattedIndices.join(',')}]`;
    };

    const hasReduction = contraction.reductionIndices.length > 0;
    const isUnary = inputs.length === 1;

    const finalIndent = '  '.repeat(currentDepth);
    const outputRef = formatTensorRef(output.name, output.indices, false);

    let opComment: string;
    let opLine: string;

    if (isUnary) {
        const inputRef = formatTensorRef(inputs[0].name, inputs[0].indices, false);
        opComment = `${finalIndent}# Unary (w=${vectorWidth} elements)`;
        opLine = `${finalIndent}${outputRef} = f(${inputRef})`;
    } else if (hasReduction) {
        const inputRefs = inputs.map(t => {
            const isBroadcast = vectorizationOption.broadcastTensors.includes(t.name);
            return formatTensorRef(t.name, t.indices, isBroadcast);
        }).join(' * ');
        opComment = `${finalIndent}# Vector MAC (w=${vectorWidth} elements)`;
        opLine = `${finalIndent}${outputRef} += ${inputRefs}`;
    } else {
        const inputRefs = inputs.map(t => {
            const isBroadcast = vectorizationOption.broadcastTensors.includes(t.name);
            return formatTensorRef(t.name, t.indices, isBroadcast);
        }).join(' \u2295 ');
        opComment = `${finalIndent}# Element-wise (w=${vectorWidth} elements)`;
        opLine = `${finalIndent}${outputRef} = ${inputRefs}`;
    }

    lines.push(opComment);
    lines.push(opLine);

    const outputLayout = layoutPerm?.tensorLayouts[output.name];
    if (outputLayout?.needsTranspose) {
        const finalOutput = `${output.name}[${output.indices.map(idx => idx === vecDim ? ':' : idx).join(',')}]`;
        lines.push('');
        lines.push(`# Transpose back: ${outputRef} → ${finalOutput}`);
    }

    return lines.join('\n');
}

/**
 * Panel showing both logical and hardware views of tensor operations
 */
const ViewPanel: React.FC<ViewPanelProps> = ({
    contraction,
    dimensionSizes,
    selectedVecDim,
    vectorizationOption,
    vectorWidth,
    currentStep,
    inputData,
    computedOutput,
    expectedOutput
}) => {
    const [viewMode, setViewMode] = useState<ViewMode>('logical');
    const [selectedPermIdx, setSelectedPermIdx] = useState<number>(0);

    // Analyze hardware layouts for current vectorization
    const hardwareConfig = useMemo(() => {
        const config = analyzeHardwareLayouts(contraction, vectorizationOption, dimensionSizes);
        setSelectedPermIdx(0); // Reset to first (loop-optimal) permutation
        return config;
    }, [contraction, vectorizationOption, dimensionSizes]);

    // Get the currently selected layout permutation
    const selectedLayoutPerm = useMemo(() => {
        return hardwareConfig.layoutPermutations[selectedPermIdx] || hardwareConfig.layoutPermutations[0];
    }, [hardwareConfig, selectedPermIdx]);

    // Check if any transposes are needed
    const transposeInfo = useMemo(() => {
        if (!selectedLayoutPerm) return { needsAny: false, tensors: [] as string[] };
        const layouts = Object.entries(selectedLayoutPerm.tensorLayouts);
        const transposedTensors = layouts
            .filter(([_, layout]) => (layout as TensorLayout).needsTranspose)
            .map(([name, _]) => name);
        return {
            needsAny: transposedTensors.length > 0,
            tensors: transposedTensors
        };
    }, [selectedLayoutPerm]);

    // Get tensor layout info
    const getTensorLayout = (tensorName: string): TensorLayout | null => {
        if (!selectedLayoutPerm) return null;
        return selectedLayoutPerm.tensorLayouts[tensorName] as TensorLayout || null;
    };

    // Create transposed tensor spec
    const getTransposedTensor = (originalTensor: TensorSpec): TensorSpec => {
        const layout = getTensorLayout(originalTensor.name);
        if (!layout || !layout.needsTranspose) {
            return originalTensor;
        }

        return {
            name: originalTensor.name,
            indices: layout.transposedIndices,
            shape: layout.transposedIndices.map(idx => dimensionSizes[idx] || 4)
        };
    };

    // Get transposed data for a tensor
    const getTransposedData = (tensorName: string, originalData: TensorData | null | undefined): TensorData | null => {
        if (!originalData) return null;
        const layout = getTensorLayout(tensorName);
        if (!layout || !layout.needsTranspose) return originalData;

        return transposeTensorData(
            originalData,
            layout.originalIndices,
            layout.transposedIndices,
            dimensionSizes
        );
    };

    // Check if a tensor needs transpose
    const needsTranspose = (tensorName: string): boolean => {
        return transposeInfo.tensors.includes(tensorName);
    };

    // Render a tensor
    const renderTensor = (
        tensor: TensorSpec,
        label: string,
        isOutput: boolean,
        data: TensorData | null,
        computedData: TensorData | null,
        expectedData: TensorData | null,
        highlight: OperationStep['highlights'][string] | null
    ) => {
        return (
            <TensorGrid
                tensor={applyDimensionSizes(tensor, dimensionSizes)}
                sizes={dimensionSizes}
                highlight={highlight}
                label={label}
                data={isOutput ? null : data}
                computedData={isOutput ? computedData : null}
                expectedData={isOutput ? expectedData : null}
                showValues={true}
            />
        );
    };

    // Render layout permutation selector
    const renderLayoutSelector = () => {
        const { layoutPermutations, isUniquelyDetermined, freeOutputDims, totalPermutations } = hardwareConfig;

        return (
            <div className="flex flex-col gap-3 px-4 py-3 bg-gray-900/60 border-b border-gray-800/50">
                {/* Header with degree of freedom info */}
                <div className="flex items-center gap-3 flex-wrap">
                    <div className="flex items-center gap-2">
                        {isUniquelyDetermined ? (
                            <Lock size={14} className="text-green-400" />
                        ) : (
                            <Shuffle size={14} className="text-amber-400" />
                        )}
                        <span className="text-xs font-medium text-gray-300">
                            Data Layout{isUniquelyDetermined ? '' : 's'}
                        </span>
                    </div>

                    {isUniquelyDetermined ? (
                        <span className="text-[11px] text-green-400 bg-green-900/30 px-2.5 py-1 rounded-md border border-green-700/50">
                            Uniquely determined by v={selectedVecDim}
                        </span>
                    ) : (
                        <span className="text-[11px] text-amber-400 bg-amber-900/30 px-2.5 py-1 rounded-md border border-amber-700/50">
                            {totalPermutations} valid layout{totalPermutations > 1 ? 's' : ''} — free dims: [{freeOutputDims.join(', ')}]
                        </span>
                    )}
                </div>

                {/* Fixed dimension info */}
                <div className="flex flex-wrap gap-2">
                    {Object.entries(selectedLayoutPerm.tensorLayouts).map(([name, layout]) => {
                        const tLayout = layout as TensorLayout;
                        return (
                            <div key={name} className="flex items-center gap-1.5 text-[10px] bg-gray-800/80 px-2 py-1 rounded border border-gray-700/50">
                                <span className="font-mono font-bold text-gray-300">{name}</span>
                                <span className="text-gray-500">→</span>
                                <span className="font-mono text-gray-400">[</span>
                                {tLayout.transposedIndices.map((idx, i) => {
                                    const isFixed = idx === tLayout.fixedDim;
                                    const isFree = tLayout.freeDims.includes(idx);
                                    return (
                                        <span key={i}>
                                            <span className={`font-mono font-bold ${isFixed
                                                ? 'text-green-400'
                                                : isFree
                                                    ? 'text-amber-400'
                                                    : 'text-gray-300'
                                                }`}>
                                                {idx}
                                            </span>
                                            {i < tLayout.transposedIndices.length - 1 && (
                                                <span className="text-gray-500">,</span>
                                            )}
                                        </span>
                                    );
                                })}
                                <span className="font-mono text-gray-400">]</span>
                                <span className={`text-[9px] px-1 rounded ${tLayout.accessType === 'vector'
                                    ? 'text-sky-400 bg-sky-900/30'
                                    : 'text-amber-400 bg-amber-900/30'
                                    }`}>
                                    {tLayout.accessType}
                                </span>
                            </div>
                        );
                    })}
                </div>

                {/* Layout permutation buttons (only if >1) */}
                {!isUniquelyDetermined && (
                    <div className="flex flex-wrap gap-2">
                        {layoutPermutations.map((perm, idx) => (
                            <button
                                key={perm.id}
                                onClick={() => setSelectedPermIdx(idx)}
                                className={`flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs font-medium transition-all border ${selectedPermIdx === idx
                                    ? 'bg-gradient-to-r from-violet-600 to-purple-600 text-white border-violet-500 shadow-lg shadow-violet-500/20'
                                    : 'bg-gray-800 text-gray-300 hover:bg-gray-700 border-gray-700'
                                    }`}
                            >
                                <span className="font-mono text-[11px]">{perm.description}</span>
                                {perm.isNaturalOrder && (
                                    <span className="text-[9px] px-1.5 py-0.5 rounded bg-blue-900/50 text-blue-400 border border-blue-700/50">
                                        natural
                                    </span>
                                )}
                            </button>
                        ))}
                    </div>
                )}

                {/* Legend for colors */}
                <div className="flex items-center gap-4 text-[10px] text-gray-500">
                    <div className="flex items-center gap-1">
                        <div className="w-2 h-2 rounded-sm bg-green-400" />
                        <span>Fixed (contiguous)</span>
                    </div>
                    {!isUniquelyDetermined && (
                        <div className="flex items-center gap-1">
                            <div className="w-2 h-2 rounded-sm bg-amber-400" />
                            <span>Free (compiler choice)</span>
                        </div>
                    )}
                </div>
            </div>
        );
    };

    // Render the hardware view with complete transformation flow
    const renderHardwareView = () => {
        if (!selectedLayoutPerm) return null;

        // If no transposes needed, show simple view with highlights
        if (!transposeInfo.needsAny) {
            return (
                <div className="flex flex-col items-center gap-4 p-4">
                    <div className="text-sm text-green-400 bg-green-900/30 px-4 py-2 rounded-lg border border-green-700/50">
                        ✓ All vector accesses are already contiguous – no transpose needed
                    </div>
                    <div className="flex items-center justify-center gap-4 sm:gap-6 flex-wrap">
                        {contraction.inputs.map((tensor, idx) => (
                            <React.Fragment key={tensor.name}>
                                {renderTensor(
                                    tensor,
                                    tensor.name,
                                    false,
                                    inputData[tensor.name],
                                    null,
                                    null,
                                    currentStep?.highlights[tensor.name] || null
                                )}
                                {idx < contraction.inputs.length - 1 && (
                                    <div className="text-2xl text-gray-600 font-bold self-center">×</div>
                                )}
                            </React.Fragment>
                        ))}
                        <div className="text-2xl text-gray-600 font-bold self-center">=</div>
                        {renderTensor(
                            contraction.output,
                            contraction.output.name,
                            true,
                            null,
                            computedOutput,
                            expectedOutput,
                            currentStep?.highlights[contraction.output.name] || null
                        )}
                    </div>

                    {/* Loop Nest Display */}
                    <div className="flex flex-col gap-4 mt-4 p-4 bg-gray-900/50 rounded-lg border border-gray-800 w-full max-w-3xl">
                        <div className="text-xs text-gray-400 uppercase font-semibold tracking-wide">Execution Pattern</div>

                        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                            <div className="flex flex-col gap-2">
                                <div className="text-[10px] text-gray-500 uppercase font-medium">Original (Scalar)</div>
                                <div className="font-mono text-sm text-gray-300 bg-gray-800 p-3 rounded border border-gray-700 whitespace-pre-wrap">
                                    {generateScalarLoopNest(contraction, dimensionSizes)}
                                </div>
                            </div>

                            <div className="flex flex-col gap-2">
                                <div className="text-[10px] text-green-400 uppercase font-medium">Vectorized (MAC)</div>
                                <div className="font-mono text-sm text-green-200 bg-green-900/30 p-3 rounded border border-green-700/50 whitespace-pre-wrap">
                                    {generateVectorizedLoopNest(contraction, vectorizationOption, selectedLayoutPerm, dimensionSizes, vectorWidth)}
                                </div>
                            </div>
                        </div>

                        <div className="text-[10px] text-gray-500 pt-2 border-t border-gray-800">
                            <span className="text-sky-400 font-mono">:</span> = vector dimension ({selectedVecDim})
                        </div>
                    </div>
                </div>
            );
        }

        // Full transformation flow view
        return (
            <div className="flex flex-col items-center gap-6 p-4 overflow-auto">
                {/* Transformation flow: inputs with transposes */}
                <div className="flex items-center gap-4 flex-wrap justify-center">
                    {/* Input tensors column */}
                    <div className="flex flex-col gap-4 items-end">
                        {contraction.inputs.map((tensor) => {
                            const needsTr = needsTranspose(tensor.name);
                            const transposedTensor = getTransposedTensor(tensor);
                            const originalData = inputData[tensor.name];
                            const transposedData = getTransposedData(tensor.name, originalData);
                            const highlight = currentStep?.highlights[tensor.name] || null;

                            // For tensors that don't need transpose, just show single tensor
                            if (!needsTr) {
                                return (
                                    <div key={tensor.name} className="flex flex-col items-center">
                                        <div className="text-[10px] text-gray-500 mb-1 font-mono">
                                            [{tensor.indices.join(',')}]
                                        </div>
                                        {renderTensor(
                                            tensor,
                                            tensor.name,
                                            false,
                                            originalData,
                                            null,
                                            null,
                                            highlight
                                        )}
                                    </div>
                                );
                            }

                            // For tensors that need transpose, show: original → arrow → transposed
                            return (
                                <div key={tensor.name} className="flex items-center gap-3">
                                    {/* Original tensor */}
                                    <div className="flex flex-col items-center">
                                        <div className="text-[10px] text-gray-500 mb-1 font-mono">
                                            [{tensor.indices.join(',')}]
                                        </div>
                                        {renderTensor(
                                            tensor,
                                            tensor.name,
                                            false,
                                            originalData,
                                            null,
                                            null,
                                            highlight
                                        )}
                                    </div>

                                    {/* Arrow with permutation */}
                                    <div className="flex flex-col items-center">
                                        <div className="text-[10px] font-bold px-2 py-0.5 rounded font-mono text-amber-400 bg-amber-900/50">
                                            {getPermutationString(tensor.indices, transposedTensor.indices)}
                                        </div>
                                        <ArrowRight size={24} className="text-amber-500" />
                                    </div>

                                    {/* Transposed tensor */}
                                    <div className="flex flex-col items-center">
                                        <div className="text-[10px] mb-1 font-mono px-2 py-0.5 rounded text-amber-300 bg-amber-900/30">
                                            [{transposedTensor.indices.join(',')}]
                                        </div>
                                        {renderTensor(
                                            transposedTensor,
                                            `${tensor.name}ᵀ`,
                                            false,
                                            transposedData,
                                            null,
                                            null,
                                            highlight
                                        )}
                                    </div>
                                </div>
                            );
                        })}
                    </div>

                    {/* Multiplication symbol */}
                    <div className="text-3xl text-gray-500 font-bold self-center px-2">×</div>

                    {/* Equals */}
                    <div className="text-3xl text-gray-500 font-bold self-center px-2">=</div>

                    {/* Output with transpose */}
                    <div className="flex items-center gap-3">
                        {(() => {
                            const needsTr = needsTranspose(contraction.output.name);
                            const transposedOutput = getTransposedTensor(contraction.output);
                            const transposedComputedData = getTransposedData(contraction.output.name, computedOutput);
                            const transposedExpectedData = getTransposedData(contraction.output.name, expectedOutput);
                            const highlight = currentStep?.highlights[contraction.output.name] || null;

                            // For outputs that don't need transpose, just show single tensor
                            if (!needsTr) {
                                return (
                                    <div className="flex flex-col items-center">
                                        <div className="text-[10px] text-gray-500 mb-1 font-mono">
                                            [{contraction.output.indices.join(',')}]
                                        </div>
                                        {renderTensor(
                                            contraction.output,
                                            contraction.output.name,
                                            true,
                                            null,
                                            computedOutput,
                                            expectedOutput,
                                            highlight
                                        )}
                                    </div>
                                );
                            }

                            // For outputs that need transpose, show: transposed output → arrow → original
                            return (
                                <>
                                    {/* Transposed output (result of computation) */}
                                    <div className="flex flex-col items-center">
                                        <div className="text-[10px] mb-1 font-mono px-2 py-0.5 rounded text-amber-300 bg-amber-900/30">
                                            [{transposedOutput.indices.join(',')}]
                                        </div>
                                        {renderTensor(
                                            transposedOutput,
                                            `${contraction.output.name}ᵀ`,
                                            true,
                                            null,
                                            transposedComputedData,
                                            transposedExpectedData,
                                            highlight
                                        )}
                                    </div>

                                    {/* Arrow back to original layout */}
                                    <div className="flex flex-col items-center">
                                        <div className="text-[10px] font-bold text-violet-400 bg-violet-900/50 px-2 py-0.5 rounded font-mono">
                                            {getPermutationString(transposedOutput.indices, contraction.output.indices)}
                                        </div>
                                        <ArrowRight size={24} className="text-violet-500" />
                                    </div>

                                    {/* Final output in original layout */}
                                    <div className="flex flex-col items-center">
                                        <div className="text-[10px] text-violet-300 mb-1 font-mono bg-violet-900/30 px-2 py-0.5 rounded">
                                            [{contraction.output.indices.join(',')}]
                                        </div>
                                        {renderTensor(
                                            contraction.output,
                                            contraction.output.name,
                                            true,
                                            null,
                                            computedOutput,
                                            expectedOutput,
                                            highlight
                                        )}
                                    </div>
                                </>
                            );
                        })()}
                    </div>
                </div>

                {/* Loop Nest Display */}
                <div className="flex flex-col gap-4 mt-4 p-4 bg-gray-900/50 rounded-lg border border-gray-800">
                    <div className="text-xs text-gray-400 uppercase font-semibold tracking-wide">Execution Pattern</div>

                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        {/* Scalar Loop Nest */}
                        <div className="flex flex-col gap-2">
                            <div className="text-[10px] text-gray-500 uppercase font-medium">Original (Scalar)</div>
                            <div className="font-mono text-sm text-gray-300 bg-gray-800 p-3 rounded border border-gray-700 whitespace-pre-wrap">
                                {generateScalarLoopNest(contraction, dimensionSizes)}
                            </div>
                        </div>

                        {/* Vectorized Loop Nest */}
                        <div className="flex flex-col gap-2">
                            <div className="text-[10px] text-amber-400 uppercase font-medium">Vectorized (MAC)</div>
                            <div className="font-mono text-sm text-amber-200 bg-amber-900/30 p-3 rounded border border-amber-700/50 whitespace-pre-wrap">
                                {generateVectorizedLoopNest(contraction, vectorizationOption, selectedLayoutPerm, dimensionSizes, vectorWidth)}
                            </div>
                        </div>
                    </div>

                    {/* Legend */}
                    <div className="text-[10px] text-gray-500 pt-2 border-t border-gray-800">
                        <span className="text-sky-400 font-mono">:</span> = vector dimension ({selectedVecDim}) &nbsp;|&nbsp;
                        <span className="text-amber-400 font-bold">ᵀ</span> = transposed &nbsp;|&nbsp;
                        <span className="font-mono">.permute()</span> = dimension reorder
                    </div>
                </div>
            </div>
        );
    };

    return (
        <div className="flex flex-col h-full">
            {/* View Mode Tabs */}
            <div className="flex items-center gap-4 px-4 py-2 bg-gray-900/60 border-b border-gray-800/50">
                <span className="text-xs text-gray-500 uppercase font-medium">View:</span>

                <div className="flex rounded-lg overflow-hidden border border-gray-700">
                    <button
                        onClick={() => setViewMode('logical')}
                        className={`flex items-center gap-2 px-4 py-1.5 text-xs font-medium transition-colors ${viewMode === 'logical'
                            ? 'bg-gradient-to-r from-sky-600 to-blue-600 text-white'
                            : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
                            }`}
                    >
                        <Layers size={14} />
                        Logical
                    </button>
                    <button
                        onClick={() => setViewMode('hardware')}
                        className={`flex items-center gap-2 px-4 py-1.5 text-xs font-medium transition-colors ${viewMode === 'hardware'
                            ? 'bg-gradient-to-r from-amber-600 to-orange-600 text-white'
                            : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
                            }`}
                    >
                        <Cpu size={14} />
                        Hardware
                    </button>
                </div>
            </div>

            {/* Layout Selector (only in hardware mode) */}
            {viewMode === 'hardware' && renderLayoutSelector()}

            {/* Tensor Visualization */}
            <div className="flex-1 flex items-center justify-center overflow-auto min-h-[300px]">
                {viewMode === 'logical' ? (
                    // Logical View - simple original layout
                    <div className="flex items-center justify-center p-4 gap-4 sm:gap-6 flex-wrap">
                        {contraction.inputs.map((tensor, idx) => (
                            <React.Fragment key={tensor.name}>
                                {renderTensor(
                                    tensor,
                                    tensor.name,
                                    false,
                                    inputData[tensor.name],
                                    null,
                                    null,
                                    currentStep?.highlights[tensor.name] || null
                                )}
                                {idx < contraction.inputs.length - 1 && (
                                    <div className="text-2xl text-gray-600 font-bold self-center">×</div>
                                )}
                            </React.Fragment>
                        ))}
                        <div className="text-2xl text-gray-600 font-bold self-center">=</div>
                        {renderTensor(
                            contraction.output,
                            contraction.output.name,
                            true,
                            null,
                            computedOutput,
                            expectedOutput,
                            currentStep?.highlights[contraction.output.name] || null
                        )}
                    </div>
                ) : (
                    // Hardware View - transformation flow
                    renderHardwareView()
                )}
            </div>
        </div>
    );
};

export default ViewPanel;
