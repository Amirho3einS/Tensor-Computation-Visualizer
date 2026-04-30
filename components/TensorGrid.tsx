import React from 'react';
import { TensorSpec, DimensionSizes, OperationStep, TensorData } from '../types';
import { Maximize2, Check, X as XIcon } from 'lucide-react';
import { getFlatIndex, formatTensorValue } from '../utils/tensorData';
import Tensor3DView from './Tensor3DView';

interface TensorGridProps {
    tensor: TensorSpec;
    sizes: DimensionSizes;
    highlight?: OperationStep['highlights'][string] | null;
    label: string;
    data?: TensorData | null;           // Actual tensor values
    computedData?: TensorData | null;    // Computed output (for output tensor)
    expectedData?: TensorData | null;    // Expected output (for verification)
    showValues?: boolean;                // Whether to show numeric values
}

/**
 * Render a tensor up to 3D as a grid visualization
 * - 1D: single row
 * - 2D: grid
 * - 3D: full 3D rotatable view
 */
const TensorGrid: React.FC<TensorGridProps> = ({
    tensor,
    sizes,
    highlight,
    label,
    data,
    computedData,
    expectedData,
    showValues = true
}) => {
    const dims = tensor.indices.map(idx => sizes[idx] || 4);
    const ndims = dims.length;

    // For 3D tensors, use the interactive 3D view
    if (ndims === 3) {
        return (
            <Tensor3DView
                tensor={tensor}
                sizes={sizes}
                highlight={highlight}
                label={label}
                data={data}
                computedData={computedData}
                expectedData={expectedData}
                showValues={showValues}
            />
        );
    }

    // For 4D+ tensors, show as a series of 3D cubes (slicing along the first dimensions)
    if (ndims >= 4) {
        // Calculate outer dimensions (all but the last 3)
        const outerDims = dims.slice(0, -3);
        const innerDims = dims.slice(-3);
        const outerIndices = tensor.indices.slice(0, -3);
        const innerIndices = tensor.indices.slice(-3);

        // Generate all combinations of outer indices
        const generateOuterCombinations = (dimensions: number[]): number[][] => {
            if (dimensions.length === 0) return [[]];
            const result: number[][] = [];
            const iterate = (remaining: number[], current: number[]) => {
                if (remaining.length === 0) {
                    result.push(current);
                    return;
                }
                const [first, ...rest] = remaining;
                for (let i = 0; i < first; i++) {
                    iterate(rest, [...current, i]);
                }
            };
            iterate(dimensions, []);
            return result;
        };

        const outerCombinations = generateOuterCombinations(outerDims);

        // Get label color based on highlight
        const getLabelColor = () => {
            if (!highlight) return 'text-gray-400';
            switch (highlight.type) {
                case 'scalar-broadcast': return 'text-amber-400';
                case 'vector': return 'text-sky-400';
                case 'output': return 'text-violet-400';
                default: return 'text-gray-400';
            }
        };

        return (
            <div className="flex flex-col items-center">
                <h3 className={`font-bold text-sm mb-2 flex items-center gap-2 ${getLabelColor()}`}>
                    {label}
                    <span className="text-xs text-gray-500 font-normal">
                        [{tensor.indices.join(',')}] = ({dims.join('×')})
                    </span>
                </h3>

                <div className="flex gap-4 flex-wrap justify-center">
                    {outerCombinations.map((outerIdx, idx) => {
                        // Create a virtual 3D tensor for this slice
                        const sliceTensor: TensorSpec = {
                            name: tensor.name,
                            indices: innerIndices,
                            shape: innerDims
                        };

                        // Create slice-specific sizes
                        const sliceSizes: DimensionSizes = {};
                        innerIndices.forEach((dimIdx, i) => {
                            sliceSizes[dimIdx] = innerDims[i];
                        });

                        // Create slice label
                        const sliceLabel = outerIndices.map((idx, i) => `${idx}=${outerIdx[i]}`).join(', ');

                        // Filter highlight for this slice
                        let sliceHighlight = highlight;
                        if (highlight) {
                            // Check if any outer index doesn't match
                            let matches = true;
                            for (let i = 0; i < outerIndices.length; i++) {
                                const dimName = outerIndices[i];
                                const range = highlight.indices[dimName];
                                if (range !== undefined) {
                                    if (Array.isArray(range)) {
                                        if (outerIdx[i] < range[0] || outerIdx[i] > range[1]) {
                                            matches = false;
                                            break;
                                        }
                                    } else {
                                        if (outerIdx[i] !== range) {
                                            matches = false;
                                            break;
                                        }
                                    }
                                }
                            }
                            if (!matches) {
                                sliceHighlight = null;
                            }
                        }

                        // Calculate offset into the flat data arrays for this slice
                        const innerSize = innerDims.reduce((a, b) => a * b, 1);
                        const sliceOffset = idx * innerSize;

                        // Extract slice data
                        const sliceData = data ? data.slice(sliceOffset, sliceOffset + innerSize) : null;
                        const sliceComputedData = computedData ? computedData.slice(sliceOffset, sliceOffset + innerSize) : null;
                        const sliceExpectedData = expectedData ? expectedData.slice(sliceOffset, sliceOffset + innerSize) : null;

                        return (
                            <div key={idx} className="flex flex-col items-center">
                                <div className="text-[10px] text-gray-500 mb-1 font-mono bg-gray-800/50 px-2 py-0.5 rounded">
                                    {sliceLabel}
                                </div>
                                <Tensor3DView
                                    tensor={sliceTensor}
                                    sizes={sliceSizes}
                                    highlight={sliceHighlight}
                                    label=""
                                    data={sliceData}
                                    computedData={sliceComputedData}
                                    expectedData={sliceExpectedData}
                                    showValues={showValues}
                                />
                            </div>
                        );
                    })}
                </div>
            </div>
        );
    }

    // Get shape for flat index calculation
    const shape = dims;

    // Determine if a cell is highlighted
    const isHighlighted = (indexValues: number[]): 'vector' | 'scalar-broadcast' | 'output' | null => {
        if (!highlight) return null;

        // Check each dimension
        for (let d = 0; d < tensor.indices.length; d++) {
            const idx = tensor.indices[d];
            const val = indexValues[d];
            const range = highlight.indices[idx];

            if (range === undefined) continue;

            if (Array.isArray(range)) {
                // Range [start, end]
                if (val < range[0] || val > range[1]) return null;
            } else {
                // Single value
                if (val !== range) return null;
            }
        }

        return highlight.type;
    };

    // Get value at index
    const getValue = (indexValues: number[], useData: TensorData | null | undefined): number | null => {
        if (!useData) return null;
        const flatIdx = getFlatIndex(shape, indexValues);
        return useData[flatIdx] ?? null;
    };

    // Check if computed matches expected for output tensor
    const getVerificationStatus = (indexValues: number[]): 'match' | 'mismatch' | null => {
        if (!computedData || !expectedData) return null;
        const flatIdx = getFlatIndex(shape, indexValues);
        const computed = computedData[flatIdx] ?? 0;
        const expected = expectedData[flatIdx] ?? 0;
        return Math.abs(computed - expected) < 0.001 ? 'match' : 'mismatch';
    };

    // Format index label like [0,1,2]
    const formatIndexLabel = (indexValues: number[]): string => {
        return `[${indexValues.join(',')}]`;
    };

    // Get cell style based on highlight state
    const getCellStyle = (state: 'vector' | 'scalar-broadcast' | 'output' | null, verification: 'match' | 'mismatch' | null) => {
        // Larger cells to fit value + index label
        const base = "relative w-11 h-12 sm:w-12 sm:h-14 border flex flex-col items-center justify-center transition-all duration-200 font-mono select-none rounded ";

        // If we have verification data for output, show that
        if (verification === 'match' && state === null) {
            return base + "bg-green-900/30 border-green-700/50 text-green-300";
        }
        if (verification === 'mismatch' && state === null) {
            return base + "bg-red-900/30 border-red-700/50 text-red-300";
        }

        switch (state) {
            case 'scalar-broadcast':
                return base + "bg-amber-500 border-amber-300 z-20 scale-110 shadow-[0_0_12px_rgba(245,158,11,0.7)]";
            case 'vector':
                return base + "bg-sky-500 border-sky-300 z-10 scale-105 shadow-[0_0_10px_rgba(14,165,233,0.6)]";
            case 'output':
                return base + "bg-violet-500 border-violet-300 z-10 scale-105 shadow-[0_0_12px_rgba(139,92,246,0.6)]";
            default:
                return base + "bg-gray-800/50 border-gray-700/50 text-gray-300";
        }
    };

    // Render cells for 1D or 2D slice
    const renderSlice = (sliceIndices: number[], rows: number, cols: number) => {
        const cells = [];
        for (let r = 0; r < rows; r++) {
            for (let c = 0; c < cols; c++) {
                let indexValues: number[];
                if (ndims === 1) {
                    indexValues = [c];
                } else if (ndims === 2) {
                    indexValues = [r, c];
                } else {
                    indexValues = [...sliceIndices, r, c];
                }

                const state = isHighlighted(indexValues);
                const verification = getVerificationStatus(indexValues);

                // Get the display value
                const displayData = computedData || data;
                const value = getValue(indexValues, displayData);

                cells.push(
                    <div key={indexValues.join('-')} className={getCellStyle(state, verification)}>
                        {state === 'scalar-broadcast' ? (
                            <>
                                <div className="absolute -top-0.5 -right-0.5 w-2 h-2 bg-amber-300 rounded-full animate-ping" />
                                <Maximize2 size={14} className="text-white" />
                                <span className="text-[7px] text-amber-200/70 mt-0.5">{formatIndexLabel(indexValues)}</span>
                            </>
                        ) : (
                            <>
                                {/* Value */}
                                <span className="text-sm font-bold leading-none">
                                    {showValues && value !== null ? formatTensorValue(value) : '–'}
                                </span>
                                {/* Index label */}
                                <span className="text-[7px] text-gray-500 mt-0.5 leading-none">
                                    {formatIndexLabel(indexValues)}
                                </span>
                            </>
                        )}
                        {/* Mini verification indicator */}
                        {verification && state === null && (
                            <div className={`absolute -top-1 -right-1 w-3.5 h-3.5 rounded-full flex items-center justify-center ${verification === 'match' ? 'bg-green-500' : 'bg-red-500'
                                }`}>
                                {verification === 'match' ? <Check size={9} className="text-white" /> : <XIcon size={9} className="text-white" />}
                            </div>
                        )}
                    </div>
                );
            }
        }
        return cells;
    };

    // Calculate display dimensions
    let outerSlices = 1;
    let rows = 1;
    let cols = dims[dims.length - 1] || 1;

    if (ndims >= 2) {
        rows = dims[dims.length - 2];
    }
    if (ndims >= 3) {
        outerSlices = dims.slice(0, -2).reduce((a, b) => a * b, 1);
    }

    // Generate outer slice indices for 3D+
    const generateSliceIndices = (total: number, dims: number[]): number[][] => {
        if (dims.length === 0) return [[]];
        const result: number[][] = [];

        const iterate = (remaining: number[], current: number[]) => {
            if (remaining.length === 0) {
                result.push(current);
                return;
            }
            const [first, ...rest] = remaining;
            for (let i = 0; i < first; i++) {
                iterate(rest, [...current, i]);
            }
        };

        iterate(dims, []);
        return result;
    };

    const sliceIndicesList = ndims >= 3 ? generateSliceIndices(outerSlices, dims.slice(0, -2)) : [[]];

    // Determine border color based on highlight
    const getBorderColor = () => {
        if (!highlight) return 'border-gray-700/30';
        switch (highlight.type) {
            case 'scalar-broadcast': return 'border-amber-500/50';
            case 'vector': return 'border-sky-500/50';
            case 'output': return 'border-violet-500/50';
            default: return 'border-gray-700/30';
        }
    };

    // Get label color
    const getLabelColor = () => {
        if (!highlight) return 'text-gray-400';
        switch (highlight.type) {
            case 'scalar-broadcast': return 'text-amber-400';
            case 'vector': return 'text-sky-400';
            case 'output': return 'text-violet-400';
            default: return 'text-gray-400';
        }
    };

    return (
        <div className="flex flex-col items-center">
            <h3 className={`font-bold text-sm mb-2 flex items-center gap-2 ${getLabelColor()}`}>
                {label}
                <span className="text-xs text-gray-500 font-normal">
                    [{tensor.indices.join(',')}] = ({dims.join('×')})
                </span>
            </h3>

            <div className={`flex gap-3 flex-wrap justify-center ${ndims >= 3 ? 'p-2' : ''}`}>
                {sliceIndicesList.map((sliceIdx, si) => (
                    <div key={si} className="flex flex-col items-center">
                        {ndims >= 3 && (
                            <div className="text-[10px] text-gray-500 mb-1 font-mono">
                                {tensor.indices.slice(0, -2).map((idx, i) => `${idx}=${sliceIdx[i]}`).join(', ')}
                            </div>
                        )}
                        <div
                            className={`grid gap-1 p-2 border rounded-lg bg-gray-900/50 backdrop-blur transition-colors ${getBorderColor()}`}
                            style={{ gridTemplateColumns: `repeat(${cols}, min-content)` }}
                        >
                            {renderSlice(sliceIdx, rows, cols)}
                        </div>
                    </div>
                ))}
            </div>

            {/* Type indicator */}
            {highlight && (
                <div className={`mt-2 text-[10px] font-medium uppercase tracking-wide ${getLabelColor()}`}>
                    {highlight.type === 'scalar-broadcast' ? '⟨ Broadcast ⟩' :
                        highlight.type === 'output' ? '⟨ Output ⟩' : '⟨ Vector ⟩'}
                </div>
            )}
        </div>
    );
};

export default TensorGrid;
