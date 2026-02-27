// Generate operation steps for tensor operation visualization
// Supports contractions (k > 0), element-wise binary (k = 0, T ≥ 2),
// and unary operations (k = 0, T = 1).

import { ContractionSpec, DimensionSizes, OperationStep, VectorizationOption } from '../types';

/**
 * Generate steps for visualizing a tensor operation
 * with a specific vectorization strategy.
 * Handles contractions (k > 0), element-wise (k = 0), and unary (T = 1).
 */
export function generateSteps(
    contraction: ContractionSpec,
    sizes: DimensionSizes,
    vectorizeDim: string,
    vectorWidth: number,
    option: VectorizationOption
): OperationStep[] {
    const steps: OperationStep[] = [];

    // Get the output indices and their sizes
    const outputIndices = contraction.outputIndices;
    const reductionIndices = contraction.reductionIndices;

    // Identify which dimension is vectorized
    const vectorizedDimSize = sizes[vectorizeDim] || 4;

    // Generate all combinations of non-vectorized output indices
    // and reduction indices for iteration
    const nonVectorizedOutputIndices = outputIndices.filter(d => d !== vectorizeDim);

    // Helper to iterate over all combinations of index values
    function* iterateIndices(indices: string[]): Generator<{ [key: string]: number }> {
        if (indices.length === 0) {
            yield {};
            return;
        }

        const [first, ...rest] = indices;
        const size = sizes[first] || 4;

        for (let val = 0; val < size; val++) {
            for (const restVals of iterateIndices(rest)) {
                yield { [first]: val, ...restVals };
            }
        }
    }

    // Iterate over non-vectorized output dims, then strip-mine vectorized dim, then reduction dims
    for (const outputVals of iterateIndices(nonVectorizedOutputIndices)) {
        // Strip-mine the vectorized dimension
        for (let vecStart = 0; vecStart < vectorizedDimSize; vecStart += vectorWidth) {
            const vecEnd = Math.min(vecStart + vectorWidth - 1, vectorizedDimSize - 1);

            // Iterate over reduction dimensions
            for (const reductionVals of iterateIndices(reductionIndices)) {
                // Build the step
                const step = buildStep(
                    steps.length,
                    contraction,
                    vectorizeDim,
                    vecStart,
                    vecEnd,
                    outputVals,
                    reductionVals,
                    option
                );
                steps.push(step);
            }
        }
    }

    return steps;
}

/**
 * Build a single operation step
 */
function buildStep(
    id: number,
    contraction: ContractionSpec,
    vectorizeDim: string,
    vecStart: number,
    vecEnd: number,
    outputVals: { [key: string]: number },
    reductionVals: { [key: string]: number },
    option: VectorizationOption
): OperationStep {
    const highlights: OperationStep['highlights'] = {};

    // Helper to format index values for formula
    const formatIndices = (indices: string[], vals: { [k: string]: number | [number, number] }) => {
        return indices.map(idx => {
            const v = vals[idx];
            if (Array.isArray(v)) {
                return v[0] === v[1] ? `${v[0]}` : `${v[0]}:${v[1]}`;
            }
            return `${v}`;
        }).join(',');
    };

    // Build index values for each tensor
    const allVals: { [key: string]: number | [number, number] } = {
        ...outputVals,
        ...reductionVals,
        [vectorizeDim]: [vecStart, vecEnd]
    };

    // Output tensor
    const outputIndices: { [k: string]: number | [number, number] } = {};
    for (const idx of contraction.output.indices) {
        outputIndices[idx] = allVals[idx] ?? 0;
    }
    highlights[contraction.output.name] = {
        indices: outputIndices,
        type: 'output'
    };

    // Input tensors
    for (const input of contraction.inputs) {
        const isBroadcast = option.broadcastTensors.includes(input.name);
        const inputIndices: { [k: string]: number | [number, number] } = {};

        for (const idx of input.indices) {
            if (idx === vectorizeDim) {
                // This tensor has the vectorized dimension
                inputIndices[idx] = [vecStart, vecEnd];
            } else {
                inputIndices[idx] = allVals[idx] ?? 0;
            }
        }

        highlights[input.name] = {
            indices: inputIndices,
            type: isBroadcast ? 'scalar-broadcast' : 'vector'
        };
    }

    // Build description and formula
    const outputStr = `${contraction.output.name}[${formatIndices(contraction.output.indices, outputIndices)}]`;
    const inputStrs = contraction.inputs.map(input => {
        const vals: { [k: string]: number | [number, number] } = {};
        for (const idx of input.indices) {
            vals[idx] = highlights[input.name].indices[idx];
        }
        const suffix = option.broadcastTensors.includes(input.name) ? ' (broadcast)' : '';
        return `${input.name}[${formatIndices(input.indices, vals)}]${suffix}`;
    });

    // Determine operation type based on presence of reduction dimensions
    const hasReduction = contraction.reductionIndices.length > 0;
    const isUnary = contraction.inputs.length === 1;

    let formula: string;
    let opLabel: string;

    if (hasReduction) {
        // Contraction: MAC with accumulation
        formula = `${outputStr} += ${inputStrs.join(' × ')}`;
        opLabel = 'MAC';
    } else if (isUnary) {
        // Unary element-wise: f(input)
        formula = `${outputStr} = f(${inputStrs[0]})`;
        opLabel = 'Unary';
    } else {
        // Element-wise binary: input1 ⊕ input2
        formula = `${outputStr} = ${inputStrs.join(' ⊕ ')}`;
        opLabel = 'Elem-wise';
    }

    const reductionDesc = hasReduction
        ? `reduction ${Object.entries(reductionVals).map(([k, v]) => `${k}=${v}`).join(', ')}`
        : 'no reduction';
    const description = `${opLabel}: Vectorize '${vectorizeDim}'[${vecStart}:${vecEnd}], ${reductionDesc}`;

    return {
        id,
        description,
        formula,
        highlights
    };
}

/**
 * Get a summary of the vectorization strategy
 */
export function getStrategyDescription(option: VectorizationOption, vectorWidth: number, contraction?: ContractionSpec): string {
    const hasReduction = contraction ? contraction.reductionIndices.length > 0 : true;
    const isUnary = contraction ? contraction.inputs.length === 1 : false;

    if (isUnary) {
        return `Unary: vectorize along '${option.dimension}' — apply f() to ${vectorWidth} elements per cycle (w=${vectorWidth})`;
    }
    if (!hasReduction && option.broadcastTensors.length === 0) {
        return `Element-wise: vectorize along '${option.dimension}' — all inputs aligned, ${vectorWidth} elements per cycle (w=${vectorWidth})`;
    }
    if (option.broadcastTensors.length === 0) {
        return `Batch vectorization along '${option.dimension}' — all inputs have this dimension, no broadcast needed! (w=${vectorWidth})`;
    }
    return `Vectorize along '${option.dimension}' — ${option.broadcastTensors.join(', ')} broadcast to all ${vectorWidth} lanes`;
}
