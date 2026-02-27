// Tensor data operations: generate random data, compute expected results, simulate operations

import { TensorSpec, DimensionSizes, TensorData, TensorDataMap, ContractionSpec, VerificationResult, OperationStep } from '../types';

/**
 * Get the flat index for a multi-dimensional tensor given index values
 */
export function getFlatIndex(shape: number[], indexValues: number[]): number {
    let flatIndex = 0;
    let multiplier = 1;
    for (let i = shape.length - 1; i >= 0; i--) {
        flatIndex += indexValues[i] * multiplier;
        multiplier *= shape[i];
    }
    return flatIndex;
}

/**
 * Get the total size of a tensor (product of dimensions)
 */
export function getTensorSize(shape: number[]): number {
    return shape.reduce((a, b) => a * b, 1);
}

/**
 * Generate random tensor data with integer values for easier visualization
 */
export function generateRandomTensor(shape: number[], maxValue: number = 9): TensorData {
    const size = getTensorSize(shape);
    const data: TensorData = [];
    for (let i = 0; i < size; i++) {
        // Use integers 1-9 for easier mental verification
        data.push(Math.floor(Math.random() * maxValue) + 1);
    }
    return data;
}

/**
 * Create a zero-initialized tensor
 */
export function createZeroTensor(shape: number[]): TensorData {
    const size = getTensorSize(shape);
    return new Array(size).fill(0);
}

/**
 * Apply dimension sizes to get actual tensor shape
 */
export function getTensorShape(tensor: TensorSpec, sizes: DimensionSizes): number[] {
    return tensor.indices.map(idx => sizes[idx] || 4);
}

/**
 * Get value from tensor at given index values
 */
export function getTensorValue(data: TensorData, shape: number[], indexValues: number[]): number {
    const flatIdx = getFlatIndex(shape, indexValues);
    return data[flatIdx] || 0;
}

/**
 * Set value in tensor at given index values
 */
export function setTensorValue(data: TensorData, shape: number[], indexValues: number[], value: number): void {
    const flatIdx = getFlatIndex(shape, indexValues);
    data[flatIdx] = value;
}

/**
 * Add value to tensor at given index values (accumulate)
 */
export function addToTensorValue(data: TensorData, shape: number[], indexValues: number[], value: number): void {
    const flatIdx = getFlatIndex(shape, indexValues);
    data[flatIdx] = (data[flatIdx] || 0) + value;
}

/**
 * Compute expected output using scalar operations (ground truth)
 */
export function computeExpectedOutput(
    contraction: ContractionSpec,
    sizes: DimensionSizes,
    inputData: TensorDataMap
): TensorData {
    const outputShape = getTensorShape(contraction.output, sizes);
    const result = createZeroTensor(outputShape);

    // Get shapes for all tensors
    const inputShapes = contraction.inputs.map(t => getTensorShape(t, sizes));

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

    // All indices involved
    const allIndices = [...new Set([
        ...contraction.outputIndices,
        ...contraction.reductionIndices
    ])];

    // Determine operation type
    const hasReduction = contraction.reductionIndices.length > 0;

    // Iterate over all combinations
    for (const indexVals of iterateIndices(allIndices)) {
        // Compute product of inputs (einsum semantics: multiply then sum over reduction)
        let product = 1;
        for (let i = 0; i < contraction.inputs.length; i++) {
            const input = contraction.inputs[i];
            const inputShape = inputShapes[i];
            const inputIndexVals = input.indices.map(idx => indexVals[idx]);
            const value = getTensorValue(inputData[input.name], inputShape, inputIndexVals);
            product *= value;
        }

        // For contractions (k > 0): accumulate (MAC)
        // For element-wise (k = 0): direct assignment (no reduction to sum over)
        const outputIndexVals = contraction.output.indices.map(idx => indexVals[idx]);
        if (hasReduction) {
            addToTensorValue(result, outputShape, outputIndexVals, product);
        } else {
            setTensorValue(result, outputShape, outputIndexVals, product);
        }
    }

    return result;
}

/**
 * Simulate a single MAC step and return updated output data
 */
export function simulateMACStep(
    step: OperationStep,
    contraction: ContractionSpec,
    sizes: DimensionSizes,
    inputData: TensorDataMap,
    currentOutput: TensorData
): TensorData {
    const result = [...currentOutput]; // Clone
    const outputShape = getTensorShape(contraction.output, sizes);

    // Extract index ranges from the step
    const outputHighlight = step.highlights[contraction.output.name];
    if (!outputHighlight) return result;

    // Get the index ranges for this step
    const outputIndices = contraction.output.indices;

    // Helper to expand range to array of values
    function expandRange(val: number | [number, number]): number[] {
        if (Array.isArray(val)) {
            const result = [];
            for (let i = val[0]; i <= val[1]; i++) result.push(i);
            return result;
        }
        return [val];
    }

    // Generate all output index combinations for this step
    function* iterateStepIndices(): Generator<{ [key: string]: number }> {
        const ranges: { idx: string; values: number[] }[] = outputIndices.map(idx => ({
            idx,
            values: expandRange(outputHighlight.indices[idx] ?? 0)
        }));

        function* iterate(remaining: typeof ranges, current: { [key: string]: number }): Generator<{ [key: string]: number }> {
            if (remaining.length === 0) {
                yield current;
                return;
            }
            const [first, ...rest] = remaining;
            for (const val of first.values) {
                yield* iterate(rest, { ...current, [first.idx]: val });
            }
        }

        yield* iterate(ranges, {});
    }

    // Determine operation type
    const hasReduction = contraction.reductionIndices.length > 0;

    // For each output position in this step, compute the operation
    for (const outputVals of iterateStepIndices()) {
        // Get values from each input
        let product = 1;
        for (const input of contraction.inputs) {
            const inputHighlight = step.highlights[input.name];
            if (!inputHighlight) continue;

            // Build index values for this input
            const inputShape = getTensorShape(input, sizes);
            const inputIndexVals = input.indices.map(idx => {
                const highlightVal = inputHighlight.indices[idx];
                if (Array.isArray(highlightVal)) {
                    // This is a vectorized dimension, use the corresponding output value
                    return outputVals[idx] ?? highlightVal[0];
                }
                return highlightVal ?? 0;
            });

            const value = getTensorValue(inputData[input.name], inputShape, inputIndexVals);
            product *= value;
        }

        // For contractions: accumulate (MAC). For element-wise: direct set.
        const outputIndexVals = outputIndices.map(idx => outputVals[idx]);
        if (hasReduction) {
            addToTensorValue(result, outputShape, outputIndexVals, product);
        } else {
            setTensorValue(result, outputShape, outputIndexVals, product);
        }
    }

    return result;
}

/**
 * Verify if computed output matches expected output
 */
export function verifyOutput(
    computed: TensorData,
    expected: TensorData,
    tolerance: number = 1e-6
): VerificationResult {
    let maxError = 0;
    let mismatchCount = 0;

    for (let i = 0; i < expected.length; i++) {
        const error = Math.abs((computed[i] || 0) - (expected[i] || 0));
        maxError = Math.max(maxError, error);
        if (error > tolerance) {
            mismatchCount++;
        }
    }

    return {
        passed: mismatchCount === 0,
        maxError,
        totalElements: expected.length,
        mismatchCount
    };
}

/**
 * Generate all tensor data for a contraction
 */
export function initializeTensorData(
    contraction: ContractionSpec,
    sizes: DimensionSizes
): { inputData: TensorDataMap; expectedOutput: TensorData } {
    const inputData: TensorDataMap = {};

    // Generate random data for each input
    for (const input of contraction.inputs) {
        const shape = getTensorShape(input, sizes);
        inputData[input.name] = generateRandomTensor(shape);
    }

    // Compute expected output
    const expectedOutput = computeExpectedOutput(contraction, sizes, inputData);

    return { inputData, expectedOutput };
}
