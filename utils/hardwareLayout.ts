/**
 * Hardware Layout Analysis
 *
 * Given a vectorization dimension v, this module determines ALL valid
 * data layouts for each tensor in the contraction.
 *
 * KEY INSIGHT:
 *   v uniquely fixes which dimension is CONTIGUOUS (last in memory) for
 *   each tensor:
 *     - Vector tensors (containing v): v must be last
 *     - Broadcast tensors (not containing v): innermost reduction dim last
 *     - Output tensor: v must be last
 *
 *   However, the ordering of the REMAINING dimensions is NOT uniquely
 *   determined. Each permutation of the free dimensions produces a valid
 *   layout. Different orderings have different row-buffer locality
 *   properties depending on the loop nesting.
 *
 *   The number of valid layouts per tensor is:
 *     (number of free dimensions)!
 *
 *   For the whole contraction, the valid layout space is the Cartesian
 *   product: Π_t (free_dims_t)! across all tensors t.
 *
 *   In practice, the "best" ordering matches the loop nesting so that
 *   the innermost loop's index is next-to-last in memory, the next loop's
 *   index is next-to-next-to-last, etc. — this minimizes row buffer misses.
 */

import { ContractionSpec, VectorizationOption, TensorSpec, DimensionSizes } from '../types';

export interface TensorLayout {
    originalIndices: string[];    // Original index order
    transposedIndices: string[];  // Index order after transpose (if needed)
    needsTranspose: boolean;      // Whether this tensor needs transposing
    transposeReason?: string;     // Why transpose is needed
    accessType: 'vector' | 'broadcast'; // How this tensor is accessed
    fixedDim: string | null;      // The dimension that MUST be contiguous (last)
    freeDims: string[];           // Dimensions whose order is a free choice
}

export interface LayoutPermutation {
    id: string;
    tensorLayouts: { [tensorName: string]: TensorLayout };
    isNaturalOrder: boolean;      // True if this matches the original einsum index ordering
    description: string;          // Human-readable description
    loopOrder: string[];          // The corresponding loop order for this layout
}

export interface HardwareViewConfig {
    vectorizeDimension: string;
    vectorType: 'output' | 'reduction';
    layoutPermutations: LayoutPermutation[];
    freeOutputDims: string[];     // Non-vectorized output dims (the free choices)
    reductionDims: string[];      // Reduction dims
    totalPermutations: number;    // Total number of valid layout combinations
    isUniquelyDetermined: boolean; // True if only 1 valid layout exists
}

/**
 * Generate all permutations of an array
 */
function permutations<T>(arr: T[]): T[][] {
    if (arr.length <= 1) return [arr];
    const result: T[][] = [];
    for (let i = 0; i < arr.length; i++) {
        const rest = [...arr.slice(0, i), ...arr.slice(i + 1)];
        for (const perm of permutations(rest)) {
            result.push([arr[i], ...perm]);
        }
    }
    return result;
}

/**
 * Determine which dimension must be contiguous (last) for a tensor
 */
function getFixedContiguousDim(
    tensor: TensorSpec,
    contraction: ContractionSpec,
    vectorDim: string,
    option: VectorizationOption
): string | null {
    const isBroadcast = option.broadcastTensors.includes(tensor.name);

    if (!isBroadcast && tensor.indices.includes(vectorDim)) {
        // Vector access: vectorized dim must be contiguous
        return vectorDim;
    }

    if (isBroadcast) {
        // Broadcast: reduction dim should be contiguous for stride-1 scalar iteration
        for (const redDim of contraction.reductionIndices) {
            if (tensor.indices.includes(redDim)) {
                return redDim;
            }
        }
        // If tensor has no reduction dims, check non-vectorized output dims
        for (const outIdx of contraction.output.indices) {
            if (tensor.indices.includes(outIdx) && outIdx !== vectorDim) {
                return outIdx;
            }
        }
    }

    // Output tensor: vectorized dim must be contiguous
    if (tensor.name === contraction.output.name) {
        return vectorDim;
    }

    return null;
}

/**
 * Build a layout for a tensor given a specific ordering of free dimensions
 */
function buildTensorLayout(
    tensor: TensorSpec,
    fixedDim: string | null,
    freeDimOrder: string[],
    accessType: 'vector' | 'broadcast'
): TensorLayout {
    let transposedIndices: string[];

    if (fixedDim && tensor.indices.includes(fixedDim)) {
        // Place free dims first (in given order), then fixed dim last
        transposedIndices = [...freeDimOrder, fixedDim];
    } else {
        // No fixed dim (1D tensor or dim not present) — use free dim order as-is
        transposedIndices = freeDimOrder.length > 0 ? freeDimOrder : [...tensor.indices];
    }

    // Filter to only dims that this tensor actually has
    transposedIndices = transposedIndices.filter(d => tensor.indices.includes(d));

    // If tensor has dims not in our computed list, they should already be there
    // (this handles edge cases)
    if (transposedIndices.length !== tensor.indices.length) {
        // Fallback: keep original order
        transposedIndices = [...tensor.indices];
    }

    const needsTranspose = JSON.stringify(transposedIndices) !== JSON.stringify(tensor.indices);
    const freeDims = fixedDim
        ? tensor.indices.filter(d => d !== fixedDim)
        : [];

    return {
        originalIndices: tensor.indices,
        transposedIndices,
        needsTranspose,
        transposeReason: needsTranspose
            ? `Reorder to [${transposedIndices.join(',')}] for contiguous ${accessType === 'vector' ? 'vector' : 'scalar'} access`
            : undefined,
        accessType,
        fixedDim,
        freeDims
    };
}

/**
 * Get the loop order that matches a given permutation of free output dims
 */
function getLoopOrder(
    freeOutputDimPerm: string[],
    vectorDim: string,
    reductionDims: string[]
): string[] {
    // Loop order: [free output dims in given order] → [vec dim tiles] → [reduction dims]
    // We don't include the vec dim in the explicit loop order since it's strip-mined
    return [...freeOutputDimPerm, ...reductionDims];
}

/**
 * Check if a permutation of free dims matches the natural ordering
 * from the original einsum expression.
 */
function isNaturalDimOrder(
    freeDimPerm: string[],
    originalOutputIndices: string[],
    vectorDim: string
): boolean {
    // Natural order = the order the free dims appear in the einsum output
    const naturalFreeOrder = originalOutputIndices.filter(d => d !== vectorDim);
    return JSON.stringify(freeDimPerm) === JSON.stringify(naturalFreeOrder);
}

/**
 * Analyze all valid hardware layouts for a given vectorization choice.
 *
 * Returns ALL valid layout permutations, with annotations about which
 * is optimal for row-buffer locality.
 */
export function analyzeHardwareLayouts(
    contraction: ContractionSpec,
    option: VectorizationOption,
    _sizes?: DimensionSizes
): HardwareViewConfig {
    const dim = option.dimension;
    const isReduction = contraction.reductionIndices.includes(dim);

    // Identify free dimensions
    const outputIndices = contraction.output.indices;
    const freeOutputDims = outputIndices.filter(d => d !== dim);
    const reductionDims = [...contraction.reductionIndices];

    // All permutations of the free output dims
    const freeOutputPerms = permutations(freeOutputDims);

    // For simplicity (and because reduction dims are usually 1),
    // we also permute reduction dims
    const reductionPerms = permutations(reductionDims);

    // Build all valid layout combinations
    const layoutPermutations: LayoutPermutation[] = [];

    for (const outputPerm of freeOutputPerms) {
        for (const redPerm of reductionPerms) {
            const loopOrder = getLoopOrder(outputPerm, dim, redPerm);
            const tensorLayouts: { [name: string]: TensorLayout } = {};

            // Check if this permutation matches the natural einsum ordering
            const isNatural = isNaturalDimOrder(outputPerm, outputIndices, dim)
                && JSON.stringify(redPerm) === JSON.stringify(reductionDims);

            // All tensors (inputs + output)
            const allTensors = [...contraction.inputs, contraction.output];

            for (const tensor of allTensors) {
                const isBroadcast = option.broadcastTensors.includes(tensor.name);
                const accessType: 'vector' | 'broadcast' = isBroadcast ? 'broadcast' : 'vector';
                const fixedDim = getFixedContiguousDim(tensor, contraction, dim, option);

                // Get the free dims for this tensor (all dims except the fixed one)
                const tensorFreeDims = tensor.indices.filter(d => d !== fixedDim);

                // Order free dims according to the current loop nesting
                // The loop order gives us the desired ordering for free dims
                const orderedFreeDims = loopOrder.filter(d => tensorFreeDims.includes(d));

                // Add any tensor dims not in the loop order (shouldn't happen but safety)
                const remainingDims = tensorFreeDims.filter(d => !orderedFreeDims.includes(d));
                const finalFreeDimOrder = [...orderedFreeDims, ...remainingDims];

                const layout = buildTensorLayout(tensor, fixedDim, finalFreeDimOrder, accessType);
                tensorLayouts[tensor.name] = layout;
            }

            // Build description
            const loopDesc = loopOrder.map((d, i) => {
                const isRed = reductionDims.includes(d);
                return `${d}${isRed ? '(red)' : ''}`;
            }).join(' → ');

            const permId = `perm-${outputPerm.join('')}-${redPerm.join('')}`;

            layoutPermutations.push({
                id: permId,
                tensorLayouts,
                isNaturalOrder: isNatural,
                description: `Loop: ${loopDesc} → vec(${dim})`,
                loopOrder
            });
        }
    }

    // If no free dims at all, ensure we have at least one layout
    if (layoutPermutations.length === 0) {
        const tensorLayouts: { [name: string]: TensorLayout } = {};
        const allTensors = [...contraction.inputs, contraction.output];

        for (const tensor of allTensors) {
            const isBroadcast = option.broadcastTensors.includes(tensor.name);
            const accessType: 'vector' | 'broadcast' = isBroadcast ? 'broadcast' : 'vector';
            const fixedDim = getFixedContiguousDim(tensor, contraction, dim, option);
            const tensorFreeDims = tensor.indices.filter(d => d !== fixedDim);

            const layout = buildTensorLayout(tensor, fixedDim, tensorFreeDims, accessType);
            tensorLayouts[tensor.name] = layout;
        }

        layoutPermutations.push({
            id: 'perm-default',
            tensorLayouts,
            isNaturalOrder: true,
            description: `Loop: ${reductionDims.join(' → ')} → vec(${dim})`,
            loopOrder: [...reductionDims]
        });
    }

    const totalPermutations = layoutPermutations.length;
    const isUniquelyDetermined = totalPermutations === 1;

    return {
        vectorizeDimension: dim,
        vectorType: isReduction ? 'reduction' : 'output',
        layoutPermutations,
        freeOutputDims,
        reductionDims,
        totalPermutations,
        isUniquelyDetermined
    };
}

/**
 * Get a human-readable description of the transpose operation
 */
export function getTransposeDescription(
    original: string[],
    transposed: string[]
): string {
    if (JSON.stringify(original) === JSON.stringify(transposed)) {
        return 'No transpose';
    }
    return `[${original.join(',')}] → [${transposed.join(',')}]`;
}

/**
 * Check if a specific tensor access pattern is contiguous for vectorization
 */
export function isAccessContiguous(
    tensorIndices: string[],
    accessedDimension: string,
    isVector: boolean
): { contiguous: boolean; reason: string } {
    if (!isVector) {
        return { contiguous: true, reason: 'Scalar/broadcast access' };
    }

    const dimPos = tensorIndices.indexOf(accessedDimension);
    if (dimPos === -1) {
        return { contiguous: true, reason: 'Dimension not in tensor' };
    }

    if (dimPos === tensorIndices.length - 1) {
        return { contiguous: true, reason: `${accessedDimension} is contiguous (last index)` };
    }

    return {
        contiguous: false,
        reason: `${accessedDimension} is at position ${dimPos}, need transpose to make contiguous`
    };
}
