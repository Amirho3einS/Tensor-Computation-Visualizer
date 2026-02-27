// Parse and analyze einsum expressions for tensor contractions

import { ContractionSpec, TensorSpec, VectorizationOption, DimensionSizes } from '../types';

/**
 * Parse an einsum string like "ik,kj->ij" into structured form
 */
export function parseEinsum(einsum: string): ContractionSpec | null {
    try {
        const [inputPart, outputPart] = einsum.split('->').map(s => s.trim());
        if (!inputPart || !outputPart) return null;

        const inputSpecs = inputPart.split(',').map(s => s.trim());
        if (inputSpecs.length < 1 || inputSpecs.length > 2) return null; // Support 1-2 inputs

        const outputIndices = outputPart.split('');

        // Gather all indices from inputs
        const allInputIndices = new Set<string>();
        inputSpecs.forEach(spec => {
            spec.split('').forEach(idx => allInputIndices.add(idx));
        });

        // Reduction indices = appear in inputs but not in output
        const outputSet = new Set(outputIndices);
        const reductionIndices = [...allInputIndices].filter(idx => !outputSet.has(idx));

        // Build tensor specs
        const tensorNames = ['A', 'B', 'C', 'D']; // Max 4 inputs
        const inputs: TensorSpec[] = inputSpecs.map((spec, i) => ({
            name: tensorNames[i],
            indices: spec.split(''),
            shape: [] // Will be filled based on dimension sizes
        }));

        const output: TensorSpec = {
            name: 'Y',
            indices: outputIndices,
            shape: []
        };

        return {
            inputs,
            output,
            reductionIndices,
            outputIndices
        };
    } catch {
        return null;
    }
}

/**
 * Determine valid vectorization options for a contraction
 * Based on the theorem: only output dimensions can be vectorized,
 * and tensors that don't have that dimension must be broadcast
 */
export function getVectorizationOptions(contraction: ContractionSpec): VectorizationOption[] {
    const options: VectorizationOption[] = [];
    const hasReduction = contraction.reductionIndices.length > 0;
    const isUnary = contraction.inputs.length === 1;

    for (const dim of contraction.outputIndices) {
        const broadcastTensors: string[] = [];

        for (const input of contraction.inputs) {
            if (!input.indices.includes(dim)) {
                broadcastTensors.push(input.name);
            }
        }

        let description: string;
        if (isUnary) {
            description = `Vectorize along '${dim}' — unary element-wise`;
        } else if (broadcastTensors.length === 0 && !hasReduction) {
            description = `Vectorize along '${dim}' — element-wise, all inputs aligned`;
        } else if (broadcastTensors.length === 0) {
            description = `Vectorize along '${dim}' — no broadcast needed (batch dimension)`;
        } else {
            description = `Vectorize along '${dim}' — broadcast ${broadcastTensors.join(', ')}`;
        }

        options.push({
            dimension: dim,
            broadcastTensors,
            description
        });
    }

    return options;
}

/**
 * Apply dimension sizes to a tensor spec
 */
export function applyDimensionSizes(tensor: TensorSpec, sizes: DimensionSizes): TensorSpec {
    return {
        ...tensor,
        shape: tensor.indices.map(idx => sizes[idx] || 4) // Default size 4
    };
}

/**
 * Get all unique indices from a contraction (for dimension size UI)
 */
export function getAllIndices(contraction: ContractionSpec): string[] {
    const indices = new Set<string>();
    contraction.inputs.forEach(t => t.indices.forEach(i => indices.add(i)));
    contraction.output.indices.forEach(i => indices.add(i));
    return [...indices].sort();
}

/**
 * Preset einsum expressions for common operations
 */
export const EINSUM_PRESETS: { label: string; einsum: string; description: string }[] = [
    // Contractions (k > 0)
    { label: 'GEMM', einsum: 'ik,kj->ij', description: 'Matrix-Matrix Multiply' },
    { label: 'GEMV', einsum: 'ik,k->i', description: 'Matrix-Vector Multiply' },
    { label: 'Outer Product', einsum: 'i,j->ij', description: 'Vector Outer Product' },
    { label: 'Batched GEMM', einsum: 'bik,bkj->bij', description: 'Batched Matrix Multiply' },
    { label: 'Batched GEMV', einsum: 'bik,bk->bi', description: 'Batched Matrix-Vector' },
    { label: 'Attention QK^T', einsum: 'bhqd,bhkd->bhqk', description: 'Query-Key Attention' },
    // Element-wise binary (k = 0, T ≥ 2)
    { label: 'Vector Add', einsum: 'i,i->i', description: 'Element-wise vector addition (residual / skip connection)' },
    { label: 'Vector Sub', einsum: 'i,i->i', description: 'Element-wise vector subtraction' },
    // Unary (k = 0, T = 1)
    { label: 'ReLU / Unary', einsum: 'i->i', description: 'Element-wise unary (ReLU, sigmoid, etc.)' },
];

/**
 * Get default dimension sizes for a contraction
 */
export function getDefaultDimensionSizes(contraction: ContractionSpec): DimensionSizes {
    const sizes: DimensionSizes = {};
    const allIndices = getAllIndices(contraction);

    // Heuristics for common dimension names
    for (const idx of allIndices) {
        switch (idx) {
            case 'b': sizes[idx] = 2; break;  // batch
            case 'h': sizes[idx] = 2; break;  // heads
            case 'q': sizes[idx] = 4; break;  // query length
            case 'k': sizes[idx] = 4; break;  // key length / reduction
            case 'd': sizes[idx] = 4; break;  // embedding dim
            case 'i': sizes[idx] = 4; break;  // row
            case 'j': sizes[idx] = 4; break;  // column
            case 'm': sizes[idx] = 4; break;  // row
            case 'n': sizes[idx] = 4; break;  // column
            default: sizes[idx] = 4;
        }
    }

    return sizes;
}

/**
 * Validate that dimension sizes are consistent across tensors
 */
export function validateDimensionSizes(contraction: ContractionSpec, sizes: DimensionSizes): string | null {
    for (const idx of getAllIndices(contraction)) {
        if (!sizes[idx] || sizes[idx] < 1) {
            return `Dimension '${idx}' must have a positive size`;
        }
    }
    return null; // Valid
}
