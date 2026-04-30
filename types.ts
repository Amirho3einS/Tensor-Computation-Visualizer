// Tensor contraction types for generalized einsum-style operations

export interface TensorSpec {
  name: string;           // e.g., "A", "B", "C"
  indices: string[];      // e.g., ["i", "k"] for A[i,k]
  shape: number[];        // e.g., [4, 8] meaning dim i=4, dim k=8
}

export interface ContractionSpec {
  inputs: TensorSpec[];           // Input tensors (2 for binary contractions)
  output: TensorSpec;             // Output tensor
  reductionIndices: string[];     // Indices summed over
  outputIndices: string[];        // Indices in output
}

export interface VectorizationOption {
  dimension: string;              // Which output dimension to vectorize along
  broadcastTensors: string[];     // Which tensors need to be broadcast
  description: string;            // Human-readable explanation
}

export interface DimensionSizes {
  [index: string]: number;        // Maps index letter to size, e.g., { i: 4, j: 8, k: 16 }
}

/** Optional metadata when a step is used by the DNN visualizer */
export interface DNNStepMeta {
  layerIdx: number;
  phase: 'forward' | 'backward';
  subPhase?: 'dX' | 'dW' | 'db';
}

export interface OperationStep {
  id: number;
  description: string;
  formula: string;
  // For each tensor, define which cells are accessed
  highlights: {
    [tensorName: string]: {
      // For each index of tensor, either a fixed value or a range [start, end]
      indices: { [indexName: string]: number | [number, number] };
      type: 'vector' | 'scalar-broadcast' | 'output';
    };
  };
  dnnMeta?: DNNStepMeta;
}

export interface AppState {
  einsum: string;                         // e.g., "ik,kj->ij"
  dimensionSizes: DimensionSizes;         // Sizes for each index
  vectorWidth: number;                    // Hardware vector width (w)
  selectedVectorization: string | null;   // Which output dim is selected for vectorization
  steps: OperationStep[];
  currentStepIndex: number;
}

// Tensor data for output verification

export type TensorData = number[];  // Flat array, row-major order

export interface TensorDataMap {
  [tensorName: string]: TensorData;
}

// Verification result
export interface VerificationResult {
  passed: boolean;
  maxError: number;
  totalElements: number;
  mismatchCount: number;
}
