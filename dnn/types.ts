import type { TensorData } from '../types';

export type LayerSpec =
  | { kind: 'conv1d'; outChannels: number; kernelSize: number }
  | { kind: 'conv2d'; outChannels: number; kernelH: number; kernelW: number }
  | { kind: 'linear'; outFeatures: number }
  | { kind: 'relu' }
  | { kind: 'flatten' };

export interface NetworkSpec {
  inputShape:
    | { mode: '1d'; B: number; C: number; L: number }
    | { mode: '2d'; B: number; C: number; H: number; W: number };
  layers: LayerSpec[];
}

export interface LayerRuntime {
  spec: LayerSpec;
  /** Shape of tensor entering this layer (row-major semantics in compute helpers) */
  inputShape: number[];
  /** Shape of tensor leaving this layer */
  outputShape: number[];
  weights?: TensorData;
  bias?: TensorData;
}

/** Per-layer cache for backward (only fields used by that layer kind) */
export interface LayerCache {
  kind: LayerSpec['kind'];
  /** Input activation to this layer (flat) */
  X: TensorData;
  /** Output activation (flat) */
  Y: TensorData;
  /** For conv1d: Cin, Cout, K, Lin, Lout, B */
  conv?: {
    B: number;
    Cin: number;
    Cout: number;
    K: number;
    Lin: number;
    Lout: number;
  };
  /** For conv2d: Cin, Cout, Kh, Kw, Hin, Win, Hout, Wout, B */
  conv2d?: {
    B: number;
    Cin: number;
    Cout: number;
    Kh: number;
    Kw: number;
    Hin: number;
    Win: number;
    Hout: number;
    Wout: number;
  };
  /** For linear: B, Din, Dout */
  linear?: {
    B: number;
    Din: number;
    Dout: number;
  };
  /** For relu: same shape as X */
  reluMask?: TensorData;
  /** For flatten: shape before flatten */
  flatten?: { shape: number[] };
}

export interface ForwardBackwardResult {
  runtimes: LayerRuntime[];
  caches: LayerCache[];
  /** h[0] = network input, h[i+1] = output of layer i */
  activations: TensorData[];
  /** Gradient w.r.t. output of each layer (same shape as activations[i+1]) */
  gradOutputs: TensorData[];
  /** Gradient w.r.t. weights (same shape as weights) */
  gradWeights: (TensorData | undefined)[];
  /** Gradient w.r.t. bias */
  gradBiases: (TensorData | undefined)[];
  /** MSE target for final output */
  target: TensorData;
}
