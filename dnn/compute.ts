import type { TensorData } from '../types';
import {
  createZeroTensor,
  generateRandomTensor,
  getTensorSize,
} from '../utils/tensorData';
import type { ForwardBackwardResult, LayerCache, LayerRuntime, LayerSpec, NetworkSpec } from './types';

// --- flat index helpers (row-major, last dim fastest) ---

export function flat3(b: number, c: number, l: number, B: number, C: number, L: number): number {
  return ((b * C + c) * L + l);
}

export function flat3W(co: number, ci: number, k: number, Cin: number, K: number): number {
  return ((co * Cin + ci) * K + k);
}

export function flat4(b: number, c: number, h: number, w: number, C: number, H: number, W: number): number {
  return (((b * C + c) * H + h) * W + w);
}

export function flat4W(
  co: number,
  ci: number,
  kh: number,
  kw: number,
  Cin: number,
  Kh: number,
  Kw: number
): number {
  return (((co * Cin + ci) * Kh + kh) * Kw + kw);
}

export function flat2(b: number, i: number, I: number): number {
  return b * I + i;
}

export function flat2W(o: number, i: number, Din: number): number {
  return o * Din + i;
}

export function outputShapeFor(spec: LayerSpec, inShape: number[]): number[] {
  if (spec.kind === 'conv1d') {
    const [, , Lin] = inShape;
    if (inShape.length !== 3) throw new Error('Conv1D expects input shape [B,C,L]');
    const Lout = Lin - spec.kernelSize + 1;
    return [inShape[0], spec.outChannels, Lout];
  }
  if (spec.kind === 'conv2d') {
    const [, , Hin, Win] = inShape;
    if (inShape.length !== 4) throw new Error('Conv2D expects input shape [B,C,H,W]');
    const Hout = Hin - spec.kernelH + 1;
    const Wout = Win - spec.kernelW + 1;
    return [inShape[0], spec.outChannels, Hout, Wout];
  }
  if (spec.kind === 'relu') {
    return [...inShape];
  }
  if (spec.kind === 'flatten') {
    if (inShape.length < 2) throw new Error('Flatten expects at least [B,*]');
    return [inShape[0], inShape.slice(1).reduce((acc, v) => acc * v, 1)];
  }
  if (spec.kind === 'linear') {
    if (inShape.length !== 2) throw new Error('Linear expects [B,Din]');
    return [inShape[0], spec.outFeatures];
  }
  return inShape;
}

export function validateNetwork(spec: NetworkSpec): string | null {
  if (!spec.layers.length) return 'Add at least one layer';
  let shape: number[];
  if (spec.inputShape.mode === '1d') {
    const { B, C, L } = spec.inputShape;
    if (B < 1 || C < 1 || L < 1) return 'Input dimensions must be positive';
    shape = [B, C, L];
  } else {
    const { B, C, H, W } = spec.inputShape;
    if (B < 1 || C < 1 || H < 1 || W < 1) return 'Input dimensions must be positive';
    shape = [B, C, H, W];
  }
  for (let i = 0; i < spec.layers.length; i++) {
    const layer = spec.layers[i];
    try {
      if (layer.kind === 'conv1d') {
        const Lin = shape[2];
        if (shape.length !== 3) return `Layer ${i + 1} (Conv1D): expected 3D input [B,C,L]`;
        if (layer.kernelSize > Lin) return `Layer ${i + 1}: kernel ${layer.kernelSize} > length ${Lin}`;
        if (Lin - layer.kernelSize + 1 < 1) return `Layer ${i + 1}: output length would be < 1`;
      }
      if (layer.kind === 'conv2d') {
        if (shape.length !== 4) return `Layer ${i + 1} (Conv2D): expected 4D input [B,C,H,W]`;
        const Hin = shape[2];
        const Win = shape[3];
        if (layer.kernelH > Hin) return `Layer ${i + 1}: kernelH ${layer.kernelH} > height ${Hin}`;
        if (layer.kernelW > Win) return `Layer ${i + 1}: kernelW ${layer.kernelW} > width ${Win}`;
        if (Hin - layer.kernelH + 1 < 1 || Win - layer.kernelW + 1 < 1) {
          return `Layer ${i + 1}: output spatial dims would be < 1`;
        }
      }
      if (layer.kind === 'linear' && shape.length !== 2) {
        return `Layer ${i + 1} (Linear): expected 2D input [B,Din] (add Flatten before Linear if needed)`;
      }
      if (layer.kind === 'flatten' && shape.length < 3) {
        return `Layer ${i + 1} (Flatten): expected at least 3D input [B,...]`;
      }
      shape = outputShapeFor(layer, shape);
    } catch (e) {
      return e instanceof Error ? e.message : String(e);
    }
  }
  return null;
}

export function buildRuntimes(spec: NetworkSpec): LayerRuntime[] {
  const err = validateNetwork(spec);
  if (err) throw new Error(err);
  let shape =
    spec.inputShape.mode === '1d'
      ? [spec.inputShape.B, spec.inputShape.C, spec.inputShape.L]
      : [spec.inputShape.B, spec.inputShape.C, spec.inputShape.H, spec.inputShape.W];
  const runtimes: LayerRuntime[] = [];
  for (const layerSpec of spec.layers) {
    const inputShape = [...shape];
    const outputShape = outputShapeFor(layerSpec, shape);
    const rt: LayerRuntime = { spec: layerSpec, inputShape, outputShape };
    if (layerSpec.kind === 'conv1d') {
      const [, Cin, Lin] = inputShape;
      const K = layerSpec.kernelSize;
      const Cout = layerSpec.outChannels;
      rt.weights = generateRandomTensor([Cout, Cin, K]);
      rt.bias = generateRandomTensor([Cout]);
    } else if (layerSpec.kind === 'conv2d') {
      const [, Cin] = inputShape;
      const Kh = layerSpec.kernelH;
      const Kw = layerSpec.kernelW;
      const Cout = layerSpec.outChannels;
      rt.weights = generateRandomTensor([Cout, Cin, Kh, Kw]);
      rt.bias = generateRandomTensor([Cout]);
    } else if (layerSpec.kind === 'linear') {
      const Din = inputShape[1];
      const Dout = layerSpec.outFeatures;
      rt.weights = generateRandomTensor([Dout, Din]);
      rt.bias = generateRandomTensor([Dout]);
    }
    runtimes.push(rt);
    shape = outputShape;
  }
  return runtimes;
}

function forwardLayer(
  spec: LayerSpec,
  X: TensorData,
  inShape: number[],
  outShape: number[],
  weights?: TensorData,
  bias?: TensorData
): { Y: TensorData; cache: LayerCache } {
  const Y = createZeroTensor(outShape);

  if (spec.kind === 'conv1d') {
    const [B, Cin, Lin] = inShape;
    const [, Cout, Lout] = outShape;
    const K = spec.kernelSize;
    if (!weights || !bias) throw new Error('Conv1D missing weights');
    for (let b = 0; b < B; b++) {
      for (let co = 0; co < Cout; co++) {
        for (let lo = 0; lo < Lout; lo++) {
          let acc = bias[co] ?? 0;
          for (let ci = 0; ci < Cin; ci++) {
            for (let k = 0; k < K; k++) {
              const li = lo + k;
              const w = weights[flat3W(co, ci, k, Cin, K)];
              const xv = X[flat3(b, ci, li, B, Cin, Lin)];
              acc += w * xv;
            }
          }
          Y[flat3(b, co, lo, B, Cout, Lout)] = acc;
        }
      }
    }
    return {
      Y,
      cache: {
        kind: 'conv1d',
        X: [...X],
        Y: [...Y],
        conv: { B, Cin, Cout: Cout, K, Lin, Lout },
      },
    };
  }

  if (spec.kind === 'conv2d') {
    const [B, Cin, Hin, Win] = inShape;
    const [, Cout, Hout, Wout] = outShape;
    const Kh = spec.kernelH;
    const Kw = spec.kernelW;
    if (!weights || !bias) throw new Error('Conv2D missing weights');
    for (let b = 0; b < B; b++) {
      for (let co = 0; co < Cout; co++) {
        for (let ho = 0; ho < Hout; ho++) {
          for (let wo = 0; wo < Wout; wo++) {
            let acc = bias[co] ?? 0;
            for (let ci = 0; ci < Cin; ci++) {
              for (let kh = 0; kh < Kh; kh++) {
                for (let kw = 0; kw < Kw; kw++) {
                  const hi = ho + kh;
                  const wi = wo + kw;
                  acc +=
                    (weights[flat4W(co, ci, kh, kw, Cin, Kh, Kw)] ?? 0) *
                    (X[flat4(b, ci, hi, wi, Cin, Hin, Win)] ?? 0);
                }
              }
            }
            Y[flat4(b, co, ho, wo, Cout, Hout, Wout)] = acc;
          }
        }
      }
    }
    return {
      Y,
      cache: {
        kind: 'conv2d',
        X: [...X],
        Y: [...Y],
        conv2d: { B, Cin, Cout, Kh, Kw, Hin, Win, Hout, Wout },
      },
    };
  }

  if (spec.kind === 'relu') {
    const mask = createZeroTensor(inShape);
    const n = X.length;
    for (let i = 0; i < n; i++) {
      const v = X[i] ?? 0;
      const m = v > 0 ? 1 : 0;
      mask[i] = m;
      Y[i] = v * m;
    }
    return {
      Y,
      cache: { kind: 'relu', X: [...X], Y: [...Y], reluMask: mask },
    };
  }

  if (spec.kind === 'flatten') {
    const n = inShape.reduce((acc, v) => acc * v, 1);
    for (let i = 0; i < n; i++) Y[i] = X[i] ?? 0;
    return {
      Y,
      cache: { kind: 'flatten', X: [...X], Y: [...Y], flatten: { shape: [...inShape] } },
    };
  }

  if (spec.kind === 'linear') {
    const [B, Din] = inShape;
    const [, Dout] = outShape;
    if (!weights || !bias) throw new Error('Linear missing weights');
    for (let b = 0; b < B; b++) {
      for (let o = 0; o < Dout; o++) {
        let acc = bias[o] ?? 0;
        for (let i = 0; i < Din; i++) {
          acc += (weights[flat2W(o, i, Din)] ?? 0) * (X[flat2(b, i, Din)] ?? 0);
        }
        Y[flat2(b, o, Dout)] = acc;
      }
    }
    return {
      Y,
      cache: {
        kind: 'linear',
        X: [...X],
        Y: [...Y],
        linear: { B, Din, Dout },
      },
    };
  }

  throw new Error(`Unknown layer ${(spec as LayerSpec).kind}`);
}

function backwardLayer(
  spec: LayerSpec,
  cache: LayerCache,
  dY: TensorData,
  weights?: TensorData
): { dX: TensorData; dW?: TensorData; db?: TensorData } {
  if (spec.kind === 'conv1d' && cache.conv) {
    const { B, Cin, Cout, K, Lin, Lout } = cache.conv;
    if (!weights) throw new Error('Conv1D backward needs weights');
    const dX = createZeroTensor([B, Cin, Lin]);
    const dW = createZeroTensor([Cout, Cin, K]);
    const db = createZeroTensor([Cout]);
    const X = cache.X;

    for (let b = 0; b < B; b++) {
      for (let co = 0; co < Cout; co++) {
        for (let lo = 0; lo < Lout; lo++) {
          const dy = dY[flat3(b, co, lo, B, Cout, Lout)];
          db[co] = (db[co] ?? 0) + dy;
          for (let ci = 0; ci < Cin; ci++) {
            for (let k = 0; k < K; k++) {
              const li = lo + k;
              const w = weights[flat3W(co, ci, k, Cin, K)];
              dW[flat3W(co, ci, k, Cin, K)] += dy * (X[flat3(b, ci, li, B, Cin, Lin)] ?? 0);
              dX[flat3(b, ci, li, B, Cin, Lin)] += dy * w;
            }
          }
        }
      }
    }
    return { dX, dW, db };
  }

  if (spec.kind === 'conv2d' && cache.conv2d) {
    const { B, Cin, Cout, Kh, Kw, Hin, Win, Hout, Wout } = cache.conv2d;
    if (!weights) throw new Error('Conv2D backward needs weights');
    const dX = createZeroTensor([B, Cin, Hin, Win]);
    const dW = createZeroTensor([Cout, Cin, Kh, Kw]);
    const db = createZeroTensor([Cout]);
    const X = cache.X;

    for (let b = 0; b < B; b++) {
      for (let co = 0; co < Cout; co++) {
        for (let ho = 0; ho < Hout; ho++) {
          for (let wo = 0; wo < Wout; wo++) {
            const dy = dY[flat4(b, co, ho, wo, Cout, Hout, Wout)] ?? 0;
            db[co] = (db[co] ?? 0) + dy;
            for (let ci = 0; ci < Cin; ci++) {
              for (let kh = 0; kh < Kh; kh++) {
                for (let kw = 0; kw < Kw; kw++) {
                  const hi = ho + kh;
                  const wi = wo + kw;
                  const wFlat = flat4W(co, ci, kh, kw, Cin, Kh, Kw);
                  dW[wFlat] += dy * (X[flat4(b, ci, hi, wi, Cin, Hin, Win)] ?? 0);
                  dX[flat4(b, ci, hi, wi, Cin, Hin, Win)] += dy * (weights[wFlat] ?? 0);
                }
              }
            }
          }
        }
      }
    }
    return { dX, dW, db };
  }

  if (spec.kind === 'relu' && cache.reluMask) {
    const n = dY.length;
    const dX = new Array(n).fill(0);
    for (let i = 0; i < n; i++) {
      dX[i] = (dY[i] ?? 0) * (cache.reluMask[i] ?? 0);
    }
    return { dX };
  }

  if (spec.kind === 'flatten' && cache.flatten) {
    const dX = createZeroTensor(cache.flatten.shape);
    for (let i = 0; i < dY.length; i++) dX[i] = dY[i] ?? 0;
    return { dX };
  }

  if (spec.kind === 'linear' && cache.linear) {
    const { B, Din, Dout } = cache.linear;
    if (!weights) throw new Error('Linear backward needs weights');
    const dX = createZeroTensor([B, Din]);
    const dW = createZeroTensor([Dout, Din]);
    const db = createZeroTensor([Dout]);
    const X = cache.X;

    for (let o = 0; o < Dout; o++) {
      for (let i = 0; i < Din; i++) {
        let sum = 0;
        for (let b = 0; b < B; b++) {
          const dy = dY[flat2(b, o, Dout)] ?? 0;
          sum += dy * (X[flat2(b, i, Din)] ?? 0);
        }
        dW[flat2W(o, i, Din)] = sum;
      }
    }
    for (let o = 0; o < Dout; o++) {
      let s = 0;
      for (let b = 0; b < B; b++) s += dY[flat2(b, o, Dout)] ?? 0;
      db[o] = s;
    }
    for (let b = 0; b < B; b++) {
      for (let i = 0; i < Din; i++) {
        let s = 0;
        for (let o = 0; o < Dout; o++) {
          s += (dY[flat2(b, o, Dout)] ?? 0) * (weights[flat2W(o, i, Din)] ?? 0);
        }
        dX[flat2(b, i, Din)] = s;
      }
    }
    return { dX, dW, db };
  }

  throw new Error('backwardLayer: unsupported cache/spec');
}

/**
 * Run forward then backward with MSE loss on final output.
 */
export function forwardBackward(
  runtimes: LayerRuntime[],
  input: TensorData
): ForwardBackwardResult {
  const n = runtimes.length;
  const caches: LayerCache[] = [];
  const activations: TensorData[] = [];
  activations.push([...input]);

  let h = [...input];
  for (let i = 0; i < n; i++) {
    const rt = runtimes[i];
    const { Y, cache } = forwardLayer(rt.spec, h, rt.inputShape, rt.outputShape, rt.weights, rt.bias);
    caches.push(cache);
    activations.push([...Y]);
    h = Y;
  }

  const finalY = activations[activations.length - 1];
  const N = Math.max(1, finalY.length);
  const target = generateRandomTensor([getTensorSize(runtimes[n - 1].outputShape)]);
  const dYLast = finalY.map((v, i) => (2 * (v - (target[i] ?? 0))) / N);

  const gradOutputs: TensorData[] = new Array(n + 1);
  gradOutputs[n] = dYLast;

  const gradWeights: (TensorData | undefined)[] = runtimes.map(() => undefined);
  const gradBiases: (TensorData | undefined)[] = runtimes.map(() => undefined);

  let dH = dYLast;
  for (let i = n - 1; i >= 0; i--) {
    const rt = runtimes[i];
    const { dX, dW, db } = backwardLayer(rt.spec, caches[i], dH, rt.weights);
    gradOutputs[i] = dX;
    if (dW) gradWeights[i] = dW;
    if (db) gradBiases[i] = db;
    dH = dX;
  }

  return {
    runtimes,
    caches,
    activations,
    gradOutputs,
    gradWeights,
    gradBiases,
    target,
  };
}

export function randomInput(spec: NetworkSpec): TensorData {
  if (spec.inputShape.mode === '1d') {
    const { B, C, L } = spec.inputShape;
    return generateRandomTensor([B, C, L]);
  }
  const { B, C, H, W } = spec.inputShape;
  return generateRandomTensor([B, C, H, W]);
}
