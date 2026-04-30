import type { DimensionSizes, OperationStep } from '../types';
import { parseEinsum, getVectorizationOptions } from '../utils/einsumParser';
import { generateSteps } from '../utils/stepGenerator';
import type { LayerRuntime, LayerSpec } from './types';

const DNN_VECTOR_WIDTH = 2;

function withMeta(
  step: OperationStep,
  layerIdx: number,
  phase: 'forward' | 'backward',
  subPhase?: 'dX' | 'dW' | 'db'
): OperationStep {
  return { ...step, dnnMeta: { layerIdx, phase, subPhase } };
}

function renumberSteps(steps: OperationStep[]): OperationStep[] {
  return steps.map((s, i) => ({ ...s, id: i }));
}

/** Rename tensor keys in highlights (e.g. A→dY, B→X, Y→dW) */
function renameHighlights(
  step: OperationStep,
  map: Record<string, string>
): OperationStep['highlights'] {
  const out: OperationStep['highlights'] = {};
  for (const [k, v] of Object.entries(step.highlights)) {
    const nk = map[k] ?? k;
    out[nk] = v;
  }
  return out;
}

function generateLinearStepsFromEinsum(
  einsum: string,
  sizes: DimensionSizes,
  layerIdx: number,
  phase: 'forward' | 'backward',
  subPhase: 'dX' | 'dW' | 'db' | undefined,
  rename: Record<string, string>
): OperationStep[] {
  const parsed = parseEinsum(einsum);
  if (!parsed) return [];
  const options = getVectorizationOptions(parsed);
  const vecDim = options[0]?.dimension ?? parsed.outputIndices[0];
  const option = options.find(o => o.dimension === vecDim) ?? options[0];
  if (!option) return [];
  const raw = generateSteps(parsed, sizes, vecDim, DNN_VECTOR_WIDTH, option);
  return renumberSteps(
    raw.map(s =>
      withMeta(
        {
          ...s,
          highlights: renameHighlights(s, rename),
        },
        layerIdx,
        phase,
        subPhase
      )
    )
  );
}

export interface LayerStepBundle {
  forward: OperationStep[];
  backward: {
    dX: OperationStep[];
    dW?: OperationStep[];
    db?: OperationStep[];
  };
}

function* iterate3D(B: number, C: number, L: number): Generator<{ b: number; c: number; l: number }> {
  for (let b = 0; b < B; b++) {
    for (let c = 0; c < C; c++) {
      for (let l = 0; l < L; l++) {
        yield { b, c, l };
      }
    }
  }
}

function* iterate4D(
  B: number,
  C: number,
  H: number,
  W: number
): Generator<{ b: number; c: number; h: number; w: number }> {
  for (let b = 0; b < B; b++) {
    for (let c = 0; c < C; c++) {
      for (let h = 0; h < H; h++) {
        for (let w = 0; w < W; w++) {
          yield { b, c, h, w };
        }
      }
    }
  }
}

export function generateLayerSteps(layerIdx: number, rt: LayerRuntime): LayerStepBundle {
  const spec = rt.spec;
  const emptyBackward = { dX: [] as OperationStep[], dW: undefined as OperationStep[] | undefined, db: undefined as OperationStep[] | undefined };

  if (spec.kind === 'conv1d') {
    const [B, Cin, Lin] = rt.inputShape;
    const [, Cout, Lout] = rt.outputShape;
    const K = spec.kernelSize;

    const forward: OperationStep[] = [];
    let id = 0;
    for (let b = 0; b < B; b++) {
      for (let co = 0; co < Cout; co++) {
        for (let lo = 0; lo < Lout; lo++) {
          forward.push(
            withMeta(
              {
                id: id++,
                description: `Conv1D forward: accumulate into Y[${b},${co},${lo}]`,
                formula: `Y[${b},${co},${lo}] = b[${co}] + sum_{c,k} W[${co},c,k]*X[${b},c,${lo}+k]`,
                highlights: {
                  X: {
                    indices: { b, c: [0, Cin - 1], l: [lo, lo + K - 1] },
                    type: 'vector',
                  },
                  W: {
                    indices: { o: co, c: [0, Cin - 1], k: [0, K - 1] },
                    type: 'vector',
                  },
                  bias: {
                    indices: { o: co },
                    type: 'scalar-broadcast',
                  },
                  Y: {
                    indices: { b, o: co, l: lo },
                    type: 'output',
                  },
                },
              },
              layerIdx,
              'forward',
              undefined
            )
          );
        }
      }
    }

    const dW: OperationStep[] = [];
    id = 0;
    for (let co = 0; co < Cout; co++) {
      for (let ci = 0; ci < Cin; ci++) {
        for (let k = 0; k < K; k++) {
          dW.push(
            withMeta(
              {
                id: id++,
                description: `Conv1D ∂L/∂W[${co},${ci},${k}]`,
                formula: `dW[${co},${ci},${k}] = sum_{b,l} dY[b,${co},l]*X[b,${ci},l+${k}]`,
                highlights: {
                  dY: {
                    indices: { b: [0, B - 1], o: co, l: [0, Lout - 1] },
                    type: 'vector',
                  },
                  X: {
                    indices: { b: [0, B - 1], c: ci, l: [k, k + Lout - 1] },
                    type: 'vector',
                  },
                  dW: {
                    indices: { o: co, c: ci, k },
                    type: 'output',
                  },
                },
              },
              layerIdx,
              'backward',
              'dW'
            )
          );
        }
      }
    }

    const dX: OperationStep[] = [];
    id = 0;
    for (let b = 0; b < B; b++) {
      for (let ci = 0; ci < Cin; ci++) {
        for (let li = 0; li < Lin; li++) {
          const contributors: { co: number; k: number; lo: number }[] = [];
          for (let co = 0; co < Cout; co++) {
            for (let k = 0; k < K; k++) {
              const lo = li - k;
              if (lo >= 0 && lo < Lout) contributors.push({ co, k, lo });
            }
          }
          if (contributors.length === 0) continue;
          const { co, k, lo } = contributors[0];
          dX.push(
            withMeta(
              {
                id: id++,
                description: `Conv1D ∂L/∂X[${b},${ci},${li}] (showing term co=${co}, k=${k})`,
                formula: `dX[${b},${ci},${li}] += sum_{o,k} dY[b,o,l]*W[o,${ci},k]  with  l=${li}-k`,
                highlights: {
                  dY: { indices: { b, o: co, l: lo }, type: 'vector' },
                  W: { indices: { o: co, c: ci, k }, type: 'vector' },
                  dX: { indices: { b, c: ci, l: li }, type: 'output' },
                },
              },
              layerIdx,
              'backward',
              'dX'
            )
          );
        }
      }
    }

    const db: OperationStep[] = [];
    id = 0;
    for (let co = 0; co < Cout; co++) {
      db.push(
        withMeta(
          {
            id: id++,
            description: `Conv1D ∂L/∂b[${co}]`,
            formula: `db[${co}] = sum_{b,l} dY[b,${co},l]`,
            highlights: {
              dY: {
                indices: { b: [0, B - 1], o: co, l: [0, Lout - 1] },
                type: 'vector',
              },
              db: { indices: { o: co }, type: 'output' },
            },
          },
          layerIdx,
          'backward',
          'db'
        )
      );
    }

    return { forward: renumberSteps(forward), backward: { dX: renumberSteps(dX), dW: renumberSteps(dW), db: renumberSteps(db) } };
  }

  if (spec.kind === 'conv2d') {
    const [B, Cin, Hin, Win] = rt.inputShape;
    const [, Cout, Hout, Wout] = rt.outputShape;
    const Kh = spec.kernelH;
    const Kw = spec.kernelW;

    const forward: OperationStep[] = [];
    let id = 0;
    for (let b = 0; b < B; b++) {
      for (let co = 0; co < Cout; co++) {
        for (let ho = 0; ho < Hout; ho++) {
          for (let wo = 0; wo < Wout; wo++) {
            forward.push(
              withMeta(
                {
                  id: id++,
                  description: `Conv2D forward: accumulate into Y[${b},${co},${ho},${wo}]`,
                  formula: `Y[${b},${co},${ho},${wo}] = b[${co}] + sum_{c,kh,kw} W[${co},c,kh,kw]*X[${b},c,${ho}+kh,${wo}+kw]`,
                  highlights: {
                    X: { indices: { b, c: [0, Cin - 1], h: [ho, ho + Kh - 1], w: [wo, wo + Kw - 1] }, type: 'vector' },
                    W: { indices: { o: co, c: [0, Cin - 1], kh: [0, Kh - 1], kw: [0, Kw - 1] }, type: 'vector' },
                    bias: { indices: { o: co }, type: 'scalar-broadcast' },
                    Y: { indices: { b, o: co, h: ho, w: wo }, type: 'output' },
                  },
                },
                layerIdx,
                'forward'
              )
            );
          }
        }
      }
    }

    const dW: OperationStep[] = [];
    id = 0;
    for (let co = 0; co < Cout; co++) {
      for (let ci = 0; ci < Cin; ci++) {
        for (let kh = 0; kh < Kh; kh++) {
          for (let kw = 0; kw < Kw; kw++) {
            dW.push(
              withMeta(
                {
                  id: id++,
                  description: `Conv2D ∂L/∂W[${co},${ci},${kh},${kw}]`,
                  formula: `dW[${co},${ci},${kh},${kw}] = sum_{b,h,w} dY[b,${co},h,w]*X[b,${ci},h+${kh},w+${kw}]`,
                  highlights: {
                    dY: { indices: { b: [0, B - 1], o: co, h: [0, Hout - 1], w: [0, Wout - 1] }, type: 'vector' },
                    X: { indices: { b: [0, B - 1], c: ci, h: [kh, kh + Hout - 1], w: [kw, kw + Wout - 1] }, type: 'vector' },
                    dW: { indices: { o: co, c: ci, kh, kw }, type: 'output' },
                  },
                },
                layerIdx,
                'backward',
                'dW'
              )
            );
          }
        }
      }
    }

    const dX: OperationStep[] = [];
    id = 0;
    for (const { b, c, h, w } of iterate4D(B, Cin, Hin, Win)) {
      // Valid reduction ranges contributing to this input pixel.
      // ho = h - kh and wo = w - kw must be in output bounds.
      const khMin = Math.max(0, h - (Hout - 1));
      const khMax = Math.min(Kh - 1, h);
      const kwMin = Math.max(0, w - (Wout - 1));
      const kwMax = Math.min(Kw - 1, w);
      const hoMin = Math.max(0, h - (Kh - 1));
      const hoMax = Math.min(Hout - 1, h);
      const woMin = Math.max(0, w - (Kw - 1));
      const woMax = Math.min(Wout - 1, w);
      if (khMin > khMax || kwMin > kwMax || hoMin > hoMax || woMin > woMax) continue;

      dX.push(
        withMeta(
          {
            id: id++,
            description: `Conv2D ∂L/∂X[${b},${c},${h},${w}]`,
            formula: `dX[${b},${c},${h},${w}] += sum_{o,kh,kw} dY[b,o,ho,wo]*W[o,${c},kh,kw]`,
            highlights: {
              // Show all output filters and spatially valid dY window.
              dY: { indices: { b, o: [0, Cout - 1], h: [hoMin, hoMax], w: [woMin, woMax] }, type: 'vector' },
              // Show all filter rows/cols that can contribute for this pixel.
              W: { indices: { o: [0, Cout - 1], c, kh: [khMin, khMax], kw: [kwMin, kwMax] }, type: 'vector' },
              dX: { indices: { b, c, h, w }, type: 'output' },
            },
          },
          layerIdx,
          'backward',
          'dX'
        )
      );
    }

    const db: OperationStep[] = [];
    id = 0;
    for (let o = 0; o < Cout; o++) {
      db.push(
        withMeta(
          {
            id: id++,
            description: `Conv2D ∂L/∂b[${o}]`,
            formula: `db[${o}] = sum_{b,h,w} dY[b,${o},h,w]`,
            highlights: {
              dY: { indices: { b: [0, B - 1], o, h: [0, Hout - 1], w: [0, Wout - 1] }, type: 'vector' },
              db: { indices: { o }, type: 'output' },
            },
          },
          layerIdx,
          'backward',
          'db'
        )
      );
    }

    return {
      forward: renumberSteps(forward),
      backward: { dX: renumberSteps(dX), dW: renumberSteps(dW), db: renumberSteps(db) },
    };
  }

  if (spec.kind === 'linear') {
    const [B, Din] = rt.inputShape;
    const [, Dout] = rt.outputShape;
    // Linear forward visualization policy:
    // one step per output element, highlighting full input/weight vectors.
    const forward: OperationStep[] = [];
    let id = 0;
    for (let b = 0; b < B; b++) {
      for (let o = 0; o < Dout; o++) {
        forward.push(
          withMeta(
            {
              id: id++,
              description: `Linear forward: compute Y[${b},${o}] from full dot product`,
              formula: `Y[${b},${o}] = b[${o}] + sum_i X[${b},i] * W[${o},i]`,
              highlights: {
                X: { indices: { b, i: [0, Din - 1] }, type: 'vector' },
                W: { indices: { o, i: [0, Din - 1] }, type: 'vector' },
                bias: { indices: { o }, type: 'scalar-broadcast' },
                Y: { indices: { b, o }, type: 'output' },
              },
            },
            layerIdx,
            'forward',
            undefined
          )
        );
      }
    }

    const dW: OperationStep[] = [];
    id = 0;
    for (let o = 0; o < Dout; o++) {
      dW.push(
        withMeta(
          {
            id: id++,
            description: `Linear backward dW: compute row dW[${o},:]`,
            formula: `dW[${o},i] = sum_b dY[b,${o}] * X[b,i]`,
            highlights: {
              dY: { indices: { b: [0, B - 1], o }, type: 'vector' },
              X: { indices: { b: [0, B - 1], i: [0, Din - 1] }, type: 'vector' },
              dW: { indices: { o, i: [0, Din - 1] }, type: 'output' },
            },
          },
          layerIdx,
          'backward',
          'dW'
        )
      );
    }

    const dX: OperationStep[] = [];
    id = 0;
    for (let b = 0; b < B; b++) {
      for (let i = 0; i < Din; i++) {
        dX.push(
          withMeta(
            {
              id: id++,
              description: `Linear backward dX: compute dX[${b},${i}]`,
              formula: `dX[${b},${i}] = sum_o dY[${b},o] * W[o,${i}]`,
              highlights: {
                dY: { indices: { b, o: [0, Dout - 1] }, type: 'vector' },
                W: { indices: { o: [0, Dout - 1], i }, type: 'vector' },
                dX: { indices: { b, i }, type: 'output' },
              },
            },
            layerIdx,
            'backward',
            'dX'
          )
        );
      }
    }

    const db: OperationStep[] = [];
    id = 0;
    for (let o = 0; o < Dout; o++) {
      db.push(
        withMeta(
          {
            id: id++,
            description: `Linear backward db: compute db[${o}]`,
            formula: `db[${o}] = sum_b dY[b,${o}]`,
            highlights: {
              dY: { indices: { b: [0, B - 1], o }, type: 'vector' },
              db: { indices: { o }, type: 'output' },
            },
          },
          layerIdx,
          'backward',
          'db'
        )
      );
    }

    return {
      forward: renumberSteps(forward),
      backward: { dX: renumberSteps(dX), dW: renumberSteps(dW), db: renumberSteps(db) },
    };
  }

  if (spec.kind === 'relu') {
    const forward: OperationStep[] = [];
    let id = 0;
    if (rt.inputShape.length === 4) {
      const [B, C, H, W] = rt.inputShape;
      for (const { b, c, h, w } of iterate4D(B, C, H, W)) {
        forward.push(
          withMeta(
            {
              id: id++,
              description: `ReLU: Y[${b},${c},${h},${w}] = max(0, X[${b},${c},${h},${w}])`,
              formula: `Y[${b},${c},${h},${w}] = relu(X[${b},${c},${h},${w}])`,
              highlights: { X: { indices: { b, c, h, w }, type: 'vector' }, Y: { indices: { b, c, h, w }, type: 'output' } },
            },
            layerIdx,
            'forward'
          )
        );
      }
    } else {
      const [B, C, L] = rt.inputShape;
      for (const { b, c, l } of iterate3D(B, C, L)) {
        forward.push(
          withMeta(
            {
              id: id++,
              description: `ReLU: Y[${b},${c},${l}] = max(0, X[${b},${c},${l}])`,
              formula: `Y[${b},${c},${l}] = relu(X[${b},${c},${l}])`,
              highlights: { X: { indices: { b, c, l }, type: 'vector' }, Y: { indices: { b, c, l }, type: 'output' } },
            },
            layerIdx,
            'forward'
          )
        );
      }
    }

    const dX: OperationStep[] = [];
    id = 0;
    if (rt.inputShape.length === 4) {
      const [B, C, H, W] = rt.inputShape;
      for (const { b, c, h, w } of iterate4D(B, C, H, W)) {
        dX.push(
          withMeta(
            {
              id: id++,
              description: `ReLU backward: dX[${b},${c},${h},${w}] = dY[${b},${c},${h},${w}] * 1{X>0}`,
              formula: `dX[${b},${c},${h},${w}] = dY[${b},${c},${h},${w}] * mask[${b},${c},${h},${w}]`,
              highlights: { dY: { indices: { b, c, h, w }, type: 'vector' }, dX: { indices: { b, c, h, w }, type: 'output' } },
            },
            layerIdx,
            'backward',
            'dX'
          )
        );
      }
    } else {
      const [B, C, L] = rt.inputShape;
      for (const { b, c, l } of iterate3D(B, C, L)) {
        dX.push(
          withMeta(
            {
              id: id++,
              description: `ReLU backward: dX[${b},${c},${l}] = dY[${b},${c},${l}] * 1{X>0}`,
              formula: `dX[${b},${c},${l}] = dY[${b},${c},${l}] * mask[${b},${c},${l}]`,
              highlights: { dY: { indices: { b, c, l }, type: 'vector' }, dX: { indices: { b, c, l }, type: 'output' } },
            },
            layerIdx,
            'backward',
            'dX'
          )
        );
      }
    }

    return { forward: renumberSteps(forward), backward: { dX: renumberSteps(dX) } };
  }

  if (spec.kind === 'flatten') {
    const B = rt.inputShape[0];
    const Din = rt.inputShape.slice(1).reduce((acc, v) => acc * v, 1);
    const forward: OperationStep[] = [
      withMeta(
        {
          id: 0,
          description: 'Flatten: reshape (B,...) → (B, D)',
          formula: `Y[b,j] = reshape(X)[b,j]   for j in 0..${Din - 1}`,
          highlights: {
            X: {
              indices: { b: [0, B - 1] },
              type: 'vector',
            },
            Y: {
              indices: { b: [0, B - 1], j: [0, Din - 1] },
              type: 'output',
            },
          },
        },
        layerIdx,
        'forward',
        undefined
      ),
    ];
    const dX: OperationStep[] = [
      withMeta(
        {
          id: 0,
          description: 'Flatten backward: reshape gradient',
          formula: `dX = reshape(dY)`,
          highlights: {
            dY: {
              indices: { b: [0, B - 1], j: [0, Din - 1] },
              type: 'vector',
            },
            dX: {
              indices: { b: [0, B - 1] },
              type: 'output',
            },
          },
        },
        layerIdx,
        'backward',
        'dX'
      ),
    ];
    return { forward, backward: { dX } };
  }

  return { forward: [], backward: emptyBackward };
}

/** Tensor specs + sizes for LayerView rendering */
export function dimensionSizesForTensor(
  name: string,
  rt: LayerRuntime,
  spec: LayerSpec
): DimensionSizes | null {
  if (spec.kind === 'conv1d') {
    const [B, Cin, Lin] = rt.inputShape;
    const [, Cout, Lout] = rt.outputShape;
    const K = spec.kernelSize;
    if (name === 'X' || name === 'dX') return { b: B, c: Cin, l: Lin };
    if (name === 'Y' || name === 'dY') return { b: B, o: Cout, l: Lout };
    if (name === 'W' || name === 'dW') return { o: Cout, c: Cin, k: K };
    if (name === 'bias' || name === 'db') return { o: Cout };
  }
  if (spec.kind === 'conv2d') {
    const [B, Cin, Hin, Win] = rt.inputShape;
    const [, Cout, Hout, Wout] = rt.outputShape;
    const Kh = spec.kernelH;
    const Kw = spec.kernelW;
    if (name === 'X' || name === 'dX') return { b: B, c: Cin, h: Hin, w: Win };
    if (name === 'Y' || name === 'dY') return { b: B, o: Cout, h: Hout, w: Wout };
    if (name === 'W' || name === 'dW') return { o: Cout, c: Cin, kh: Kh, kw: Kw };
    if (name === 'bias' || name === 'db') return { o: Cout };
  }
  if (spec.kind === 'linear') {
    const [B, Din] = rt.inputShape;
    const [, Dout] = rt.outputShape;
    if (name === 'X' || name === 'dX') return { b: B, i: Din };
    if (name === 'Y' || name === 'dY') return { b: B, o: Dout };
    if (name === 'W' || name === 'dW') return { o: Dout, i: Din };
    if (name === 'bias' || name === 'db') return { o: Dout };
  }
  if (spec.kind === 'relu') {
    if (rt.inputShape.length === 4) {
      const [B, C, H, W] = rt.inputShape;
      if (name === 'X' || name === 'dX' || name === 'Y' || name === 'dY') return { b: B, c: C, h: H, w: W };
    } else {
      const [B, C, L] = rt.inputShape;
      if (name === 'X' || name === 'dX' || name === 'Y' || name === 'dY') return { b: B, c: C, l: L };
    }
  }
  if (spec.kind === 'flatten') {
    const B = rt.inputShape[0];
    const Din = rt.inputShape.slice(1).reduce((acc, v) => acc * v, 1);
    if (name === 'X' || name === 'dX') {
      if (rt.inputShape.length === 4) {
        const [, C, H, W] = rt.inputShape;
        return { b: B, c: C, h: H, w: W };
      }
      const [, C, L] = rt.inputShape;
      return { b: B, c: C, l: L };
    }
    if (name === 'Y' || name === 'dY') return { b: B, j: Din };
  }
  return null;
}

export function tensorSpecForName(name: string, rt: LayerRuntime): { name: string; indices: string[] } | null {
  const spec = rt.spec;
  if (spec.kind === 'conv1d') {
    if (name === 'X' || name === 'dX') return { name, indices: ['b', 'c', 'l'] };
    if (name === 'Y' || name === 'dY') return { name, indices: ['b', 'o', 'l'] };
    if (name === 'W' || name === 'dW') return { name, indices: ['o', 'c', 'k'] };
    if (name === 'bias' || name === 'db') return { name, indices: ['o'] };
  }
  if (spec.kind === 'conv2d') {
    if (name === 'X' || name === 'dX') return { name, indices: ['b', 'c', 'h', 'w'] };
    if (name === 'Y' || name === 'dY') return { name, indices: ['b', 'o', 'h', 'w'] };
    if (name === 'W' || name === 'dW') return { name, indices: ['o', 'c', 'kh', 'kw'] };
    if (name === 'bias' || name === 'db') return { name, indices: ['o'] };
  }
  if (spec.kind === 'linear') {
    if (name === 'X' || name === 'dX') return { name, indices: ['b', 'i'] };
    if (name === 'Y' || name === 'dY') return { name, indices: ['b', 'o'] };
    if (name === 'W' || name === 'dW') return { name, indices: ['o', 'i'] };
    if (name === 'bias' || name === 'db') return { name, indices: ['o'] };
  }
  if (spec.kind === 'relu') {
    if (rt.inputShape.length === 4) {
      if (name === 'X' || name === 'dX' || name === 'Y' || name === 'dY') return { name, indices: ['b', 'c', 'h', 'w'] };
    } else {
      if (name === 'X' || name === 'dX' || name === 'Y' || name === 'dY') return { name, indices: ['b', 'c', 'l'] };
    }
  }
  if (spec.kind === 'flatten') {
    if (name === 'X' || name === 'dX') {
      return rt.inputShape.length === 4 ? { name, indices: ['b', 'c', 'h', 'w'] } : { name, indices: ['b', 'c', 'l'] };
    }
    if (name === 'Y' || name === 'dY') return { name, indices: ['b', 'j'] };
  }
  return null;
}
