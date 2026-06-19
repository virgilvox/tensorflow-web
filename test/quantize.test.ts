/**
 * Graph level tests for quantizeGraph. A tiny float CONV_2D graph is built by
 * hand, given a calibration, and quantized. The assertions check the structural
 * contract from DESIGN.md: int8 per channel weights, int32 bias, float
 * boundaries via QUANTIZE and DEQUANTIZE, and float graph input and output.
 */
import { describe, expect, it } from 'vitest';
import { GraphBuilder } from '../src/ir';
import type { IRGraph } from '../src/ir';
import type { Calibration } from '../src/types';
import { f32ToBytes, bytesToI8 } from '../src/dtype';
import { quantizeGraph } from '../src/quantize';

/** Builds a single CONV_2D float graph: input -> conv(weight, bias) -> output. */
function buildConvGraph(): IRGraph {
  const g = new GraphBuilder('conv-test');
  const input = g.addActivation('input', [1, 1, 1, 1], 'float32');
  // CONV_2D weights are [outC, kh, kw, inC]. Two output channels.
  const weight = g.addConst('conv/kernel', [2, 1, 1, 1], 'float32', f32ToBytes([2, -4]));
  const bias = g.addConst('conv/bias', [2], 'float32', f32ToBytes([1, -1]));
  const output = g.addActivation('conv', [1, 1, 1, 2], 'float32');
  g.addOp({
    kind: 'CONV_2D',
    inputs: [input, weight, bias],
    outputs: [output],
    options: { padding: 'VALID', strideW: 1, strideH: 1, fusedActivation: 'NONE', dilationW: 1, dilationH: 1 },
  });
  g.setInputs([input]);
  g.setOutputs([output]);
  return g.build();
}

const calibration: Calibration = {
  ranges: {
    input: { min: 0, max: 1 },
    conv: { min: -2, max: 6 },
  },
  method: 'minmax',
  sampleCount: 4,
};

describe('quantizeGraph', () => {
  it('quantizes a CONV_2D graph to int8 with float boundaries', () => {
    const out = quantizeGraph(buildConvGraph(), calibration);

    // Exactly one QUANTIZE at the input and one DEQUANTIZE at the output.
    const quantizeOps = out.ops.filter((o) => o.kind === 'QUANTIZE');
    const dequantizeOps = out.ops.filter((o) => o.kind === 'DEQUANTIZE');
    expect(quantizeOps).toHaveLength(1);
    expect(dequantizeOps).toHaveLength(1);

    const tensor = (i: number) => {
      const t = out.tensors[i];
      if (!t) throw new Error(`no tensor at ${i}`);
      return t;
    };

    // The graph input and output stay float32.
    expect(out.inputs).toHaveLength(1);
    expect(out.outputs).toHaveLength(1);
    const inputIdx = out.inputs[0]!;
    const outputIdx = out.outputs[0]!;
    expect(tensor(inputIdx).dtype).toBe('float32');
    expect(tensor(outputIdx).dtype).toBe('float32');

    const quant = quantizeOps[0]!;
    const dequant = dequantizeOps[0]!;

    // The QUANTIZE op reads the float input and writes an int8 tensor.
    expect(quant.inputs[0]).toBe(inputIdx);
    expect(tensor(quant.outputs[0]!).dtype).toBe('int8');

    // The DEQUANTIZE op writes the float output from an int8 tensor.
    expect(dequant.outputs[0]).toBe(outputIdx);
    expect(tensor(dequant.inputs[0]!).dtype).toBe('int8');

    // The CONV_2D op survives, now reading int8 input, weight, and bias.
    const conv = out.ops.find((o) => o.kind === 'CONV_2D');
    expect(conv).toBeDefined();
    const actIdx = conv!.inputs[0]!;
    const weightIdx = conv!.inputs[1]!;
    const biasIdx = conv!.inputs[2]!;

    const actTensor = tensor(actIdx);
    expect(actTensor.dtype).toBe('int8');
    // Its scale matches the asymmetric activation quant of [0, 1]: 1/255.
    expect(actTensor.quantization!.scale[0]).toBeCloseTo(1 / 255, 10);

    // Weight is int8, per channel, one scale per output channel (outC == 2).
    const weightTensor = tensor(weightIdx);
    expect(weightTensor.dtype).toBe('int8');
    expect(weightTensor.quantization!.quantizedDimension).toBe(0);
    expect(weightTensor.quantization!.scale).toHaveLength(2);
    expect(weightTensor.quantization!.zeroPoint).toEqual([0, 0]);
    // Channel 0 maxAbs 2 -> scale 2/127; channel 1 maxAbs 4 -> scale 4/127.
    expect(weightTensor.quantization!.scale[0]).toBeCloseTo(2 / 127, 10);
    expect(weightTensor.quantization!.scale[1]).toBeCloseTo(4 / 127, 10);
    // Values round to the symmetric extreme for both channels.
    expect(Array.from(bytesToI8(weightTensor.data!))).toEqual([127, -127]);

    // Bias is int32, per channel, scale = inputScale * weightScale_c.
    const biasTensor = tensor(biasIdx);
    expect(biasTensor.dtype).toBe('int32');
    expect(biasTensor.quantization!.scale).toHaveLength(2);
    const inputScale = 1 / 255;
    expect(biasTensor.quantization!.scale[0]).toBeCloseTo(inputScale * (2 / 127), 14);
    expect(biasTensor.quantization!.scale[1]).toBeCloseTo(inputScale * (4 / 127), 14);

    // The int8 output activation carries the [-2, 6] range quant: scale 8/255.
    const int8Output = tensor(dequant.inputs[0]!);
    expect(int8Output.quantization!.scale[0]).toBeCloseTo(8 / 255, 10);
    expect(int8Output.quantization!.zeroPoint[0]).toBe(-64);
  });

  it('throws when a calibration range is missing', () => {
    const partial: Calibration = {
      ranges: { input: { min: 0, max: 1 } },
      method: 'minmax',
      sampleCount: 1,
    };
    expect(() => quantizeGraph(buildConvGraph(), partial)).toThrow(/calibration range/);
  });

  it('forces FULLY_CONNECTED weights to per tensor even when per channel is requested', () => {
    // FC weights are [units, inDim] = [2, 1]. The TFLite int8 FC kernel reads a
    // single weight scale, so per channel here would be misread.
    const g = new GraphBuilder('fc');
    const input = g.addActivation('input', [1, 1], 'float32');
    const weight = g.addConst('fc/kernel', [2, 1], 'float32', f32ToBytes([2, -4]));
    const bias = g.addConst('fc/bias', [2], 'float32', f32ToBytes([0, 0]));
    const output = g.addActivation('fc', [1, 2], 'float32');
    g.addOp({
      kind: 'FULLY_CONNECTED',
      inputs: [input, weight, bias],
      outputs: [output],
      options: { fusedActivation: 'NONE', keepNumDims: true },
    });
    g.setInputs([input]);
    g.setOutputs([output]);
    const cal: Calibration = {
      ranges: { input: { min: 0, max: 1 }, fc: { min: -4, max: 4 } },
      method: 'minmax',
      sampleCount: 1,
    };
    const out = quantizeGraph(g.build(), cal, { weights: 'per-channel' });
    const fc = out.ops.find((o) => o.kind === 'FULLY_CONNECTED')!;
    const weightTensor = out.tensors[fc.inputs[1]!]!;
    // A single scale = maxAbs over the whole tensor / 127 = 4/127, not two scales.
    expect(weightTensor.quantization!.scale).toHaveLength(1);
    expect(weightTensor.quantization!.scale[0]).toBeCloseTo(4 / 127, 10);
  });

  it('overrides the SOFTMAX output scale to the kernel-mandated 1/256, ignoring calibration', () => {
    const g = new GraphBuilder('sm');
    const input = g.addActivation('input', [1, 3], 'float32');
    const output = g.addActivation('sm', [1, 3], 'float32');
    g.addOp({ kind: 'SOFTMAX', inputs: [input], outputs: [output], options: { beta: 1 } });
    g.setInputs([input]);
    g.setOutputs([output]);
    // A deliberately wrong calibration for the softmax output. The kernel fixes
    // the scale, so the quantizer must ignore this range.
    const cal: Calibration = {
      ranges: { input: { min: -5, max: 5 }, sm: { min: -50, max: 50 } },
      method: 'minmax',
      sampleCount: 1,
    };
    const out = quantizeGraph(g.build(), cal);
    const dequant = out.ops.find((o) => o.kind === 'DEQUANTIZE')!;
    const smOut = out.tensors[dequant.inputs[0]!]!;
    expect(smOut.quantization!.scale[0]).toBeCloseTo(1 / 256, 12);
    expect(smOut.quantization!.zeroPoint[0]).toBe(-128);
  });

  it('gives a MAX_POOL_2D output the same scale and zero point as its input', () => {
    const g = new GraphBuilder('pool');
    const input = g.addActivation('input', [1, 4, 4, 1], 'float32');
    const output = g.addActivation('pool', [1, 2, 2, 1], 'float32');
    g.addOp({
      kind: 'MAX_POOL_2D',
      inputs: [input],
      outputs: [output],
      options: { padding: 'VALID', strideW: 2, strideH: 2, filterW: 2, filterH: 2, fusedActivation: 'NONE' },
    });
    g.setInputs([input]);
    g.setOutputs([output]);
    // Different calibrated ranges for input and output. The int8 max pool kernel
    // takes the max of raw codes, so the output must inherit the input's quant.
    const cal: Calibration = {
      ranges: { input: { min: 0, max: 8 }, pool: { min: 0, max: 3 } },
      method: 'minmax',
      sampleCount: 1,
    };
    const out = quantizeGraph(g.build(), cal);
    const pool = out.ops.find((o) => o.kind === 'MAX_POOL_2D')!;
    const inT = out.tensors[pool.inputs[0]!]!;
    const outT = out.tensors[pool.outputs[0]!]!;
    expect(outT.quantization!.scale[0]).toBe(inT.quantization!.scale[0]);
    expect(outT.quantization!.zeroPoint[0]).toBe(inT.quantization!.zeroPoint[0]);
    // It is the input's scale (8/255), not the pool output's calibration (3/255).
    expect(outT.quantization!.scale[0]).toBeCloseTo(8 / 255, 10);
  });

  it('rejects a graph with more than one input', () => {
    const g = new GraphBuilder('multi');
    const a = g.addActivation('input', [1, 2], 'float32');
    const b = g.addActivation('input2', [1, 2], 'float32');
    const output = g.addActivation('out', [1, 2], 'float32');
    g.addOp({ kind: 'ADD', inputs: [a, b], outputs: [output], options: { fusedActivation: 'NONE' } });
    g.setInputs([a, b]);
    g.setOutputs([output]);
    const cal: Calibration = { ranges: {}, method: 'minmax', sampleCount: 1 };
    expect(() => quantizeGraph(g.build(), cal)).toThrow(/one graph input/);
  });
});
