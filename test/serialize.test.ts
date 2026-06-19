/**
 * Round trip tests for the serializer. These prove the emitted bytes are valid
 * TFLite by parsing them back through the generated reader and checking the
 * structure: version, operator codes, tensors, buffers, quantization, and op
 * options. Numerical parity against the real interpreter is a separate, browser
 * based check that lives in the test app, because tfjs-tflite is WASM only.
 */
import { describe, expect, it } from 'vitest';
import * as flatbuffers from 'flatbuffers';
import { GraphBuilder } from '../src/ir';
import { serialize } from '../src/serialize';
import { f32ToBytes, i8ToBytes, i32ToBytes } from '../src/dtype';
import { BuiltinOperator, BuiltinOptions, Conv2DOptions, Model } from '../src/serialize/schema/tflite';

function parse(bytes: Uint8Array): Model {
  return Model.getRootAsModel(new flatbuffers.ByteBuffer(bytes));
}

describe('serialize', () => {
  it('emits a valid model with the TFL3 identifier', () => {
    const g = new GraphBuilder('test');
    const input = g.addActivation('input', [1, 4], 'float32');
    const output = g.addActivation('output', [1, 4], 'float32');
    g.addOp({ kind: 'LOGISTIC', inputs: [input], outputs: [output] });
    g.setInputs([input]);
    g.setOutputs([output]);

    const bytes = serialize(g.build());
    // The file identifier sits at bytes 4..8.
    expect(new TextDecoder().decode(bytes.subarray(4, 8))).toBe('TFL3');

    const model = parse(bytes);
    expect(model.version()).toBe(3);
    expect(model.subgraphsLength()).toBe(1);
    expect(model.operatorCodesLength()).toBe(1);
    expect(model.operatorCodes(0)?.builtinCode()).toBe(BuiltinOperator.LOGISTIC);

    const sg = model.subgraphs(0)!;
    expect(sg.tensorsLength()).toBe(2);
    expect(sg.inputs(0)).toBe(input);
    expect(sg.outputs(0)).toBe(output);
    expect(sg.operatorsLength()).toBe(1);
  });

  it('lays out constant buffers and per-channel int8 quantization for conv2d', () => {
    const g = new GraphBuilder('conv');
    const input = g.addActivation('input', [1, 3, 3, 1], 'int8', {
      scale: [0.05],
      zeroPoint: [-2],
      quantizedDimension: 0,
    });
    // 2 output channels, 2x2 kernel, 1 input channel. OHWI layout.
    const weightValues = new Int8Array(2 * 2 * 2 * 1).map((_, i) => i - 4);
    const weight = g.addConst(
      'conv/kernel',
      [2, 2, 2, 1],
      'int8',
      i8ToBytes(weightValues),
      { scale: [0.01, 0.02], zeroPoint: [0, 0], quantizedDimension: 0 },
    );
    const bias = g.addConst('conv/bias', [2], 'int32', i32ToBytes([10, -10]), {
      scale: [0.0005, 0.001],
      zeroPoint: [0, 0],
      quantizedDimension: 0,
    });
    const output = g.addActivation('output', [1, 2, 2, 2], 'int8', {
      scale: [0.1],
      zeroPoint: [3],
      quantizedDimension: 0,
    });
    g.addOp({
      kind: 'CONV_2D',
      inputs: [input, weight, bias],
      outputs: [output],
      options: { padding: 'VALID', strideW: 1, strideH: 1, fusedActivation: 'NONE', dilationW: 1, dilationH: 1 },
    });
    g.setInputs([input]);
    g.setOutputs([output]);

    const model = parse(serialize(g.build()));
    const sg = model.subgraphs(0)!;
    expect(sg.tensorsLength()).toBe(4);

    // The weight tensor carries per-channel scales and a real buffer.
    const weightTensor = sg.tensors(weight)!;
    const q = weightTensor.quantization()!;
    expect(q.scaleLength()).toBe(2);
    expect(q.scale(0)).toBeCloseTo(0.01, 6);
    expect(q.scale(1)).toBeCloseTo(0.02, 6);
    expect(q.quantizedDimension()).toBe(0);
    const weightBuffer = model.buffers(weightTensor.buffer())!;
    expect(weightBuffer.dataLength()).toBe(weightValues.length);

    // Activation input shares the empty buffer 0.
    expect(sg.tensors(input)!.buffer()).toBe(0);

    // Conv options survive the round trip.
    const op = sg.operators(0)!;
    expect(op.builtinOptionsType()).toBe(BuiltinOptions.Conv2DOptions);
    const opts = op.builtinOptions(new Conv2DOptions()) as Conv2DOptions;
    expect(opts.strideW()).toBe(1);
    expect(opts.strideH()).toBe(1);
  });

  it('rejects a graph with no inputs', () => {
    const g = new GraphBuilder();
    const t = g.addActivation('x', [1], 'float32');
    g.setOutputs([t]);
    expect(() => g.build()).toThrow(/no inputs/);
  });

  it('encodes float32 weights as little endian bytes', () => {
    const bytes = f32ToBytes([1.5]);
    // 1.5 in IEEE-754 little endian is 00 00 C0 3F.
    expect(Array.from(bytes)).toEqual([0x00, 0x00, 0xc0, 0x3f]);
  });
});
