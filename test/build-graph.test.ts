import { describe, it, expect } from 'vitest';
import type { SymbolicTensor } from '@tensorflow/tfjs';
import { useCpu } from './helpers/tf';
import { buildFloatGraph } from '../src/convert/build-graph';
import { UnsupportedLayerError } from '../src/ops/registry';

describe('buildFloatGraph', () => {
  it('lowers a small Sequential model to a float IR graph', async () => {
    const tf = await useCpu();
    const model = tf.sequential();
    model.add(
      tf.layers.conv2d({
        inputShape: [8, 8, 3],
        filters: 4,
        kernelSize: 3,
        padding: 'same',
        activation: 'relu',
      }),
    );
    model.add(tf.layers.maxPooling2d({ poolSize: [2, 2], strides: [2, 2] }));
    model.add(tf.layers.flatten());
    model.add(tf.layers.dense({ units: 5, activation: 'softmax' }));

    const graph = buildFloatGraph(model);

    // Op kinds in order. Dense softmax is not fusable, so it splits into
    // FULLY_CONNECTED then SOFTMAX. Conv relu fuses into CONV_2D.
    const kinds = graph.ops.map((o) => o.kind);
    expect(kinds).toEqual([
      'CONV_2D',
      'MAX_POOL_2D',
      'RESHAPE',
      'FULLY_CONNECTED',
      'SOFTMAX',
    ]);

    // The conv op fused relu.
    const conv = graph.ops[0]!;
    expect(conv.kind).toBe('CONV_2D');
    if (conv.kind === 'CONV_2D') {
      expect(conv.options.fusedActivation).toBe('RELU');
      expect(conv.options.padding).toBe('SAME');
      // inputs ordered [input, weight, bias].
      expect(conv.inputs.length).toBe(3);
      const weightTensor = graph.tensors[conv.inputs[1]!]!;
      // OHWI: [outC, kh, kw, inC] = [4, 3, 3, 3].
      expect(weightTensor.shape).toEqual([4, 3, 3, 3]);
      const biasTensor = graph.tensors[conv.inputs[2]!]!;
      expect(biasTensor.shape).toEqual([4]);
    }

    // Input named 'input', shape with batch 1.
    const inputTensor = graph.tensors[graph.inputs[0]!]!;
    expect(inputTensor.name).toBe('input');
    expect(inputTensor.shape).toEqual([1, 8, 8, 3]);

    // Output is the softmax result.
    expect(graph.outputs.length).toBe(1);
    const outputTensor = graph.tensors[graph.outputs[0]!]!;
    expect(outputTensor.shape).toEqual([1, 5]);

    // Tensor count: input + conv(kernel,bias) + maxpool + reshape(shape const) +
    // fc(kernel,bias) + softmax.
    // input=1, conv act=1, conv weights=2, pool act=1, reshape act=1,
    // reshape shape const=1, fc act=1, fc weights=2, softmax act=1 = 11.
    expect(graph.tensors.length).toBe(11);

    // Indices set and valid.
    expect(graph.inputs[0]).toBeGreaterThanOrEqual(0);
    expect(graph.outputs[0]).toBeLessThan(graph.tensors.length);
  });

  it('throws UnsupportedLayerError naming the class for an unsupported layer', async () => {
    const tf = await useCpu();
    const model = tf.sequential();
    model.add(tf.layers.lstm({ inputShape: [5, 4], units: 3 }));

    expect(() => buildFloatGraph(model)).toThrow(UnsupportedLayerError);
    try {
      buildFloatGraph(model);
    } catch (e) {
      expect((e as UnsupportedLayerError).layerClass).toBe('LSTM');
    }
  });

  it('passes a strided conv through to the op options', async () => {
    const tf = await useCpu();
    const model = tf.sequential();
    model.add(tf.layers.conv2d({ inputShape: [8, 8, 1], filters: 2, kernelSize: 3, strides: [2, 2] }));

    const graph = buildFloatGraph(model);
    const conv = graph.ops.find((o) => o.kind === 'CONV_2D');
    expect(conv?.kind).toBe('CONV_2D');
    if (conv?.kind === 'CONV_2D') {
      expect(conv.options.strideW).toBe(2);
      expect(conv.options.strideH).toBe(2);
    }
  });

  it('lowers a functional model with a skip connection to an ADD op', async () => {
    const tf = await useCpu();
    const input = tf.input({ shape: [8, 8, 1] });
    const a = tf.layers
      .conv2d({ filters: 4, kernelSize: 3, padding: 'same', activation: 'relu' })
      .apply(input) as SymbolicTensor;
    const b = tf.layers
      .conv2d({ filters: 4, kernelSize: 3, padding: 'same', activation: 'relu' })
      .apply(a) as SymbolicTensor;
    const sum = tf.layers.add().apply([a, b]) as SymbolicTensor;
    const flat = tf.layers.flatten().apply(sum) as SymbolicTensor;
    const out = tf.layers.dense({ units: 3, activation: 'softmax' }).apply(flat) as SymbolicTensor;
    const model = tf.model({ inputs: input, outputs: out });

    const graph = buildFloatGraph(model);
    expect(graph.ops.some((o) => o.kind === 'ADD')).toBe(true);
  });
});
