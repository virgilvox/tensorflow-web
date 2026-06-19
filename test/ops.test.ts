import { describe, it, expect } from 'vitest';
import { useCpu } from './helpers/tf';
import { GraphBuilder } from '../src/ir';
import { bytesToF32 } from '../src/dtype';
import '../src/ops';
import { getConverter, hasConverter, UnsupportedLayerError } from '../src/ops/registry';
import type { ConvertContext } from '../src/ops/types';
import { f32ToBytes, i32ToBytes } from '../src/dtype';
import type { IRTensor } from '../src/ir';

/** A test context that records the tensors converters add. */
function makeCtx(builder: GraphBuilder): ConvertContext {
  return {
    builder,
    addFloatConst(name, tensor) {
      const data = tensor.dataSync() as Float32Array;
      return builder.addConst(name, [...tensor.shape], 'float32', f32ToBytes(data));
    },
    addInt32Const(name, values, shape) {
      return builder.addConst(name, shape ?? [values.length], 'int32', i32ToBytes(values));
    },
  };
}

function floatData(t: IRTensor): number[] {
  return Array.from(bytesToF32(t.data!));
}

describe('Conv2D weight transpose', () => {
  it('maps tfjs [kh,kw,inC,outC] to OHWI [outC,kh,kw,inC]', async () => {
    const tf = await useCpu();
    // kh=1, kw=1, inC=2, outC=3. Each [inC,outC] block lets us track the map.
    const kernel = tf.tensor([[[[1, 2, 3], [4, 5, 6]]]]); // shape [1,1,2,3]
    expect(kernel.shape).toEqual([1, 1, 2, 3]);
    const layer = tf.layers.conv2d({ filters: 3, kernelSize: 1, useBias: false });
    layer.apply(tf.input({ shape: [4, 4, 2] }));
    layer.setWeights([kernel]);

    const builder = new GraphBuilder();
    const ctx = makeCtx(builder);
    getConverter('Conv2D').convert(layer, [99], ctx);

    // The convert call added the kernel const at index 0, bias at index 1.
    const w = builder.getTensor(0);
    expect(w.shape).toEqual([3, 1, 1, 2]); // OHWI
    // value at [outC,kh,kw,inC]; original value[kh,kw,inC,outC].
    // out=0 -> inC0=1, inC1=4 ; out=1 -> 2,5 ; out=2 -> 3,6
    expect(floatData(w)).toEqual([1, 4, 2, 5, 3, 6]);
  });
});

describe('Dense weight transpose', () => {
  it('maps tfjs [inDim,units] to [units,inDim]', async () => {
    const tf = await useCpu();
    const kernel = tf.tensor2d([[1, 2, 3], [4, 5, 6]]); // [inDim=2, units=3]
    const layer = tf.layers.dense({ units: 3, useBias: false });
    layer.apply(tf.input({ shape: [2] }));
    layer.setWeights([kernel]);

    const builder = new GraphBuilder();
    const ctx = makeCtx(builder);
    getConverter('Dense').convert(layer, [0], ctx);

    const w = builder.getTensor(0);
    expect(w.shape).toEqual([3, 2]); // [units, inDim]
    expect(floatData(w)).toEqual([1, 4, 2, 5, 3, 6]);
  });
});

describe('DepthwiseConv2D weight reshape', () => {
  it('maps tfjs [kh,kw,inC,mult] to [1,kh,kw,inC*mult] and sets depthMultiplier', async () => {
    const tf = await useCpu();
    // kh=1,kw=1,inC=2,mult=2 -> values flatten in the same order to [1,1,1,4]
    const kernel = tf.tensor([[[[1, 2], [3, 4]]]]); // [1,1,2,2]
    const layer = tf.layers.depthwiseConv2d({ kernelSize: 1, depthMultiplier: 2, useBias: false });
    layer.apply(tf.input({ shape: [4, 4, 2] }));
    layer.setWeights([kernel]);

    const builder = new GraphBuilder();
    const ctx = makeCtx(builder);
    getConverter('DepthwiseConv2D').convert(layer, [0], ctx);

    const w = builder.getTensor(0);
    expect(w.shape).toEqual([1, 1, 1, 4]);
    expect(floatData(w)).toEqual([1, 2, 3, 4]);
  });
});

describe('useBias false still synthesizes a zero bias', () => {
  it('adds a zero bias const for Dense without bias', async () => {
    const tf = await useCpu();
    const layer = tf.layers.dense({ units: 3, useBias: false });
    layer.apply(tf.input({ shape: [2] }));

    const builder = new GraphBuilder();
    const ctx = makeCtx(builder);
    const out = getConverter('Dense').convert(layer, [99], ctx);

    const bias = builder.getTensor(1);
    expect(bias.shape).toEqual([3]);
    expect(floatData(bias)).toEqual([0, 0, 0]);
    // FULLY_CONNECTED inputs ordered [input, weight, bias].
    expect(out.length).toBe(1);
  });
});

describe('unsupported layer', () => {
  it('throws UnsupportedLayerError naming the class', async () => {
    expect(hasConverter('LSTM')).toBe(false);
    expect(() => getConverter('LSTM')).toThrow(UnsupportedLayerError);
    try {
      getConverter('LSTM');
    } catch (e) {
      expect((e as UnsupportedLayerError).layerClass).toBe('LSTM');
      expect((e as Error).message).toContain('LSTM');
    }
  });
});
