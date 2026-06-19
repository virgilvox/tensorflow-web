/**
 * Add layer converter. Lowers a tfjs Add layer to a TFLite ADD op with no fused
 * activation. The layer takes two or more input tensors and produces their
 * elementwise sum.
 */
import type * as tf from '@tensorflow/tfjs';
import { registerConverter } from './registry';
import type { ConvertContext, LayerConverter } from './types';
import { layerOutputShape } from './shape';

const converter: LayerConverter = {
  layerClass: 'Add',
  convert(layer: tf.layers.Layer, inputs: number[], ctx: ConvertContext): number[] {
    if (inputs.length < 2) {
      throw new Error(`Add "${layer.name}" expects at least two inputs, got ${inputs.length}.`);
    }
    const out = ctx.builder.addActivation(layer.name, layerOutputShape(layer), 'float32');
    ctx.builder.addOp({
      kind: 'ADD',
      inputs: [...inputs],
      outputs: [out],
      options: { fusedActivation: 'NONE' },
    });
    return [out];
  },
};

registerConverter(converter);
