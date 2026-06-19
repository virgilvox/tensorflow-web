/**
 * Softmax layer converter. Lowers a tfjs Softmax layer to a TFLite SOFTMAX op
 * with beta 1.
 */
import type * as tf from '@tensorflow/tfjs';
import { registerConverter } from './registry';
import type { ConvertContext, LayerConverter } from './types';
import { layerOutputShape } from './shape';

const converter: LayerConverter = {
  layerClass: 'Softmax',
  convert(layer: tf.layers.Layer, inputs: number[], ctx: ConvertContext): number[] {
    if (inputs.length !== 1) {
      throw new Error(`Softmax "${layer.name}" expects one input, got ${inputs.length}.`);
    }
    const out = ctx.builder.addActivation(layer.name, layerOutputShape(layer), 'float32');
    ctx.builder.addOp({ kind: 'SOFTMAX', inputs: [inputs[0] as number], outputs: [out], options: { beta: 1 } });
    return [out];
  },
};

registerConverter(converter);
