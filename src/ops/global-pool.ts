/**
 * GlobalAveragePooling2D converter. Lowers to a TFLite MEAN op that reduces the
 * spatial axes [1, 2] of an NHWC tensor with keepDims false, leaving [N, C]. The
 * reduction axes are supplied as a constant int32 tensor input.
 */
import type * as tf from '@tensorflow/tfjs';
import { registerConverter } from './registry';
import type { ConvertContext, LayerConverter } from './types';
import { layerOutputShape } from './shape';

const converter: LayerConverter = {
  layerClass: 'GlobalAveragePooling2D',
  convert(layer: tf.layers.Layer, inputs: number[], ctx: ConvertContext): number[] {
    if (inputs.length !== 1) {
      throw new Error(
        `GlobalAveragePooling2D "${layer.name}" expects one input, got ${inputs.length}.`,
      );
    }
    const axisIdx = ctx.addInt32Const(`${layer.name}/axis`, [1, 2]);
    const out = ctx.builder.addActivation(layer.name, layerOutputShape(layer), 'float32');
    ctx.builder.addOp({
      kind: 'MEAN',
      inputs: [inputs[0] as number, axisIdx],
      outputs: [out],
      options: { keepDims: false },
    });
    return [out];
  },
};

registerConverter(converter);
