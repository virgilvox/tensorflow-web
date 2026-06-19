/**
 * Flatten and Reshape converters. Both lower to a TFLite RESHAPE op. The target
 * shape is the layer's output shape with the batch dimension fixed to 1. It is
 * carried both as the newShape option and as a constant int32 shape tensor
 * input, which RESHAPE reads as its second operand.
 */
import type * as tf from '@tensorflow/tfjs';
import { registerConverter } from './registry';
import type { ConvertContext, LayerConverter } from './types';
import { layerOutputShape } from './shape';

function makeConverter(layerClass: string): LayerConverter {
  return {
    layerClass,
    convert(layer: tf.layers.Layer, inputs: number[], ctx: ConvertContext): number[] {
      if (inputs.length !== 1) {
        throw new Error(`${layerClass} "${layer.name}" expects one input, got ${inputs.length}.`);
      }
      const newShape = layerOutputShape(layer);
      const shapeIdx = ctx.addInt32Const(`${layer.name}/shape`, newShape);
      const out = ctx.builder.addActivation(layer.name, newShape, 'float32');
      ctx.builder.addOp({
        kind: 'RESHAPE',
        inputs: [inputs[0] as number, shapeIdx],
        outputs: [out],
        options: { newShape },
      });
      return [out];
    },
  };
}

registerConverter(makeConverter('Flatten'));
registerConverter(makeConverter('Reshape'));
