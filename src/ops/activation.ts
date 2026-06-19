/**
 * Activation and ReLU layer converters. An Activation layer maps its configured
 * activation to the matching TFLite op: relu and relu6 to RELU and RELU6,
 * sigmoid to LOGISTIC, tanh to TANH, softmax to SOFTMAX. A ReLU layer maps to
 * RELU, or RELU6 when its maxValue is 6.
 */
import type * as tf from '@tensorflow/tfjs';
import { registerConverter } from './registry';
import type { ConvertContext, LayerConverter } from './types';
import { layerOutputShape } from './shape';
import { emitActivation } from './activation-op';

const activationConverter: LayerConverter = {
  layerClass: 'Activation',
  convert(layer: tf.layers.Layer, inputs: number[], ctx: ConvertContext): number[] {
    if (inputs.length !== 1) {
      throw new Error(`Activation "${layer.name}" expects one input, got ${inputs.length}.`);
    }
    const activation = layer.getConfig().activation as string;
    return [emitActivation(ctx, activation, inputs[0] as number, layer.name, layerOutputShape(layer))];
  },
};

const reluConverter: LayerConverter = {
  layerClass: 'ReLU',
  convert(layer: tf.layers.Layer, inputs: number[], ctx: ConvertContext): number[] {
    if (inputs.length !== 1) {
      throw new Error(`ReLU "${layer.name}" expects one input, got ${inputs.length}.`);
    }
    const maxValue = layer.getConfig().maxValue as number | null | undefined;
    let activation: string;
    if (maxValue == null) {
      activation = 'relu';
    } else if (maxValue === 6) {
      activation = 'relu6';
    } else {
      // TFLite has no fused activation for an arbitrary upper bound. Failing
      // loud beats silently dropping the cap and emitting an unclamped ReLU.
      throw new Error(
        `ReLU "${layer.name}" has maxValue ${maxValue}; only an uncapped ReLU or maxValue 6 (RELU6) is supported.`,
      );
    }
    return [emitActivation(ctx, activation, inputs[0] as number, layer.name, layerOutputShape(layer))];
  },
};

registerConverter(activationConverter);
registerConverter(reluConverter);
