/**
 * Dense converter. Lowers a tfjs dense layer to a TFLite FULLY_CONNECTED op.
 *
 * Weight layout: tfjs kernel [inDim, units] becomes TFLite [units, inDim]. Bias
 * is [units]; a missing bias is synthesized as zeros. keepNumDims is true so a
 * rank greater than two input keeps its leading dims, matching Keras. A
 * relu or relu6 activation is fused; any other activation becomes a separate op.
 */
import * as tf from '@tensorflow/tfjs';
import { registerConverter } from './registry';
import type { ConvertContext, LayerConverter } from './types';
import { toFusedActivation } from './common';
import { layerOutputShape } from './shape';
import { emitActivation } from './activation-op';

const converter: LayerConverter = {
  layerClass: 'Dense',
  convert(layer: tf.layers.Layer, inputs: number[], ctx: ConvertContext): number[] {
    if (inputs.length !== 1) {
      throw new Error(`Dense "${layer.name}" expects one input, got ${inputs.length}.`);
    }
    const input = inputs[0] as number;
    const config = layer.getConfig();
    const weights = layer.getWeights();
    const kernel = weights[0] as tf.Tensor;
    const units = kernel.shape[1] as number;

    // tfjs [inDim, units] -> TFLite [units, inDim].
    const transposed = tf.transpose(kernel, [1, 0]);
    const weightIdx = ctx.addFloatConst(`${layer.name}/kernel`, transposed);
    transposed.dispose();

    let biasIdx: number;
    if (config.useBias !== false && weights.length > 1) {
      biasIdx = ctx.addFloatConst(`${layer.name}/bias`, weights[1] as tf.Tensor);
    } else {
      const zeros = tf.zeros([units]);
      biasIdx = ctx.addFloatConst(`${layer.name}/bias`, zeros);
      zeros.dispose();
    }

    const activation = config.activation as string | undefined;
    const fused = toFusedActivation(activation);
    const outShape = layerOutputShape(layer);

    const needsSeparate = fused === null;
    const fcOutName = needsSeparate ? `${layer.name}/fc` : layer.name;
    const fcOut = ctx.builder.addActivation(fcOutName, outShape, 'float32');
    ctx.builder.addOp({
      kind: 'FULLY_CONNECTED',
      inputs: [input, weightIdx, biasIdx],
      outputs: [fcOut],
      options: { fusedActivation: fused ?? 'NONE', keepNumDims: true },
    });

    if (needsSeparate) {
      return [emitActivation(ctx, activation as string, fcOut, layer.name, outShape)];
    }
    return [fcOut];
  },
};

registerConverter(converter);
