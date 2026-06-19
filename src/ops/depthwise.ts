/**
 * DepthwiseConv2D converter. Lowers a tfjs depthwise convolution to a TFLite
 * DEPTHWISE_CONV_2D op.
 *
 * Weight layout: tfjs kernel [kh, kw, inC, mult] becomes TFLite
 * [1, kh, kw, inC * mult]. The inC and mult axes are already contiguous and in
 * the right order, so a reshape produces the TFLite layout. depthMultiplier is
 * set to mult. A missing bias is synthesized as zeros of length inC * mult.
 */
import * as tf from '@tensorflow/tfjs';
import { registerConverter } from './registry';
import type { ConvertContext, LayerConverter } from './types';
import { toFusedActivation, toPadding, toStridePair } from './common';
import { layerOutputShape } from './shape';
import { emitActivation } from './activation-op';

const converter: LayerConverter = {
  layerClass: 'DepthwiseConv2D',
  convert(layer: tf.layers.Layer, inputs: number[], ctx: ConvertContext): number[] {
    if (inputs.length !== 1) {
      throw new Error(`DepthwiseConv2D "${layer.name}" expects one input, got ${inputs.length}.`);
    }
    const input = inputs[0] as number;
    const config = layer.getConfig();
    const weights = layer.getWeights();
    const kernel = weights[0] as tf.Tensor;
    const [kh, kw, inC, mult] = kernel.shape as [number, number, number, number];
    const outC = inC * mult;

    const reshaped = tf.reshape(kernel, [1, kh, kw, outC]);
    const weightIdx = ctx.addFloatConst(`${layer.name}/kernel`, reshaped);
    reshaped.dispose();

    let biasIdx: number;
    if (config.useBias !== false && weights.length > 1) {
      biasIdx = ctx.addFloatConst(`${layer.name}/bias`, weights[1] as tf.Tensor);
    } else {
      const zeros = tf.zeros([outC]);
      biasIdx = ctx.addFloatConst(`${layer.name}/bias`, zeros);
      zeros.dispose();
    }

    const activation = config.activation as string | undefined;
    const fused = toFusedActivation(activation);
    const outShape = layerOutputShape(layer);
    const [strideH, strideW] = toStridePair(config.strides);
    const [dilationH, dilationW] = toStridePair(config.dilationRate);

    const needsSeparate = fused === null;
    const convOutName = needsSeparate ? `${layer.name}/conv` : layer.name;
    const convOut = ctx.builder.addActivation(convOutName, outShape, 'float32');
    ctx.builder.addOp({
      kind: 'DEPTHWISE_CONV_2D',
      inputs: [input, weightIdx, biasIdx],
      outputs: [convOut],
      options: {
        padding: toPadding(config.padding),
        strideW,
        strideH,
        dilationW,
        dilationH,
        depthMultiplier: mult,
        fusedActivation: fused ?? 'NONE',
      },
    });

    if (needsSeparate) {
      return [emitActivation(ctx, activation as string, convOut, layer.name, outShape)];
    }
    return [convOut];
  },
};

registerConverter(converter);
