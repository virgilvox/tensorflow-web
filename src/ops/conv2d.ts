/**
 * Conv2D converter. Lowers a tfjs 2D convolution to a TFLite CONV_2D op.
 *
 * Weight layout: tfjs kernel [kh, kw, inC, outC] becomes TFLite OHWI
 * [outC, kh, kw, inC]. Bias is [outC]; a missing bias is synthesized as zeros so
 * downstream quantization always sees three operands. A relu or relu6 activation
 * is fused into the op; any other activation becomes a separate op.
 */
import * as tf from '@tensorflow/tfjs';
import { registerConverter } from './registry';
import type { ConvertContext, LayerConverter } from './types';
import { toFusedActivation, toPadding, toStridePair } from './common';
import { layerOutputShape } from './shape';
import { emitActivation } from './activation-op';

const converter: LayerConverter = {
  layerClass: 'Conv2D',
  convert(layer: tf.layers.Layer, inputs: number[], ctx: ConvertContext): number[] {
    if (inputs.length !== 1) {
      throw new Error(`Conv2D "${layer.name}" expects one input, got ${inputs.length}.`);
    }
    const input = inputs[0] as number;
    const config = layer.getConfig();
    const weights = layer.getWeights();
    const kernel = weights[0] as tf.Tensor;
    const outC = kernel.shape[3] as number;

    // tfjs [kh, kw, inC, outC] -> TFLite OHWI [outC, kh, kw, inC].
    const ohwi = tf.transpose(kernel, [3, 0, 1, 2]);
    const weightIdx = ctx.addFloatConst(`${layer.name}/kernel`, ohwi);
    ohwi.dispose();

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
      kind: 'CONV_2D',
      inputs: [input, weightIdx, biasIdx],
      outputs: [convOut],
      options: {
        padding: toPadding(config.padding),
        strideW,
        strideH,
        dilationW,
        dilationH,
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
