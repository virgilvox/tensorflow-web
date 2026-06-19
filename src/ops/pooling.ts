/**
 * Pooling converters. MaxPooling2D lowers to MAX_POOL_2D and AveragePooling2D
 * lowers to AVERAGE_POOL_2D. Filter size comes from poolSize, strides default to
 * the pool size when absent, matching tfjs behavior.
 */
import type * as tf from '@tensorflow/tfjs';
import { registerConverter } from './registry';
import type { ConvertContext, LayerConverter } from './types';
import { toPadding, toStridePair } from './common';
import { layerOutputShape } from './shape';
import type { Pool2DOpts } from '../ir';

function poolOpts(layer: tf.layers.Layer): Pool2DOpts {
  const config = layer.getConfig();
  const [filterH, filterW] = toStridePair(config.poolSize, 2);
  const strides = config.strides ?? config.poolSize;
  const [strideH, strideW] = toStridePair(strides);
  return {
    padding: toPadding(config.padding),
    strideW,
    strideH,
    filterW,
    filterH,
    fusedActivation: 'NONE',
  };
}

function makeConverter(layerClass: string, kind: 'MAX_POOL_2D' | 'AVERAGE_POOL_2D'): LayerConverter {
  return {
    layerClass,
    convert(layer: tf.layers.Layer, inputs: number[], ctx: ConvertContext): number[] {
      if (inputs.length !== 1) {
        throw new Error(`${layerClass} "${layer.name}" expects one input, got ${inputs.length}.`);
      }
      const out = ctx.builder.addActivation(layer.name, layerOutputShape(layer), 'float32');
      ctx.builder.addOp({ kind, inputs: [inputs[0] as number], outputs: [out], options: poolOpts(layer) });
      return [out];
    },
  };
}

registerConverter(makeConverter('MaxPooling2D', 'MAX_POOL_2D'));
registerConverter(makeConverter('AveragePooling2D', 'AVERAGE_POOL_2D'));
