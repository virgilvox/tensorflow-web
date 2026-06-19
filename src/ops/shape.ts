/**
 * Output shape helpers for converters. A converter names its output activation
 * after the layer and shapes it from the layer's output shape with the batch
 * dimension fixed to 1.
 */
import type * as tf from '@tensorflow/tfjs';

/**
 * Returns the layer's output shape with a null or negative batch dimension
 * replaced by 1.
 * Throws if the layer reports more than one output shape (branching), which v1
 * does not support.
 */
export function layerOutputShape(layer: tf.layers.Layer): number[] {
  const raw = layer.outputShape as number[] | number[][];
  if (Array.isArray(raw[0])) {
    throw new Error(
      `Layer "${layer.name}" has multiple output shapes; v1 supports one output per layer only.`,
    );
  }
  const shape = raw as number[];
  return shape.map((d, i) => (i === 0 && (d == null || d < 0) ? 1 : (d as number)));
}
