/**
 * The converter contract. One converter maps one TensorFlow.js layer class to
 * TFLite tensors and a single operator in the graph IR. Converters are pure with
 * respect to quantization: they emit float weights and never look at calibration
 * data. See DESIGN.md for the weight layout rules each converter must follow.
 */
import type * as tf from '@tensorflow/tfjs';
import type { GraphBuilder } from '../ir';

export interface ConvertContext {
  readonly builder: GraphBuilder;
  /** Adds a float32 constant tensor from a tfjs tensor. Returns its IR index. */
  addFloatConst(name: string, tensor: tf.Tensor): number;
  /** Adds a constant int32 vector, used for shape, axis, and paddings inputs. */
  addInt32Const(name: string, values: number[], shape?: number[]): number;
}

export interface LayerConverter {
  /** The TensorFlow.js class name this converter handles, e.g. "Conv2D". */
  readonly layerClass: string;
  /**
   * Converts one layer. `inputs` are the IR tensor indices feeding the layer.
   * Returns the IR tensor indices the layer produces.
   */
  convert(layer: tf.layers.Layer, inputs: number[], ctx: ConvertContext): number[];
}
