/**
 * Architecture description types. A model is a list of layer specs the builder
 * turns into a tf.LayersModel. Every spec maps to a layer the export op registry
 * supports, so the Expert layer editor can only ever describe an exportable
 * network. Adding a spec kind means adding a supported converter, not a guess.
 */

/** A single layer in an architecture, restricted to exportable operators. */
export type LayerSpec =
  | { type: 'conv2d'; filters: number; kernelSize: number; activation: 'relu' | 'none'; padding: 'same' | 'valid' }
  | { type: 'depthwiseConv2d'; kernelSize: number; activation: 'relu' | 'none'; padding: 'same' | 'valid' }
  | { type: 'maxPool2d'; poolSize: number }
  | { type: 'avgPool2d'; poolSize: number }
  | { type: 'globalAvgPool2d' }
  | { type: 'flatten' }
  | { type: 'dense'; units: number; activation: 'relu' | 'softmax' | 'none' };

/** A complete architecture: an ordered list of layers, head included. */
export interface ModelSpec {
  layers: LayerSpec[];
}

/** Input tensor shape without the batch dimension: [height, width, channels]. */
export type InputShape = [number, number, number];

/** A short, human readable summary of a built model for the Model stage. */
export interface ModelSummary {
  paramCount: number;
  /** Rough int8 weight size in bytes (one byte per parameter, plus a little). */
  estimatedWeightBytes: number;
  layerCount: number;
}
