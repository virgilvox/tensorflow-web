/**
 * The constrained model builder. Turns a ModelSpec into a compiled
 * tf.LayersModel using only layers the export op registry supports, and mirrors
 * the library's export guard so an unsupported layer fails loud here too, before
 * a user spends time training a model that cannot be shipped.
 */
import * as tf from '@tensorflow/tfjs';
import { supportedLayers, UnsupportedLayerError } from 'tensorflow-web';
import type { LayerSpec, ModelSpec, ModelSummary } from './types';

/** Adds one spec'd layer to a sequential model, applying the input shape once. */
function addLayer(model: tf.Sequential, spec: LayerSpec, inputShape?: number[]): void {
  const input = inputShape ? { inputShape } : {};
  switch (spec.type) {
    case 'conv2d':
      model.add(
        tf.layers.conv2d({
          ...input,
          filters: spec.filters,
          kernelSize: spec.kernelSize,
          padding: spec.padding,
          activation: spec.activation === 'relu' ? 'relu' : undefined,
        }),
      );
      break;
    case 'depthwiseConv2d':
      model.add(
        tf.layers.depthwiseConv2d({
          ...input,
          kernelSize: spec.kernelSize,
          padding: spec.padding,
          activation: spec.activation === 'relu' ? 'relu' : undefined,
        }),
      );
      break;
    case 'maxPool2d':
      model.add(tf.layers.maxPooling2d({ ...input, poolSize: spec.poolSize }));
      break;
    case 'avgPool2d':
      model.add(tf.layers.averagePooling2d({ ...input, poolSize: spec.poolSize }));
      break;
    case 'globalAvgPool2d':
      model.add(tf.layers.globalAveragePooling2d({ ...input }));
      break;
    case 'flatten':
      model.add(tf.layers.flatten({ ...input }));
      break;
    case 'dense':
      model.add(
        tf.layers.dense({
          ...input,
          units: spec.units,
          activation:
            spec.activation === 'none' ? undefined : (spec.activation as 'relu' | 'softmax'),
        }),
      );
      break;
  }
}

/**
 * Builds and compiles a model from a spec.
 *
 * @param spec the architecture to build.
 * @param inputShape input shape without the batch dimension.
 * @param learningRate Adam learning rate.
 * @returns a compiled tf.LayersModel ready to train.
 * @throws if the spec produces a layer the export registry does not support.
 */
export function buildModel(spec: ModelSpec, inputShape: number[], learningRate = 0.01): tf.LayersModel {
  if (spec.layers.length === 0) throw new Error('A model needs at least one layer.');
  const model = tf.sequential();
  spec.layers.forEach((layer, i) => addLayer(model, layer, i === 0 ? inputShape : undefined));
  model.compile({
    optimizer: tf.train.adam(learningRate),
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy'],
  });
  assertExportable(model);
  return model;
}

/**
 * Checks every layer against the export op registry and throws on the first
 * unsupported one, naming it. This is the same constraint the library enforces
 * at export; failing here means the user finds out before training, not after.
 *
 * @throws UnsupportedLayerError naming the first unsupported layer class.
 */
export function assertExportable(model: tf.LayersModel): void {
  const supported = new Set(supportedLayers());
  for (const layer of model.layers) {
    const className = layer.getClassName();
    // tfjs inserts an InputLayer that carries no weights and no op; the library
    // walks real layers only, so it is not part of the export constraint.
    if (className === 'InputLayer') continue;
    if (!supported.has(className)) {
      throw new UnsupportedLayerError(className, supportedLayers());
    }
  }
}

/** Product of a shape's dimensions, treating a null batch dimension as 1. */
function shapeElems(shape: Array<number | null>): number {
  return shape.reduce<number>((n, d) => n * (d ?? 1), 1);
}

/**
 * Summarizes a built model for the Model stage: parameter count, a rough int8
 * weight size, and the layer count.
 */
export function summarize(model: tf.LayersModel): ModelSummary {
  const paramCount = model.countParams();
  return {
    paramCount,
    // int8 weights are one byte each; add a small allowance for scales and bias.
    estimatedWeightBytes: Math.round(paramCount * 1.1),
    layerCount: model.layers.filter((l) => l.getClassName() !== 'InputLayer').length,
  };
}

/**
 * Estimates the runtime tensor arena in bytes: the largest sum of two adjacent
 * activation tensors that must be live at once, quantized to int8. This is a
 * floor the interpreter refines; the device budget pads it. See PLAN section 8.
 */
export function estimateArenaBytes(model: tf.LayersModel): number {
  // The activation sizes in order: the model input, then each layer's output.
  const sizes: number[] = [];
  const inShape = model.inputs[0]?.shape ?? [];
  sizes.push(shapeElems(inShape));
  for (const layer of model.layers) {
    const out = layer.outputShape;
    const shape = (Array.isArray(out[0]) ? out[0] : out) as Array<number | null>;
    sizes.push(shapeElems(shape));
  }
  let maxPair = 0;
  for (let i = 1; i < sizes.length; i++) {
    maxPair = Math.max(maxPair, sizes[i - 1]! + sizes[i]!);
  }
  // int8 activations are one byte each; pad for the float input and output
  // boundary tensors and interpreter bookkeeping.
  return Math.round(maxPair + sizes[0]! * 4 + 2048);
}
