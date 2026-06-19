/**
 * Runs the float model over representative data and records per tensor ranges
 * keyed by layer name. See DESIGN.md.
 */
import * as tf from '@tensorflow/tfjs';
import { toFusedActivation } from './ops/common';
import type {
  Calibration,
  CalibrateOptions,
  Model,
  RepresentativeData,
  TensorRange,
} from './types';

/** A single batch of representative input pulled from the data source. */
type Batch = tf.Tensor;

/**
 * Builds a model that exposes every layer activation, runs it over the
 * representative data, and records the observed numeric range of each
 * activation plus the model input.
 *
 * The returned ranges are keyed by `layer.name` for layer activations and by
 * the literal key `input` for the model input, matching the tensor names the
 * graph builder assigns. Each range is widened to include zero so the
 * quantizer can represent the zero point. Method `minmax` is implemented;
 * `percentile` throws because it is not implemented yet.
 *
 * @param model The trained float model to observe.
 * @param data Representative inputs: a single tensor, an array of tensors, or a
 *   tf.data.Dataset yielding tensors or `{ xs }` objects. Labels are ignored.
 * @param options Calibration options. `method` defaults to `minmax`.
 * @returns A Calibration with per tensor `{ min, max }` ranges, the method
 *   used, and the number of samples observed.
 * @throws Error when `method` is `percentile`, which is not implemented yet.
 * @throws Error when the data source yields no batches, leaving nothing to
 *   calibrate.
 */
export async function calibrate(
  model: Model,
  data: RepresentativeData,
  options?: CalibrateOptions,
): Promise<Calibration> {
  const method = options?.method ?? 'minmax';
  if (method === 'percentile') {
    throw new Error(
      'calibrate() method "percentile" is not implemented yet. Use "minmax".',
    );
  }

  // Every activation the quantizer needs a range for: each layer output, plus
  // the pre-activation of layers whose built in activation the converters lower
  // to a separate op, so that intermediate gets a scale too.
  const specs: OutputSpec[] = [
    ...model.layers.map((layer) => ({
      key: layer.name,
      symbolic: layer.output as tf.SymbolicTensor,
    })),
    ...preActivationOutputs(model),
  ];
  const activationModel = tf.model({ inputs: model.inputs, outputs: specs.map((s) => s.symbolic) });

  // Accumulate running min and max per output key. The input is keyed 'input';
  // every other output is keyed to match the tensor names the graph builder assigns.
  const outputKeys = ['input', ...specs.map((s) => s.key)];
  const mins = new Map<string, number>();
  const maxes = new Map<string, number>();

  let sampleCount = 0;

  const observe = (key: string, batchMin: number, batchMax: number): void => {
    const priorMin = mins.get(key);
    const priorMax = maxes.get(key);
    mins.set(key, priorMin === undefined ? batchMin : Math.min(priorMin, batchMin));
    maxes.set(key, priorMax === undefined ? batchMax : Math.max(priorMax, batchMax));
  };

  // Process one input batch: run the multi-output model, fold each activation
  // and the input into the running ranges, then dispose every intermediate.
  const processBatch = async (batch: Batch): Promise<void> => {
    sampleCount += batch.shape[0] ?? 0;

    const [inputMin, inputMax] = await tensorRange(batch);
    observe('input', inputMin, inputMax);

    const predicted = activationModel.predict(batch);
    const activations = Array.isArray(predicted) ? predicted : [predicted];
    try {
      for (let i = 0; i < activations.length; i++) {
        const spec = specs[i];
        const act = activations[i];
        if (spec === undefined || act === undefined) continue;
        const [aMin, aMax] = await tensorRange(act);
        observe(spec.key, aMin, aMax);
      }
    } finally {
      for (const act of activations) act.dispose();
    }
  };

  // The activation model reuses the source model's layer objects, so disposing
  // it would dispose those shared layers and corrupt the caller's model. It
  // allocates no new weight tensors, so leaving it for garbage collection is
  // correct. Per batch activations are disposed explicitly in processBatch.
  for await (const batch of iterateBatches(data)) {
    const owned = batch.owned;
    try {
      await processBatch(batch.tensor);
    } finally {
      if (owned) batch.tensor.dispose();
    }
  }

  if (sampleCount === 0) {
    throw new Error(
      'calibrate() received no representative samples. Provide at least one batch.',
    );
  }

  const ranges: Record<string, TensorRange> = {};
  for (const key of outputKeys) {
    const rawMin = mins.get(key);
    const rawMax = maxes.get(key);
    if (rawMin === undefined || rawMax === undefined) continue;
    // Widen to include zero so the asymmetric zero point stays representable.
    ranges[key] = { min: Math.min(rawMin, 0), max: Math.max(rawMax, 0) };
  }

  return { ranges, method: 'minmax', sampleCount };
}

/** A named symbolic output to calibrate: an activation and the key to record it under. */
interface OutputSpec {
  key: string;
  symbolic: tf.SymbolicTensor;
}

/** Layer classes whose built in activation the converters split into a separate op. */
const SPLIT_SUFFIX: Record<string, string> = {
  Conv2D: '/conv',
  DepthwiseConv2D: '/conv',
  Dense: '/fc',
};

/**
 * Builds pre-activation (logits) outputs for layers whose built in activation is
 * lowered to its own op (softmax, sigmoid, tanh). The graph builder names that
 * intermediate `${layer.name}/conv` or `${layer.name}/fc`, and the quantizer
 * needs its range. A linear twin of the layer, sharing the trained weights,
 * reproduces the pre-activation exactly. Layers with a fusable or linear
 * activation are skipped because they produce no separate intermediate.
 *
 * @returns the extra activations to calibrate, keyed to match the graph builder.
 */
function preActivationOutputs(model: Model): OutputSpec[] {
  const extras: OutputSpec[] = [];
  for (const layer of model.layers) {
    const suffix = SPLIT_SUFFIX[layer.getClassName()];
    if (suffix === undefined) continue;
    if (toFusedActivation(layer.getConfig().activation) !== null) continue;
    const input = layer.input;
    if (Array.isArray(input)) continue;
    const twin = cloneLinear(layer);
    const symbolic = twin.apply(input) as tf.SymbolicTensor;
    twin.setWeights(layer.getWeights());
    extras.push({ key: `${layer.name}${suffix}`, symbolic });
  }
  return extras;
}

/**
 * Clones a conv or dense layer with a linear activation so its output is the
 * pre-activation value. The config from getConfig round trips through the same
 * layer factory tfjs uses to load models, and the trained weights are copied in
 * by the caller. Returns the new, unbuilt layer.
 */
function cloneLinear(layer: tf.layers.Layer): tf.layers.Layer {
  const config = { ...layer.getConfig(), activation: 'linear', name: `${layer.name}__preact` };
  delete (config as { batchInputShape?: unknown }).batchInputShape;
  delete (config as { inputShape?: unknown }).inputShape;
  switch (layer.getClassName()) {
    case 'Conv2D':
      return tf.layers.conv2d(config as unknown as Parameters<typeof tf.layers.conv2d>[0]);
    case 'DepthwiseConv2D':
      return tf.layers.depthwiseConv2d(
        config as unknown as Parameters<typeof tf.layers.depthwiseConv2d>[0],
      );
    case 'Dense':
      return tf.layers.dense(config as unknown as Parameters<typeof tf.layers.dense>[0]);
    default:
      throw new Error(`Cannot clone layer class "${layer.getClassName()}" for calibration.`);
  }
}

/**
 * Reads the minimum and maximum element of a tensor as plain numbers. The two
 * reduction tensors are disposed before returning so the call leaks nothing.
 *
 * @returns A `[min, max]` pair. An empty tensor yields `[+Infinity, -Infinity]`
 *   from tfjs, which the running fold then ignores.
 */
async function tensorRange(tensor: tf.Tensor): Promise<[number, number]> {
  const minTensor = tensor.min();
  const maxTensor = tensor.max();
  try {
    const minData = await minTensor.data();
    const maxData = await maxTensor.data();
    return [minData[0] ?? 0, maxData[0] ?? 0];
  } finally {
    minTensor.dispose();
    maxTensor.dispose();
  }
}

/** A batch tensor and whether the iterator created it (and must dispose it). */
interface OwnedBatch {
  tensor: tf.Tensor;
  owned: boolean;
}

/**
 * Normalizes the supported representative data shapes into a stream of input
 * batches. A raw tensor or array of tensors is yielded as is and left for the
 * caller to own. Dataset elements are unwrapped from `{ xs }` when present and
 * marked owned so the caller disposes them after use.
 */
async function* iterateBatches(
  data: RepresentativeData,
): AsyncGenerator<OwnedBatch> {
  if (data instanceof tf.Tensor) {
    yield { tensor: data, owned: false };
    return;
  }

  if (Array.isArray(data)) {
    for (const tensor of data) {
      yield { tensor, owned: false };
    }
    return;
  }

  // A tf.data.Dataset. Pull elements one at a time so each batch is disposed
  // before the next is materialized, which keeps memory flat over large sets.
  const iterator = await data.iterator();
  for (;;) {
    const next = await iterator.next();
    if (next.done) break;
    const element = next.value;
    if (element == null) continue;
    const tensor = element instanceof tf.Tensor ? element : (element as { xs: tf.Tensor }).xs;
    yield { tensor, owned: true };
  }
}
