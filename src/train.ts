/**
 * Thin wrapper over model.fit and model.fitDataset that forwards progress
 * callbacks and supports cooperative cancellation. See DESIGN.md.
 */
import type * as tf from '@tensorflow/tfjs';
import type { DataSource, Logs, Model, TrainOptions, TrainResult } from './types';

/** Narrows a DataSource to the raw tensor pair shape. */
function isTensorPair(data: DataSource): data is { xs: tf.Tensor; ys: tf.Tensor } {
  return (
    typeof (data as { xs?: unknown }).xs !== 'undefined' &&
    typeof (data as { ys?: unknown }).ys !== 'undefined'
  );
}

/**
 * Trains an already compiled model and forwards progress events.
 *
 * Accepts training data as either a `{ xs, ys }` tensor pair, which is sent to
 * `model.fit`, or a `tf.data.Dataset`, which is sent to `model.fitDataset`.
 * `onEpoch` is invoked from the underlying `onEpochEnd` callback once per epoch,
 * and `onBatch` from `onBatchEnd` once per training batch. When `options.signal`
 * is provided and aborted, training stops at the next epoch boundary by setting
 * `model.stopTraining`, and the call resolves with the model trained so far. If
 * the signal is already aborted before training starts, no epochs run.
 *
 * The model must already be compiled by the caller; this function does not
 * compile it.
 *
 * @param model The compiled tfjs LayersModel to train.
 * @param options Data, epoch and batch counts, validation settings, progress
 *   callbacks, and an optional AbortSignal.
 * @returns The trained model and the tfjs History for the run.
 * @throws Whatever `model.fit` or `model.fitDataset` throws, for example when
 *   the model was not compiled or the data shapes do not match the model.
 */
export async function train(model: Model, options: TrainOptions): Promise<TrainResult> {
  const { data, epochs, batchSize, validationSplit, validationData, shuffle, onEpoch, onBatch, signal } =
    options;

  // Build the shared callbacks. onEpochEnd is the single place we honor an
  // abort: stopping there lets the in flight epoch finish so the weights and
  // History stay consistent, then tfjs ends the run cleanly.
  const callbacks: tf.CustomCallbackArgs = {
    onEpochEnd: async (epoch: number, logs?: Logs) => {
      if (onEpoch) await onEpoch(epoch, (logs ?? {}) as Logs);
      if (signal?.aborted) model.stopTraining = true;
    },
    onBatchEnd: async (batch: number, logs?: Logs) => {
      if (onBatch) await onBatch(batch, (logs ?? {}) as Logs);
    },
  };

  // Honor an already aborted signal without running any epoch.
  if (signal?.aborted) {
    model.stopTraining = true;
  }

  let history: tf.History;
  if (isTensorPair(data)) {
    history = await model.fit(data.xs, data.ys, {
      epochs,
      batchSize,
      validationSplit,
      validationData: validationData
        ? toTensorValidation(validationData)
        : undefined,
      shuffle,
      callbacks,
    });
  } else {
    history = await model.fitDataset(data, {
      epochs,
      validationData: validationData ? toDatasetValidation(validationData) : undefined,
      callbacks,
    });
  }

  // Reset the flag so a later training call on the same model is not blocked by
  // a stop left over from this run.
  model.stopTraining = false;

  return { model, history };
}

/**
 * Coerces a DataSource into the `[xVal, yVal]` tensor tuple that model.fit
 * accepts as validationData. model.fit does not validate against a Dataset, so
 * a Dataset validation source is rejected when training from raw tensors.
 */
function toTensorValidation(data: DataSource): [tf.Tensor, tf.Tensor] {
  if (isTensorPair(data)) return [data.xs, data.ys];
  throw new Error(
    'validationData for a { xs, ys } training run must itself be a { xs, ys } tensor pair, not a tf.data.Dataset.',
  );
}

/**
 * Coerces a DataSource into the validationData shape model.fitDataset accepts.
 * fitDataset validates against a Dataset, so a raw tensor pair is not a valid
 * validation source here and is rejected.
 */
function toDatasetValidation(
  data: DataSource,
): tf.data.Dataset<{ xs: tf.Tensor; ys: tf.Tensor }> {
  if (isTensorPair(data)) {
    throw new Error(
      'validationData for a Dataset training run must itself be a tf.data.Dataset, not a { xs, ys } tensor pair.',
    );
  }
  return data;
}
