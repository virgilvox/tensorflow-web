/**
 * Tensor assembly helpers. Stacks per sample feature vectors into the batched
 * tensors the library's train, calibrate, and verify calls expect, and one hot
 * encodes labels. The caller owns disposing the returned tensors. These are the
 * only place outside the composables that touches tfjs tensors directly.
 */
import * as tf from '@tensorflow/tfjs';
import type { Sample } from '../types';

export interface BatchedData {
  /** Stacked features, shape [n, ...featureShape]. */
  xs: tf.Tensor;
  /** One hot labels, shape [n, classCount]. */
  ys: tf.Tensor2D;
  /** Integer label per row, for confusion reporting. */
  labels: number[];
}

/**
 * Builds batched tensors from samples. Each sample is run through featureFn to a
 * flat Float32Array, then reshaped to [n, ...featureShape]. Labels are the index
 * of each sample's class within classIds, one hot encoded.
 *
 * @param samples the samples to batch, in the order returned.
 * @param classIds the class id order that defines the label index of each class.
 * @param featureFn turns one sample into its flat feature vector.
 * @param featureShape the per sample shape without the batch dimension.
 * @returns the batched xs and ys tensors plus the integer labels.
 * @throws if a sample references a class id not present in classIds.
 */
export function buildBatchedData(
  samples: Sample[],
  classIds: string[],
  featureFn: (sample: Sample) => Float32Array,
  featureShape: number[],
): BatchedData {
  const n = samples.length;
  const perSample = featureShape.reduce((a, b) => a * b, 1);
  const flat = new Float32Array(n * perSample);
  const labels: number[] = new Array(n);
  const index = new Map(classIds.map((id, i) => [id, i]));

  samples.forEach((sample, i) => {
    const labelIndex = index.get(sample.classId);
    if (labelIndex === undefined) {
      throw new Error(`Sample ${sample.id} references unknown class ${sample.classId}`);
    }
    labels[i] = labelIndex;
    flat.set(featureFn(sample), i * perSample);
  });

  const xs = tf.tensor(flat, [n, ...featureShape]);
  const ys = tf.tidy(
    () => tf.oneHot(tf.tensor1d(labels, 'int32'), classIds.length) as tf.Tensor2D,
  );
  return { xs, ys, labels };
}
