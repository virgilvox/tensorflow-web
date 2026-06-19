import { describe, it, expect } from 'vitest';
import { useCpu } from './helpers/tf';
import { train } from '../src/train';

/** Builds and compiles a tiny dense regression model. */
async function tinyModel() {
  const tf = await useCpu();
  const model = tf.sequential();
  model.add(tf.layers.dense({ units: 4, inputShape: [3], activation: 'relu' }));
  model.add(tf.layers.dense({ units: 1 }));
  model.compile({ optimizer: 'sgd', loss: 'meanSquaredError' });
  return model;
}

describe('train', () => {
  it('runs the requested epochs and forwards onEpoch with a loss history', async () => {
    const tf = await useCpu();
    const xs = tf.randomNormal([16, 3]);
    const ys = tf.randomNormal([16, 1]);
    const model = await tinyModel();

    const epochsSeen: number[] = [];
    let batchCalls = 0;

    const { model: out, history } = await train(model, {
      data: { xs, ys },
      epochs: 2,
      batchSize: 4,
      onEpoch: (epoch) => {
        epochsSeen.push(epoch);
      },
      onBatch: () => {
        batchCalls += 1;
      },
    });

    expect(out).toBe(model);
    expect(epochsSeen).toEqual([0, 1]);
    expect(batchCalls).toBeGreaterThan(0);
    expect(Array.isArray(history.history.loss)).toBe(true);
    expect((history.history.loss as number[]).length).toBe(2);

    tf.dispose([xs, ys]);
    model.dispose();
  });

  it('trains from a tf.data.Dataset and reports a loss history', async () => {
    const tf = await useCpu();
    const model = await tinyModel();

    const dataset = tf.data
      .zip({
        xs: tf.data.array([
          [0, 0, 0],
          [1, 1, 1],
          [2, 2, 2],
          [3, 3, 3],
        ]),
        ys: tf.data.array([[0], [1], [2], [3]]),
      })
      .batch(2) as unknown as import('@tensorflow/tfjs').data.Dataset<{
      xs: import('@tensorflow/tfjs').Tensor;
      ys: import('@tensorflow/tfjs').Tensor;
    }>;

    const epochsSeen: number[] = [];
    const { history } = await train(model, {
      data: dataset,
      epochs: 2,
      onEpoch: (epoch) => {
        epochsSeen.push(epoch);
      },
    });

    expect(epochsSeen).toEqual([0, 1]);
    expect((history.history.loss as number[]).length).toBe(2);

    model.dispose();
  });

  it('stops early when the signal is already aborted before training', async () => {
    const tf = await useCpu();
    const xs = tf.randomNormal([16, 3]);
    const ys = tf.randomNormal([16, 1]);
    const model = await tinyModel();

    const controller = new AbortController();
    controller.abort();

    const epochsSeen: number[] = [];
    const { model: out, history } = await train(model, {
      data: { xs, ys },
      epochs: 5,
      batchSize: 4,
      signal: controller.signal,
      onEpoch: (epoch) => {
        epochsSeen.push(epoch);
      },
    });

    expect(out).toBe(model);
    // With the signal aborted up front, training stops at the first epoch
    // boundary, so at most one epoch of loss is recorded.
    const loss = (history.history.loss as number[]) ?? [];
    expect(loss.length).toBeLessThanOrEqual(1);
    expect(epochsSeen.length).toBeLessThanOrEqual(1);

    tf.dispose([xs, ys]);
    model.dispose();
  });

  it('stops at the first epoch boundary when aborted during the first epoch', async () => {
    const tf = await useCpu();
    const xs = tf.randomNormal([16, 3]);
    const ys = tf.randomNormal([16, 1]);
    const model = await tinyModel();

    const controller = new AbortController();

    let epochsSeen = 0;
    const { history } = await train(model, {
      data: { xs, ys },
      epochs: 5,
      batchSize: 4,
      signal: controller.signal,
      onBatch: () => {
        // Abort while the first epoch is still running.
        controller.abort();
      },
      onEpoch: () => {
        epochsSeen += 1;
      },
    });

    expect(epochsSeen).toBe(1);
    expect((history.history.loss as number[]).length).toBe(1);

    tf.dispose([xs, ys]);
    model.dispose();
  });
});
