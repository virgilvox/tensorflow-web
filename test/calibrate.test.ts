import { describe, it, expect } from 'vitest';
import { useCpu } from './helpers/tf';
import { calibrate } from '../src/calibrate';

describe('calibrate', () => {
  it('records ranges for input and every layer with min <= max', async () => {
    const tf = await useCpu();

    const model = tf.sequential();
    model.add(tf.layers.dense({ units: 3, inputShape: [4], activation: 'relu' }));
    model.add(tf.layers.dense({ units: 2, activation: 'softmax' }));

    const xs = tf.randomNormal([8, 4]);
    const cal = await calibrate(model, xs);

    expect(cal.method).toBe('minmax');
    expect(cal.sampleCount).toBe(8);

    expect(cal.ranges.input).toBeDefined();
    for (const layer of model.layers) {
      expect(cal.ranges[layer.name]).toBeDefined();
    }
    for (const key of Object.keys(cal.ranges)) {
      const r = cal.ranges[key]!;
      expect(r.min).toBeLessThanOrEqual(r.max);
      // Ranges are widened to include zero.
      expect(r.min).toBeLessThanOrEqual(0);
      expect(r.max).toBeGreaterThanOrEqual(0);
    }

    xs.dispose();
    model.dispose();
  });

  it('computes a known dense activation range from all ones input', async () => {
    const tf = await useCpu();

    // A single dense unit with weights all 1 and bias 0. Input all ones over 4
    // features gives a pre-activation of 4. linear activation keeps it at 4.
    const model = tf.sequential();
    const dense = tf.layers.dense({
      units: 1,
      inputShape: [4],
      activation: 'linear',
      useBias: false,
      kernelInitializer: 'ones',
    });
    model.add(dense);

    const xs = tf.ones([2, 4]);
    const cal = await calibrate(model, xs);

    const layerName = model.layers[0]!.name;
    const range = cal.ranges[layerName]!;
    expect(range.max).toBeCloseTo(4, 5);
    // Output is constant 4, widened to include zero.
    expect(range.min).toBeCloseTo(0, 5);

    // Input is all ones, widened to include zero.
    expect(cal.ranges.input!.max).toBeCloseTo(1, 5);
    expect(cal.ranges.input!.min).toBeCloseTo(0, 5);

    xs.dispose();
    model.dispose();
  });

  it('accepts an array of tensors and sums their sample counts', async () => {
    const tf = await useCpu();

    const model = tf.sequential();
    model.add(tf.layers.dense({ units: 2, inputShape: [3], activation: 'relu' }));

    const batches = [tf.ones([2, 3]), tf.ones([3, 3]).mul(-1)];
    const cal = await calibrate(model, batches);

    expect(cal.sampleCount).toBe(5);
    expect(cal.ranges.input!.min).toBeCloseTo(-1, 5);
    expect(cal.ranges.input!.max).toBeCloseTo(1, 5);

    for (const b of batches) b.dispose();
    model.dispose();
  });

  it('accepts a tf.data.Dataset of {xs} and disposes batches', async () => {
    const tf = await useCpu();

    const model = tf.sequential();
    model.add(tf.layers.dense({ units: 2, inputShape: [3], activation: 'relu' }));

    // A dataset with more batches than the prior cases. tf.data.array clones
    // the source tensors, so the only way these clones do not leak is calibrate
    // disposing each one. Run twice and require the per call tensor growth to be
    // identical: any per batch leak would scale with batch count and diverge.
    const makeDs = () =>
      tf.data.array(
        Array.from({ length: 4 }, (_unused, i) => ({ xs: tf.ones([2, 3]).mul(i + 1) })),
      );

    const start = tf.memory().numTensors;
    const cal = await calibrate(model, makeDs());
    const afterFirst = tf.memory().numTensors;
    await calibrate(model, makeDs());
    const afterSecond = tf.memory().numTensors;

    expect(cal.sampleCount).toBe(8);
    expect(cal.ranges.input!.max).toBeCloseTo(4, 5);
    // Constant per call overhead and no per batch accumulation.
    expect(afterSecond - afterFirst).toBe(afterFirst - start);

    model.dispose();
  });

  it('throws a clear error for the percentile method', async () => {
    const tf = await useCpu();
    const model = tf.sequential();
    model.add(tf.layers.dense({ units: 1, inputShape: [2] }));
    const xs = tf.ones([1, 2]);

    await expect(calibrate(model, xs, { method: 'percentile' })).rejects.toThrow(
      /not implemented yet/i,
    );

    xs.dispose();
    model.dispose();
  });

  it('throws when given no representative samples', async () => {
    const tf = await useCpu();
    const model = tf.sequential();
    model.add(tf.layers.dense({ units: 2, inputShape: [3] }));

    await expect(calibrate(model, [])).rejects.toThrow(/no representative samples/i);

    model.dispose();
  });
});
