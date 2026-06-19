/**
 * End to end pipeline test in Node. Trains a small conv model, calibrates,
 * quantizes to int8, serializes a real .tflite, and runs the Node side
 * structural check on the bytes. This proves every module composes into one
 * working pipeline. Numerical parity against the real interpreter is a separate
 * browser check (verify), exercised by the test app, because tfjs-tflite is WASM.
 */
import { describe, it, expect } from 'vitest';
import { useCpu } from './helpers/tf';
import { train } from '../src/train';
import { calibrate } from '../src/calibrate';
import { quantize } from '../src/quantize';
import { buildFloatGraph } from '../src/convert/build-graph';
import { toTFLite } from '../src/emit';
import { structuralCheck } from '../src/verify';

describe('full pipeline (Node)', () => {
  it('trains, calibrates, quantizes int8, and emits a structurally valid .tflite', async () => {
    const tf = await useCpu();

    const model = tf.sequential();
    model.add(
      tf.layers.conv2d({ inputShape: [8, 8, 1], filters: 4, kernelSize: 3, activation: 'relu' }),
    );
    model.add(tf.layers.maxPooling2d({ poolSize: 2 }));
    model.add(tf.layers.flatten());
    model.add(tf.layers.dense({ units: 3, activation: 'softmax' }));
    model.compile({ optimizer: 'adam', loss: 'categoricalCrossentropy' });

    const xs = tf.randomNormal([16, 8, 8, 1]);
    const ys = tf.oneHot(tf.randomUniform([16], 0, 3, 'int32'), 3);

    let epochs = 0;
    await train(model, { data: { xs, ys }, epochs: 2, batchSize: 8, onEpoch: () => { epochs += 1; } });
    expect(epochs).toBe(2);

    const calibration = await calibrate(model, xs);
    const quantized = quantize(model, calibration);
    expect(quantized.scheme).toBe('int8');

    const bytes = await toTFLite(quantized);
    expect(new TextDecoder().decode(bytes.subarray(4, 8))).toBe('TFL3');

    const check = structuralCheck(bytes);
    expect(check.ok).toBe(true);
    expect(check.issues).toEqual([]);
    expect(check.operatorCount).toBeGreaterThan(0);

    xs.dispose();
    ys.dispose();
    model.dispose();
  });

  it('emits a float32 .tflite that passes the structural check', async () => {
    const tf = await useCpu();

    const model = tf.sequential();
    model.add(tf.layers.dense({ units: 4, inputShape: [6], activation: 'relu' }));
    model.add(tf.layers.dense({ units: 2, activation: 'softmax' }));

    const bytes = await toTFLite({ graph: buildFloatGraph(model), scheme: 'float' });
    const check = structuralCheck(bytes);
    expect(check.ok).toBe(true);

    model.dispose();
  });
});
