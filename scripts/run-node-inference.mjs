/**
 * Proves the emitted .tflite is a portable artifact by running it in a Node
 * native TFLite runtime, independent of the browser.
 *
 * The flow mirrors what a user does: import the published build, train a small
 * model, quantize it to int8, and write a real .tflite file to disk. Then a
 * separate runtime (tfjs-tflite-node, the TensorFlow Lite C++ library bound to
 * Node through N-API) loads that file and runs inference. The output is compared
 * to the float TensorFlow.js reference. If a different runtime on a different
 * substrate than the browser WASM interpreter loads the file and computes the
 * right answers, the file is genuinely transportable.
 *
 * Run with: npm run test:node-tflite
 */
import * as tf from '@tensorflow/tfjs';
import { writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { loadTFLiteModel } from 'tfjs-tflite-node';
import { calibrate, quantize, toTFLite, train } from '../dist/index.js';

/** Seeded LCG so the run is reproducible. */
function lcg(seed) {
  let s = seed >>> 0;
  return () => {
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
    return s / 2 ** 32;
  };
}

/** Band task: the label is the horizontal band carrying a bright stripe. */
function bandData(n, seed) {
  const rand = lcg(seed);
  const xs = new Float32Array(n * 64);
  const ys = new Float32Array(n * 3);
  const labels = [];
  for (let i = 0; i < n; i++) {
    const c = Math.floor(rand() * 3);
    labels.push(c);
    ys[i * 3 + c] = 1;
    for (let r = 0; r < 8; r++) {
      const band = r < 3 ? 0 : r < 6 ? 1 : 2;
      for (let col = 0; col < 8; col++) xs[i * 64 + r * 8 + col] = rand() * 0.3 + (band === c ? 1 : 0);
    }
  }
  return { xs: tf.tensor4d(xs, [n, 8, 8, 1]), ys: tf.tensor2d(ys, [n, 3]), labels };
}

function argmax(arr) {
  let best = 0;
  for (let i = 1; i < arr.length; i++) if (arr[i] > arr[best]) best = i;
  return best;
}

async function main() {
  await tf.setBackend('cpu');
  await tf.ready();

  const model = tf.sequential();
  model.add(tf.layers.conv2d({ inputShape: [8, 8, 1], filters: 8, kernelSize: 3, activation: 'relu' }));
  model.add(tf.layers.maxPooling2d({ poolSize: 2 }));
  model.add(tf.layers.flatten());
  model.add(tf.layers.dense({ units: 16, activation: 'relu' }));
  model.add(tf.layers.dense({ units: 3, activation: 'softmax' }));
  model.compile({ optimizer: tf.train.adam(0.01), loss: 'categoricalCrossentropy' });

  const trainData = bandData(256, 1);
  const testData = bandData(64, 999);

  console.log('training a small CNN...');
  await train(model, { data: trainData, epochs: 25, batchSize: 32 });

  // Quantize to int8 and write a real .tflite file to disk.
  const bytes = await toTFLite(quantize(model, await calibrate(model, trainData.xs)));
  const file = join(tmpdir(), 'tensorflow-web-node-test.tflite');
  writeFileSync(file, bytes);
  console.log(`wrote ${file} (${bytes.length} bytes)`);

  // Load that file in the independent Node native TFLite runtime.
  const interpreter = await loadTFLiteModel(file);
  console.log('loaded the .tflite into tfjs-tflite-node (TFLite C++ via N-API)');

  // Run both models over the test set and compare.
  const refValues = await model.predict(testData.xs).array();
  const n = testData.xs.shape[0];
  let maxAbsError = 0;
  let floatCorrect = 0;
  let int8Correct = 0;
  for (let i = 0; i < n; i++) {
    const sample = tf.gather(testData.xs, [i]);
    const out = interpreter.predict(sample);
    const emit = await (Array.isArray(out) ? out[0] : out).data();
    sample.dispose();
    const ref = refValues[i];
    for (let k = 0; k < ref.length; k++) maxAbsError = Math.max(maxAbsError, Math.abs(ref[k] - emit[k]));
    if (argmax(ref) === testData.labels[i]) floatCorrect++;
    if (argmax(Array.from(emit)) === testData.labels[i]) int8Correct++;
  }

  const floatAcc = floatCorrect / n;
  const int8Acc = int8Correct / n;
  console.log('\n=== node inference result ===');
  console.log(`samples        ${n}`);
  console.log(`float accuracy ${floatAcc.toFixed(3)}`);
  console.log(`int8 accuracy  ${int8Acc.toFixed(3)} (in tfjs-tflite-node)`);
  console.log(`maxAbsError    ${maxAbsError.toExponential(3)}`);

  const ok = floatAcc >= 0.8 && floatAcc - int8Acc <= 0.15 && maxAbsError <= 0.2;
  if (ok) {
    console.log('\nThe emitted .tflite loaded and ran correctly in a Node TFLite runtime.');
    console.log('It is a transportable file: the same bytes run in the browser WASM');
    console.log('interpreter and in the native TFLite C++ library through Node.');
  } else {
    console.log('\nFAIL: the .tflite did not run correctly in the Node runtime.');
  }
  return ok ? 0 : 1;
}

main()
  .then((code) => process.exit(code))
  .catch((err) => {
    console.error('node inference test failed:', err);
    process.exit(1);
  });
