/**
 * Browser parity check. Trains practical models on learnable tasks, emits a
 * float32 and an int8 .tflite through the real library, loads each back into the
 * actual TensorFlow Lite WASM interpreter, and measures inference accuracy.
 *
 * Two models are exercised so the whole op registry is verified numerically:
 *   cnn       conv, conv, maxpool, flatten, dense, dense, relu fusion, softmax
 *   separable conv, depthwise conv, pointwise conv, global average pool, dense
 *
 * For each, the float export must match the TensorFlow.js reference to within
 * floating point noise (proving the serializer and converters), and the int8
 * export must stay inside an error budget and keep the model's accuracy.
 *
 * The result is published on window.__parity for scripts/run-parity.mjs.
 */
import * as tf from '@tensorflow/tfjs';
import { buildFloatGraph, calibrate, quantize, toTFLite, train, verify } from 'tensorflow-web';

// alpha.8 is used for the CDN runtime because the alpha.10 npm package omits the
// WASM binaries. The web API is the same; loadTFLiteModel and setWasmPath match.
const TFLITE_CDN = 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-tflite@0.0.1-alpha.8/dist/';

type Report = Awaited<ReturnType<typeof verify>>;

interface ModelResult {
  floatBytes?: number;
  int8Bytes?: number;
  float?: Report;
  floatError?: string;
  int8?: Report;
  int8Error?: string;
}

interface ParityResult {
  done: boolean;
  error?: string;
  cnn?: ModelResult;
  separable?: ModelResult;
  mixed?: ModelResult;
  residual?: ModelResult;
}

function describe(e: unknown): string {
  return e instanceof Error ? (e.stack ?? e.message) : String(e);
}

function publish(result: ParityResult): void {
  (window as unknown as { __parity?: ParityResult }).__parity = result;
  const out = document.getElementById('out');
  if (out) out.textContent = JSON.stringify(result, null, 2);
}

/** A small seeded linear congruential generator, for reproducible data. */
function lcg(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
    return s / 2 ** 32;
  };
}

/**
 * Band task: the label is the horizontal band (top, middle, bottom) carrying a
 * bright stripe over a noise floor. Spatial, so it survives pooling and flatten.
 */
function bandData(n: number, seed: number): { xs: tf.Tensor; ys: tf.Tensor } {
  const rand = lcg(seed);
  const xs = new Float32Array(n * 64);
  const ys = new Float32Array(n * 3);
  for (let i = 0; i < n; i++) {
    const c = Math.floor(rand() * 3);
    ys[i * 3 + c] = 1;
    for (let r = 0; r < 8; r++) {
      const band = r < 3 ? 0 : r < 6 ? 1 : 2;
      for (let col = 0; col < 8; col++) {
        xs[i * 64 + r * 8 + col] = rand() * 0.3 + (band === c ? 1 : 0);
      }
    }
  }
  return { xs: tf.tensor4d(xs, [n, 8, 8, 1]), ys: tf.tensor2d(ys, [n, 3]) };
}

/**
 * Brightness task: the label is the overall brightness bin. Driven by the mean,
 * so it survives global average pooling, which the band task would not.
 */
function brightnessData(n: number, seed: number): { xs: tf.Tensor; ys: tf.Tensor } {
  const rand = lcg(seed);
  const base = [0.15, 0.5, 0.85];
  const xs = new Float32Array(n * 64);
  const ys = new Float32Array(n * 3);
  for (let i = 0; i < n; i++) {
    const c = Math.floor(rand() * 3);
    ys[i * 3 + c] = 1;
    for (let p = 0; p < 64; p++) xs[i * 64 + p] = Math.min(1, Math.max(0, base[c]! + (rand() - 0.5) * 0.2));
  }
  return { xs: tf.tensor4d(xs, [n, 8, 8, 1]), ys: tf.tensor2d(ys, [n, 3]) };
}

function cnnModel(): tf.LayersModel {
  const m = tf.sequential();
  m.add(tf.layers.conv2d({ inputShape: [8, 8, 1], filters: 8, kernelSize: 3, activation: 'relu' }));
  m.add(tf.layers.conv2d({ filters: 8, kernelSize: 3, activation: 'relu' }));
  m.add(tf.layers.maxPooling2d({ poolSize: 2 }));
  m.add(tf.layers.flatten());
  m.add(tf.layers.dense({ units: 16, activation: 'relu' }));
  m.add(tf.layers.dense({ units: 3, activation: 'softmax' }));
  return m;
}

function separableModel(): tf.LayersModel {
  const m = tf.sequential();
  m.add(
    tf.layers.conv2d({
      inputShape: [8, 8, 1],
      filters: 8,
      kernelSize: 3,
      padding: 'same',
      activation: 'relu',
    }),
  );
  m.add(tf.layers.depthwiseConv2d({ kernelSize: 3, padding: 'same', activation: 'relu' }));
  m.add(tf.layers.conv2d({ filters: 16, kernelSize: 1, activation: 'relu' }));
  m.add(tf.layers.globalAveragePooling2d({ dataFormat: 'channelsLast' }));
  m.add(tf.layers.dense({ units: 3, activation: 'softmax' }));
  return m;
}

function mixedModel(): tf.LayersModel {
  // Exercises AVERAGE_POOL_2D and a hidden sigmoid (LOGISTIC, fixed int8 output
  // scale 1/256). The band task survives 2x2 average pooling.
  const m = tf.sequential();
  m.add(tf.layers.conv2d({ inputShape: [8, 8, 1], filters: 8, kernelSize: 3, activation: 'relu' }));
  m.add(tf.layers.averagePooling2d({ poolSize: 2 }));
  m.add(tf.layers.flatten());
  m.add(tf.layers.dense({ units: 8, activation: 'sigmoid' }));
  m.add(tf.layers.dense({ units: 3, activation: 'softmax' }));
  return m;
}

function residualModel(): tf.LayersModel {
  // A functional model with a skip connection, to exercise the ADD op and the
  // fan out path in the graph builder. Brightness task survives global pooling.
  const input = tf.input({ shape: [8, 8, 1] });
  const a = tf.layers
    .conv2d({ filters: 8, kernelSize: 3, padding: 'same', activation: 'relu' })
    .apply(input) as tf.SymbolicTensor;
  const b = tf.layers
    .conv2d({ filters: 8, kernelSize: 3, padding: 'same', activation: 'relu' })
    .apply(a) as tf.SymbolicTensor;
  const sum = tf.layers.add().apply([a, b]) as tf.SymbolicTensor;
  const pooled = tf.layers
    .globalAveragePooling2d({ dataFormat: 'channelsLast' })
    .apply(sum) as tf.SymbolicTensor;
  const out = tf.layers.dense({ units: 3, activation: 'softmax' }).apply(pooled) as tf.SymbolicTensor;
  return tf.model({ inputs: input, outputs: out });
}

async function runModel(
  model: tf.LayersModel,
  trainData: { xs: tf.Tensor; ys: tf.Tensor },
  testData: { xs: tf.Tensor; ys: tf.Tensor },
): Promise<ModelResult> {
  model.compile({
    optimizer: tf.train.adam(0.01),
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy'],
  });
  await train(model, { data: trainData, epochs: 30, batchSize: 32 });

  const result: ModelResult = {};
  try {
    const floatBytes = await toTFLite({ graph: buildFloatGraph(model), scheme: 'float' });
    result.floatBytes = floatBytes.length;
    result.float = await verify(floatBytes, model, testData, { tolerance: 1e-3 });
  } catch (e) {
    result.floatError = describe(e);
  }
  try {
    const calibration = await calibrate(model, trainData.xs);
    const int8Bytes = await toTFLite(quantize(model, calibration));
    result.int8Bytes = int8Bytes.length;
    result.int8 = await verify(int8Bytes, model, testData, { tolerance: 0.06 });
  } catch (e) {
    result.int8Error = describe(e);
  }
  return result;
}

async function run(): Promise<ParityResult> {
  await tf.setBackend('cpu');
  await tf.ready();

  const tfliteGlobal = (window as unknown as { tflite?: { setWasmPath?: (p: string) => void } })
    .tflite;
  tfliteGlobal?.setWasmPath?.(TFLITE_CDN);

  const cnn = await runModel(cnnModel(), bandData(256, 1), bandData(64, 999));
  const separable = await runModel(
    separableModel(),
    brightnessData(256, 2),
    brightnessData(64, 888),
  );
  const mixed = await runModel(mixedModel(), bandData(256, 5), bandData(64, 555));
  const residual = await runModel(residualModel(), brightnessData(256, 7), brightnessData(64, 333));
  return { done: true, cnn, separable, mixed, residual };
}

run()
  .then(publish)
  .catch((e: unknown) => publish({ done: true, error: describe(e) }));
