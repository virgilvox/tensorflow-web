/**
 * Test app for tensorflow-web. Drives the whole loop in the browser:
 * generate a self contained synthetic shape dataset, build and train a small
 * convolutional model, calibrate, quantize to int8, emit a real .tflite with
 * toTFLite, verify the emitted file against the float reference, and offer a
 * download. Nothing here is published. The library stays headless; this file
 * owns every visual decision.
 */
import * as tf from '@tensorflow/tfjs';
import {
  train,
  calibrate,
  quantize,
  toTFLite,
  verify,
  type VerifyReport,
} from 'tensorflow-web';

// The three shapes the model learns to tell apart. Index is the class label.
const CLASS_NAMES = ['square', 'disc', 'ring'] as const;

interface DatasetSpec {
  sampleCount: number;
  imageSize: number;
}

interface Dataset {
  xs: tf.Tensor4D;
  ys: tf.Tensor2D;
  labels: number[];
  imageSize: number;
}

// Document references to the controls and readouts once.
const el = {
  sampleCount: byId<HTMLInputElement>('sample-count'),
  imageSize: byId<HTMLInputElement>('image-size'),
  epochs: byId<HTMLInputElement>('epochs'),
  batchSize: byId<HTMLInputElement>('batch-size'),
  tolerance: byId<HTMLInputElement>('tolerance'),
  run: byId<HTMLButtonElement>('run'),
  download: byId<HTMLButtonElement>('download'),
  status: byId<HTMLParagraphElement>('status'),
  epochValue: byId<HTMLSpanElement>('epoch-value'),
  lossValue: byId<HTMLSpanElement>('loss-value'),
  accValue: byId<HTMLSpanElement>('acc-value'),
  lossLog: byId<HTMLPreElement>('loss-log'),
  report: byId<HTMLPreElement>('report'),
};

let lastTflite: Uint8Array | null = null;

function byId<T extends HTMLElement>(id: string): T {
  const node = document.getElementById(id);
  if (!node) throw new Error(`Missing element #${id}`);
  return node as T;
}

function numberFrom(input: HTMLInputElement, fallback: number): number {
  const value = Number(input.value);
  return Number.isFinite(value) && value > 0 ? value : fallback;
}

function setStatus(text: string, state: 'idle' | 'busy' | 'done' | 'error' = 'idle'): void {
  el.status.textContent = text;
  el.status.dataset.state = state;
}

/**
 * Draws one procedural grayscale image into a flat row major buffer. Shapes are
 * generated from geometry alone, so the dataset needs no assets and no network.
 * A small amount of jitter on size and position keeps the classes learnable
 * rather than memorizable.
 */
function drawShape(label: number, size: number): Float32Array {
  const out = new Float32Array(size * size);
  const cx = size / 2 + (Math.random() - 0.5) * size * 0.15;
  const cy = size / 2 + (Math.random() - 0.5) * size * 0.15;
  const radius = size * (0.28 + Math.random() * 0.08);
  const half = radius;

  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const dx = x + 0.5 - cx;
      const dy = y + 0.5 - cy;
      const dist = Math.sqrt(dx * dx + dy * dy);
      let on = false;
      if (label === 0) {
        // Filled square.
        on = Math.abs(dx) <= half && Math.abs(dy) <= half;
      } else if (label === 1) {
        // Filled disc.
        on = dist <= radius;
      } else {
        // Ring: an annulus.
        on = dist <= radius && dist >= radius * 0.55;
      }
      out[y * size + x] = on ? 1 : 0;
    }
  }
  // Light pixel noise so the model sees variation.
  for (let i = 0; i < out.length; i++) {
    out[i] = Math.min(1, Math.max(0, (out[i] ?? 0) + (Math.random() - 0.5) * 0.1));
  }
  return out;
}

/**
 * Builds a balanced synthetic dataset of shapes. Returns image tensors shaped
 * [N, size, size, 1], one hot labels shaped [N, 3], and the integer labels for
 * later confusion reporting. The caller owns the returned tensors.
 */
function makeDataset(spec: DatasetSpec): Dataset {
  const { sampleCount, imageSize } = spec;
  const total = sampleCount * CLASS_NAMES.length;
  const buffer = new Float32Array(total * imageSize * imageSize);
  const labels: number[] = [];

  let offset = 0;
  for (let c = 0; c < CLASS_NAMES.length; c++) {
    for (let n = 0; n < sampleCount; n++) {
      const img = drawShape(c, imageSize);
      buffer.set(img, offset);
      offset += img.length;
      labels.push(c);
    }
  }

  // Shuffle index order so classes are interleaved before training.
  const order = shuffledIndices(total);
  const xs = tf.tidy(() => {
    const flat = tf.tensor4d(buffer, [total, imageSize, imageSize, 1]);
    const gathered = tf.gather(flat, order);
    return gathered as tf.Tensor4D;
  });
  const shuffledLabels = order.map((i) => labels[i] as number);
  const ys = tf.tidy(
    () => tf.oneHot(tf.tensor1d(shuffledLabels, 'int32'), CLASS_NAMES.length) as tf.Tensor2D,
  );

  return { xs, ys, labels: shuffledLabels, imageSize };
}

function shuffledIndices(n: number): number[] {
  const order = Array.from({ length: n }, (_, i) => i);
  for (let i = n - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    const tmp = order[i] as number;
    order[i] = order[j] as number;
    order[j] = tmp;
  }
  return order;
}

/**
 * Builds a small convolutional classifier sized for the synthetic task. Every
 * layer is one the export op registry supports: Conv2D, MaxPooling2D, Flatten,
 * and Dense with a softmax head.
 */
function buildModel(imageSize: number): tf.LayersModel {
  const model = tf.sequential();
  model.add(
    tf.layers.conv2d({
      inputShape: [imageSize, imageSize, 1],
      filters: 8,
      kernelSize: 3,
      padding: 'same',
      activation: 'relu',
    }),
  );
  model.add(tf.layers.maxPooling2d({ poolSize: 2 }));
  model.add(
    tf.layers.conv2d({
      filters: 16,
      kernelSize: 3,
      padding: 'same',
      activation: 'relu',
    }),
  );
  model.add(tf.layers.maxPooling2d({ poolSize: 2 }));
  model.add(tf.layers.flatten());
  model.add(tf.layers.dense({ units: 16, activation: 'relu' }));
  model.add(tf.layers.dense({ units: CLASS_NAMES.length, activation: 'softmax' }));

  model.compile({
    optimizer: tf.train.adam(0.01),
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy'],
  });
  return model;
}

function fmt(value: number | undefined, digits = 4): string {
  return value === undefined || Number.isNaN(value) ? '—' : value.toFixed(digits);
}

function renderReport(report: VerifyReport): string {
  const lines = [
    `parity        ${report.parity ? 'pass' : 'fail'}`,
    `maxAbsError   ${fmt(report.maxAbsError, 6)}`,
    `samples       ${report.sampleCount}`,
  ];
  if (report.floatAcc !== undefined) lines.push(`floatAcc      ${fmt(report.floatAcc)}`);
  if (report.int8Acc !== undefined) lines.push(`int8Acc       ${fmt(report.int8Acc)}`);
  if (report.arenaBytes !== undefined) lines.push(`arenaBytes    ${report.arenaBytes}`);
  if (report.confusion) {
    lines.push('');
    lines.push('confusion (rows = true label)');
    lines.push(`              ${CLASS_NAMES.map((n) => n.padStart(7)).join(' ')}`);
    report.confusion.forEach((row, i) => {
      const name = (CLASS_NAMES[i] ?? String(i)).padEnd(8);
      lines.push(`  ${name}  ${row.map((v) => String(v).padStart(7)).join(' ')}`);
    });
  }
  return lines.join('\n');
}

function disposeDataset(data: Dataset): void {
  data.xs.dispose();
  data.ys.dispose();
}

/** Runs the full pipeline once, surfacing every error in the status area. */
async function runPipeline(): Promise<void> {
  el.run.disabled = true;
  el.download.disabled = true;
  lastTflite = null;
  el.lossLog.textContent = '';
  el.report.textContent = 'No report yet.';
  el.epochValue.textContent = '—';
  el.lossValue.textContent = '—';
  el.accValue.textContent = '—';

  const spec: DatasetSpec = {
    sampleCount: Math.round(numberFrom(el.sampleCount, 120)),
    imageSize: Math.round(numberFrom(el.imageSize, 16)),
  };
  const epochs = Math.round(numberFrom(el.epochs, 12));
  const batchSize = Math.round(numberFrom(el.batchSize, 32));
  const tolerance = numberFrom(el.tolerance, 0.1);

  let trainSet: Dataset | null = null;
  let testSet: Dataset | null = null;
  let model: tf.LayersModel | null = null;

  try {
    await tf.ready();

    setStatus('Generating dataset...', 'busy');
    trainSet = makeDataset(spec);
    testSet = makeDataset({ ...spec, sampleCount: Math.max(10, Math.round(spec.sampleCount / 4)) });

    setStatus('Building model...', 'busy');
    model = buildModel(spec.imageSize);

    setStatus('Training...', 'busy');
    const logLines: string[] = [];
    await train(model, {
      data: { xs: trainSet.xs, ys: trainSet.ys },
      epochs,
      batchSize,
      shuffle: true,
      onEpoch: (epoch, logs) => {
        const loss = typeof logs.loss === 'number' ? logs.loss : Number(logs.loss);
        const acc =
          typeof logs.acc === 'number' ? logs.acc : typeof logs.accuracy === 'number' ? logs.accuracy : undefined;
        el.epochValue.textContent = `${epoch + 1} / ${epochs}`;
        el.lossValue.textContent = fmt(loss);
        el.accValue.textContent = fmt(acc);
        logLines.push(`epoch ${String(epoch + 1).padStart(2)}   loss ${fmt(loss)}   acc ${fmt(acc)}`);
        el.lossLog.textContent = logLines.join('\n');
      },
    });

    setStatus('Calibrating...', 'busy');
    const calibration = await calibrate(model, trainSet.xs, { method: 'minmax' });

    setStatus('Quantizing to int8...', 'busy');
    const quantized = quantize(model, calibration);

    setStatus('Emitting .tflite...', 'busy');
    const tflite = await toTFLite(quantized);
    lastTflite = tflite;
    el.download.disabled = false;

    setStatus('Verifying against float reference...', 'busy');
    const report = await verify(
      tflite,
      model,
      { xs: testSet.xs, ys: testSet.ys },
      { tolerance },
    );
    el.report.textContent = renderReport(report);

    const verdict = report.parity ? 'parity holds' : 'parity FAILED';
    setStatus(
      `Done. ${tflite.length} bytes emitted, ${verdict} at tolerance ${tolerance}.`,
      report.parity ? 'done' : 'error',
    );
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    setStatus(`Error: ${message}`, 'error');
    // The .tflite may still be downloadable if the failure was in verify.
    if (lastTflite) el.download.disabled = false;
  } finally {
    el.run.disabled = false;
    if (model) model.dispose();
    if (trainSet) disposeDataset(trainSet);
    if (testSet) disposeDataset(testSet);
  }
}

/** Triggers a browser download of the last emitted .tflite. */
function downloadTflite(): void {
  if (!lastTflite) return;
  const view = new Uint8Array(lastTflite.length);
  view.set(lastTflite);
  const blob = new Blob([view.buffer], { type: 'application/octet-stream' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'model.tflite';
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

el.run.addEventListener('click', () => {
  void runPipeline();
});
el.download.addEventListener('click', downloadTflite);

setStatus('Ready. Set parameters and run the pipeline.', 'idle');
