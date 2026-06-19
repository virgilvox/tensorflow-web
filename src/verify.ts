/**
 * The trust layer. Loads the emitted .tflite in tfjs-tflite and compares its
 * outputs to the float reference model. The interpreter is a browser WASM build,
 * so verify runs in the browser; it imports tfjs-tflite dynamically so Node can
 * import the rest of the library without pulling in WASM.
 *
 * structuralCheck is a Node side sanity pass that parses the bytes and confirms
 * the model is well formed. See DESIGN.md.
 */
import * as flatbuffers from 'flatbuffers';
import type * as tf from '@tensorflow/tfjs';
import { Model } from './serialize/schema/tflite';
import type { DataSource, Model as ReferenceModel, VerifyReport } from './types';

export interface StructuralCheck {
  ok: boolean;
  issues: string[];
  tensorCount: number;
  operatorCount: number;
}

/**
 * A flattened view of a parsed subgraph. structuralCheck reads this out of the
 * FlatBuffers model so the range validation logic is a pure function over plain
 * numbers and can be unit tested without building a flatbuffer.
 */
export interface ModelShape {
  /** Number of operator codes declared on the model. */
  operatorCodeCount: number;
  /** Number of buffers declared on the model. */
  bufferCount: number;
  /** Number of tensors in the single subgraph. */
  tensorCount: number;
  /** opcodeIndex for each operator, in order. */
  operatorOpcodeIndices: number[];
  /** buffer index for each tensor, in order. */
  tensorBufferIndices: number[];
  /** Subgraph input tensor indices. */
  inputs: number[];
  /** Subgraph output tensor indices. */
  outputs: number[];
}

/**
 * Validates a flattened model shape: every operator opcodeIndex is in range,
 * every tensor buffer index is in range, and every subgraph input and output
 * references an existing tensor.
 *
 * @returns the list of human readable issues. An empty list means the shape is
 *   well formed. This function does not throw.
 */
export function validateModelShape(shape: ModelShape): string[] {
  const issues: string[] = [];

  shape.operatorOpcodeIndices.forEach((opcodeIndex, i) => {
    if (opcodeIndex < 0 || opcodeIndex >= shape.operatorCodeCount) {
      issues.push(
        `operator ${i} opcodeIndex ${opcodeIndex} is out of range ` +
          `(have ${shape.operatorCodeCount} operator codes)`,
      );
    }
  });

  shape.tensorBufferIndices.forEach((bufferIndex, i) => {
    if (bufferIndex < 0 || bufferIndex >= shape.bufferCount) {
      issues.push(
        `tensor ${i} buffer index ${bufferIndex} is out of range ` +
          `(have ${shape.bufferCount} buffers)`,
      );
    }
  });

  shape.inputs.forEach((tensorIndex, i) => {
    if (tensorIndex < 0 || tensorIndex >= shape.tensorCount) {
      issues.push(
        `subgraph input ${i} references tensor ${tensorIndex}, ` +
          `which does not exist (have ${shape.tensorCount} tensors)`,
      );
    }
  });

  shape.outputs.forEach((tensorIndex, i) => {
    if (tensorIndex < 0 || tensorIndex >= shape.tensorCount) {
      issues.push(
        `subgraph output ${i} references tensor ${tensorIndex}, ` +
          `which does not exist (have ${shape.tensorCount} tensors)`,
      );
    }
  });

  return issues;
}

/**
 * Parses .tflite bytes with the generated FlatBuffers reader and checks that the
 * model is well formed: the TFL3 file identifier is present, there is exactly
 * one subgraph, every operator opcodeIndex is in range, every tensor buffer
 * index is in range, and the subgraph inputs and outputs reference existing
 * tensors.
 *
 * This runs in Node and never touches WASM. It does not throw on a malformed
 * model; every problem is reported through the returned issues list and the ok
 * flag. A genuinely corrupt buffer that the reader cannot parse is reported as
 * an issue rather than propagated.
 *
 * @returns ok true with an empty issues list when the model is well formed,
 *   along with the tensor and operator counts of the single subgraph.
 */
export function structuralCheck(tflite: Uint8Array): StructuralCheck {
  const issues: string[] = [];
  const bb = new flatbuffers.ByteBuffer(tflite);

  if (!Model.bufferHasIdentifier(bb)) {
    issues.push('missing TFL3 file identifier');
    return { ok: false, issues, tensorCount: 0, operatorCount: 0 };
  }

  let model: Model;
  try {
    model = Model.getRootAsModel(bb);
  } catch (err) {
    issues.push(`failed to parse model: ${(err as Error).message}`);
    return { ok: false, issues, tensorCount: 0, operatorCount: 0 };
  }

  const subgraphCount = model.subgraphsLength();
  if (subgraphCount !== 1) {
    issues.push(`expected exactly one subgraph, found ${subgraphCount}`);
    // Without a single subgraph there are no counts to report.
    return { ok: false, issues, tensorCount: 0, operatorCount: 0 };
  }

  const subgraph = model.subgraphs(0);
  if (!subgraph) {
    issues.push('subgraph 0 is null');
    return { ok: false, issues, tensorCount: 0, operatorCount: 0 };
  }

  const tensorCount = subgraph.tensorsLength();
  const operatorCount = subgraph.operatorsLength();

  const operatorOpcodeIndices: number[] = [];
  for (let i = 0; i < operatorCount; i++) {
    operatorOpcodeIndices.push(subgraph.operators(i)!.opcodeIndex());
  }

  const tensorBufferIndices: number[] = [];
  for (let i = 0; i < tensorCount; i++) {
    tensorBufferIndices.push(subgraph.tensors(i)!.buffer());
  }

  const inputs: number[] = [];
  for (let i = 0; i < subgraph.inputsLength(); i++) {
    inputs.push(subgraph.inputs(i)!);
  }

  const outputs: number[] = [];
  for (let i = 0; i < subgraph.outputsLength(); i++) {
    outputs.push(subgraph.outputs(i)!);
  }

  issues.push(
    ...validateModelShape({
      operatorCodeCount: model.operatorCodesLength(),
      bufferCount: model.buffersLength(),
      tensorCount,
      operatorOpcodeIndices,
      tensorBufferIndices,
      inputs,
      outputs,
    }),
  );

  return { ok: issues.length === 0, issues, tensorCount, operatorCount };
}

export interface VerifyOptions {
  tolerance?: number;
}

/** Default parity tolerance when the caller does not supply one. */
const DEFAULT_TOLERANCE = 1e-2;

/**
 * Pulls the test data into a single batched { xs, ys } pair of tensors. Accepts
 * a tfjs Dataset or an already materialized { xs, ys }. Labels (ys) may be
 * absent on a Dataset element, in which case ys is undefined.
 */
async function collectTestData(
  tfns: typeof tf,
  testData: DataSource,
): Promise<{ xs: tf.Tensor; ys?: tf.Tensor }> {
  if ('xs' in testData && testData.xs instanceof tfns.Tensor) {
    const pair = testData as { xs: tf.Tensor; ys: tf.Tensor };
    return { xs: pair.xs, ys: pair.ys };
  }

  const dataset = testData as tf.data.Dataset<{ xs: tf.Tensor; ys: tf.Tensor }>;
  const xsParts: tf.Tensor[] = [];
  const ysParts: tf.Tensor[] = [];
  await dataset.forEachAsync((batch) => {
    xsParts.push(batch.xs);
    if (batch.ys) ysParts.push(batch.ys);
  });
  const xs = xsParts.length === 1 ? xsParts[0]! : tfns.concat(xsParts);
  const ys =
    ysParts.length === 0 ? undefined : ysParts.length === 1 ? ysParts[0]! : tfns.concat(ysParts);
  return { xs, ys };
}

/** argmax along the last axis of a 2D [n, classes] tensor, as a number array. */
function argmaxRows(values: Float32Array, rows: number, cols: number): number[] {
  const out = new Array<number>(rows);
  for (let r = 0; r < rows; r++) {
    let best = 0;
    let bestVal = values[r * cols]!;
    for (let c = 1; c < cols; c++) {
      const v = values[r * cols + c]!;
      if (v > bestVal) {
        bestVal = v;
        best = c;
      }
    }
    out[r] = best;
  }
  return out;
}

/**
 * Loads the emitted .tflite back into tfjs-tflite, runs it over the test inputs,
 * runs the float reference model over the same inputs, and compares the two.
 *
 * Reports maxAbsError across all outputs and a parity boolean that is true when
 * the largest absolute elementwise error is within the tolerance
 * (options.tolerance, default 1e-2). When the test data carries labels it also
 * reports the float and int8 top-1 accuracy and a row indexed confusion matrix,
 * where rows are the true class and columns are the int8 model prediction.
 *
 * This path requires a browser. tfjs-tflite is a WebAssembly build, so it is
 * imported dynamically inside this function; importing this module in Node does
 * not pull in WASM. Calling verify in Node will reject when the dynamic import
 * of @tensorflow/tfjs-tflite fails to find a usable runtime.
 *
 * @returns a VerifyReport describing parity, error, optional accuracy, and the
 *   evaluated sample count.
 * @throws if the dynamic import of @tensorflow/tfjs-tflite fails (for example in
 *   a plain Node process with no WASM runtime), or if the interpreter cannot
 *   load the supplied bytes.
 */
/** A loaded TFLite interpreter, narrowed to the one method verify calls. */
interface Interpreter {
  predict(x: tf.Tensor): tf.Tensor | tf.Tensor[] | Record<string, tf.Tensor>;
}

/**
 * Loads a TFLite interpreter for the given model bytes. Prefers a global
 * `tflite` object, which is how @tensorflow/tfjs-tflite is exposed when it is
 * loaded from a script tag. That is the common browser setup and the one the
 * test app uses, and it avoids bundling the interpreter, whose WASM assets load
 * relative to its own dist. Falls back to a dynamic import of the npm package.
 */
async function loadInterpreter(buffer: ArrayBuffer): Promise<Interpreter> {
  const fromGlobal = (globalThis as { tflite?: { loadTFLiteModel(b: ArrayBuffer): Promise<Interpreter> } })
    .tflite;
  if (fromGlobal?.loadTFLiteModel) {
    return fromGlobal.loadTFLiteModel(buffer);
  }
  const mod = await import('@tensorflow/tfjs-tflite');
  return mod.loadTFLiteModel(buffer) as unknown as Promise<Interpreter>;
}

export async function verify(
  tflite: Uint8Array,
  reference: ReferenceModel,
  testData: DataSource,
  options?: VerifyOptions,
): Promise<VerifyReport> {
  const tolerance = options?.tolerance ?? DEFAULT_TOLERANCE;

  // Dynamic import keeps tfjs out of the Node import graph for this module. tfjs
  // is a peer dependency the host application provides.
  const tfns = (await import('@tensorflow/tfjs')) as typeof tf;

  // loadTFLiteModel takes an ArrayBuffer. Slice to the exact backing region so a
  // Uint8Array that views a larger buffer does not hand over trailing bytes.
  const buffer = tflite.buffer.slice(
    tflite.byteOffset,
    tflite.byteOffset + tflite.byteLength,
  ) as ArrayBuffer;
  const interpreter = await loadInterpreter(buffer);

  const { xs, ys } = await collectTestData(tfns, testData);
  const sampleCount = xs.shape[0] ?? 0;

  // The float reference runs the whole batch at once.
  const referenceOut = reference.predict(xs) as tf.Tensor;
  const refValues = referenceOut.dataSync() as Float32Array;
  const perSample = sampleCount > 0 ? refValues.length / sampleCount : refValues.length;

  // The emitted .tflite has a fixed input batch of 1, as TFLite models do, so
  // the interpreter is run one sample at a time and the outputs concatenated.
  // The emitted outputs come from the interpreter's own tf core; reading them as
  // plain data and comparing in plain arrays avoids mixing tensors across cores.
  const emitValues = new Float32Array(refValues.length);
  for (let i = 0; i < sampleCount; i++) {
    const sample = tfns.gather(xs, [i]);
    const out = asTensor(interpreter.predict(sample));
    const data = out.dataSync() as Float32Array;
    emitValues.set(data.subarray(0, perSample), i * perSample);
    out.dispose();
    sample.dispose();
  }

  let maxAbsError = 0;
  for (let i = 0; i < refValues.length; i++) {
    const d = Math.abs((refValues[i] as number) - (emitValues[i] as number));
    if (d > maxAbsError) maxAbsError = d;
  }
  const parity = maxAbsError <= tolerance;

  const report: VerifyReport = {
    parity,
    maxAbsError,
    sampleCount,
  };

  // Accuracy and confusion matrix only make sense with labels and class scores.
  if (ys && referenceOut.rank === 2) {
    const rows = sampleCount;
    const cols = referenceOut.shape[1] ?? perSample;

    // Labels may be one hot [n, classes] or a dense [n] class index vector.
    const labels =
      ys.rank === 2
        ? argmaxRows(ys.dataSync() as Float32Array, rows, ys.shape[1] ?? cols)
        : Array.from(ys.dataSync() as Float32Array).map((v) => Math.round(v));

    const refPred = argmaxRows(refValues, rows, cols);
    const emitPred = argmaxRows(emitValues, rows, cols);

    let floatCorrect = 0;
    let int8Correct = 0;
    const confusion: number[][] = Array.from({ length: cols }, () =>
      new Array<number>(cols).fill(0),
    );
    for (let r = 0; r < rows; r++) {
      const truth = labels[r]!;
      if (refPred[r] === truth) floatCorrect++;
      if (emitPred[r] === truth) int8Correct++;
      if (truth >= 0 && truth < cols) confusion[truth]![emitPred[r]!]!++;
    }
    report.floatAcc = rows === 0 ? 0 : floatCorrect / rows;
    report.int8Acc = rows === 0 ? 0 : int8Correct / rows;
    report.confusion = confusion;
  }

  referenceOut.dispose();

  return report;
}

/** Coerces an interpreter predict result to a single output tensor. */
function asTensor(out: tf.Tensor | tf.Tensor[] | Record<string, tf.Tensor>): tf.Tensor {
  if (Array.isArray(out)) return out[0] as tf.Tensor;
  if (typeof (out as tf.Tensor).dataSync === 'function') return out as tf.Tensor;
  return Object.values(out as Record<string, tf.Tensor>)[0] as tf.Tensor;
}
