/**
 * Verifies emitted bytes against the float reference in the real TFLite
 * interpreter and returns the report the studio shows: parity, max absolute
 * error, the float versus int8 accuracy, and the confusion matrix. Sets the WASM
 * path on the global interpreter first, as the interpreter requires. This is the
 * trust layer; nothing is shown as shippable until it matches.
 */
import * as tf from '@tensorflow/tfjs';
import { verify, type Model, type VerifyReport } from 'tensorflow-web';
import { useInterpreter } from './useInterpreter';

export function useVerifier() {
  const interpreter = useInterpreter();

  /**
   * Runs verify over labelled test data.
   *
   * @param bytes the emitted .tflite.
   * @param reference the float model the emitted file must match.
   * @param testData held out inputs and one hot labels.
   * @param tolerance max absolute output error allowed for parity.
   * @returns the verify report.
   * @throws if the interpreter global is missing or the bytes fail to load.
   */
  async function run(
    bytes: Uint8Array,
    reference: Model,
    testData: { xs: tf.Tensor; ys: tf.Tensor },
    tolerance = 0.05,
  ): Promise<VerifyReport> {
    if (!interpreter.ready()) {
      throw new Error(
        'The TFLite interpreter is not loaded, so parity cannot be checked. Confirm the CDN scripts in index.html are reachable.',
      );
    }
    interpreter.ensureWasmPath();
    return verify(bytes, reference, testData, { tolerance });
  }

  return { run };
}
